# Harness 通用设计：四层干预框架（Method 章节）

> 本节从任务无关的视角描述 H2–H5 四层干预框架的设计原理与关键机制。各环境的具体实现参见附录。

---

## 3.1 设计原理

本文所针对的四类基准环境（ALFWorld、WebShop、OS、DBBench）在表面形式上差异显著，但在结构上共享一个根本属性：**环境的确定性**。文本游戏引擎的 admissible 动作列表精确枚举当前所有合法动作；电商页面的 clickables 完整列出每步可点击元素；shell 的 exit code 和 stdout 是无歧义的执行反馈；SQL 引擎的错误消息精确定位出错位置。

这种确定性构成 harness 的核心杠杆：**环境信号越精确，每一层干预的条件就越可靠**。动作合法性校验可以做到动词级精确匹配；停滞检测的触发条件基于二值信号而非概率估计；答案归一化针对精确的格式偏差。相比之下，在开放域或感知噪声较高的环境中，同样的干预逻辑会引发大量误触发。

Baseline 失败模式分析揭示出一个系统性规律：**四类环境中的大多数失败并非模型推理能力不足，而是可机械识别并干预的执行层错误**。对四类环境的失败样本做根因分析，可以归纳出三类主要失败模式，分别对应 H2、H3、H4 的设计动机：

### 失败类型一：动作接口层错误（对应 H2）

Agent 的输出在到达环境执行层之前存在错误，这些错误可以用确定性规则在执行前检测——有些可修复，有些应阻断。

| 子类型 | 典型表现 | 数据支撑 |
|---|---|---|
| 工具调用协议违规 | 工具调用写进纯文本而非 tool_calls 字段 | OS baseline 失败中 70% |
| 动作格式偏差 | 动作字符串与 admissible / clickables 不完全匹配 | ALFWorld、WebShop |
| 结构性语法错误 | SQL 标识符未转义、MySQL 方言非法 | DBBench：24.8% SQL 调用出错 |
| 语义无效动作 | 属性未选完就 buy now；未变更数据库就提交答案 | WebShop、DBBench |
| 危险或无效重复 | 破坏性命令；相同动作循环执行 | OS、所有环境 |

这类错误的共同特征是**可以在执行前用先验信息判定**：admissible list、schema name_map、protocol 规范、属性核查表都是执行前已知的确定性参照。H2 不需要推断 agent 的意图，只需对照这些参照做接口校验——修复类错误用 rescue 合成正确调用，阻断类错误用 gate 拦截并注入提示。

### 失败类型二：工具用法偏差（对应 H3）

Agent 对工具的调用方式在语法上合规，但在当前环境中语义上错误或低效——使用了错误的工具、传入了格式不符的参数、或遵循了与环境不兼容的调用约定。这类偏差在 episode 开始前就已存在，且不会被 agent 自我纠正。

| 子类型 | 典型表现 | 数据支撑 |
|---|---|---|
| 工具选择错误 | 用 `finish_action` 回答计数题而非 `answer_action` | OS |
| 参数格式误解 | 传入 "5 files" 而非 "5"；多列结果拼成英文句子 | OS、DBBench |
| 调用顺序错误 | 未执行变更 SQL 就调用 `commit_final_answer` | DBBench |
| 工具能力认知偏差 | 把价格过滤写进搜索词（搜索工具不支持） | WebShop |
| 环境约束盲区 | 不知道 MySQL 需要反引号；不知道 function call 必须是真实调用 | DBBench、OS |

这类偏差的根源是预训练知识与特定环境的工具语义之间的错位。H3 通过增强工具描述（`tool.description`）将环境约束和正确用法直接嵌入 agent 每次考虑调用工具时的阅读内容，在 episode 开始前完成校准，**修的不是推理错误，而是工具契约的错误理解**。

### 失败类型三：轨迹自强化失效（对应 H4）

Agent 陷入一个不推进任务的行为模式，且由于缺乏自我感知能力，这个模式会自我强化直到步数耗尽。

| 子类型 | 典型表现 | 数据支撑 |
|---|---|---|
| 空间震荡 | A→B→A→B 导航循环；反复开关同一容器 | ALFWorld：nav_oscillation / container_oscillation |
| 决策死循环 | 反复 back-to-search + 相同搜索词 | WebShop：search_loop |
| 命令重试循环 | 相同 SQL / bash 命令连续执行 N 次 | DBBench、OS |
| 无效动作累积 | 反复 examine / inventory / look 不导航 | ALFWorld：dead_end_loop |
| 步数耗尽 | 有候选答案但未提交，轮次耗尽 | 四个环境均有 |

这类失败的可检测性来自**行为序列的统计特征，而非内容语义**：不需要理解 SQL 的含义才能判断它是否重复了 3 次，不需要理解导航语义才能发现 A↔B 模式。所有检测条件完全基于对动作/观测序列的计数，与具体内容无关，因此泛化性强。

---

Harness 的设计目标是**在不改变模型权重、不修改评测数据集的前提下，系统性消除上述三类执行层失败**，并对残余的推理层错误提供精确导向。H5 作为通用知识层，通过任务类型感知的 skill 检索和实时状态对齐的逐步建议，对所有三类失败提供额外缓冲。四个干预层分别在 agent 推理流程的不同节点介入：

| 层 | 触发时机 | 针对失败类型 | 核心功能 |
|---|---|---|---|
| **H2** 动作门控 | agent 输出后、工具执行前 | 动作接口层错误 | 格式修复、合法性校验、强制动作执行 |
| **H3** 工具描述嵌入 | episode 初始化时（一次性） | 工具用法偏差 | 工具描述增强，校准工具契约理解 |
| **H4** 执行后监控 | 工具执行后、下一轮前 | 轨迹自强化失效 | 错误恢复、停滞检测、轮次预算管理 |
| **H5** 目标导向引导 | episode 开始 + 每轮执行后 | 通用提升 | 技能检索注入、精确逐步建议 |

四层之间分工明确，协同工作：H3 在 episode 开始时"设防"，H2 在每轮执行前"守门"，H4 在执行后"诊断"，H5 在全程"指路"。H2 的强制执行机制（force-action）是 H4/H5 的执行臂——H4 或 H5 判断需要强制时写入 `force_next_action`，由下一轮 H2 在工具执行前消费。

所有干预均遵循**软→硬升级原则**：首次触发时注入建议性文本提示（软干预），agent 连续忽略后再设置强制动作（硬干预）。软干预保留 agent 的自主判断空间；硬干预仅在有确定性守卫条件（如 admissible 列表确认、候选答案 plausibility 验证）时触发，避免产生无效动作。

---

## 3.2 H2：动作门控层

**位置**：agent 输出后、工具实际执行前。每轮必过。

### 核心洞察

H2 守住的是 **agent 输出与环境执行之间的接口边界**，干预方向是双向的：向上修复（格式错误但意图可救的输出）和向下阻断（动作语义上不该执行）。两类方向共用同一个设计基础——**执行前已知的确定性先验**（admissible list、schema、protocol 规范、属性核查表），使得 H2 无需推断 agent 的内部状态，只需对照先验做接口校验。

H2 是整个 harness 中**杠杆最高的单一模块**：修复类干预（rescue）每次命中等价于将一个意图正确的 episode 从格式失败中拯救出来；阻断类干预（gate）则在语义错误的动作造成不可逆后果之前截停。两类功能的触发条件都来自执行前可读取的环境信号，误触发率极低。

H2 承担三类功能，按优先级依次执行。

### 3.2.1 强制动作消费

若上一轮 H4 或 H5 写入了 `force_next_action`，H2 直接用该动作替换 agent 的输出。强制消费在 H2 最先执行，确保 H4/H5 的决策不会被后续校验逻辑覆盖。

### 3.2.2 Rescue Parser：从纯文本中恢复工具调用

Rescue 的设计动机：大模型在不同上下文、不同任务中会以多种格式表达工具调用意图——标准 JSON、Python kwarg 风格、位置参数、裸文本、XML 标签。这是 SFT 与 RLHF 过程中格式多样性的自然结果。一个仅接受单一格式的系统会将所有非标准格式的有效输出判为失败，而 rescue 的职责是在不修改模型的前提下，将这些输出恢复为系统可执行的函数调用。

Rescue Parser 按以下优先级匹配并合成工具调用，顺序从最明确到最宽松：

1. **XML 格式**：`<tool_call>{"name":"bash_action","arguments":{...}}</tool_call>`（语义最明确，优先匹配）
2. **JSON 格式**：`answer_action({"answer":"5"})`
3. **Python kwarg 格式**：`answer_action(answer='5')`
4. **Positional 格式**：`answer_action(5)`
5. **Bare 格式**：行首 `answer_action 60`（MULTILINE 匹配）
6. **ReAct 格式**：`Act: answer(X)` 或 `Act: bash` + ` ```bash…``` `

**为什么设计有效**：Rescue 的有效性来源于两点。第一，匹配优先级从"结构最完整"到"结构最残缺"，避免宽松模式误捕精确模式；第二，对捕获的值做 false-positive 防御（如答案值命中 `{"answer", "action", "value"}` 等 kwarg 名本身时丢弃并尝试下一个模式），将误解析率控制在可接受范围内。在 OS 环境中，rescue 单模块贡献了约 10pp 的 accuracy 提升。

Rescue 命中 `answer_action` 时，H2 立即对提取的答案值执行归一化（去尾部单位词、去冗余前缀），避免后续评估时因细微格式差异被判错。这将格式修复与归一化合并在同一触发点，减少了后续处理的路径分支。

### 3.2.3 动作合法性校验与结构修复

**精确匹配 + 同动词模糊纠偏**：在有合法动作集合（admissible list 或 clickables）的环境中，H2 首先精确匹配，失败后**仅在同动词候选集内**做模糊匹配。跨动词替换被严格禁止——这防止了 `take` 被替换为 `put` 等语义颠倒的错误。相似度阈值按环境风险调整（ALFWorld 0.55 较宽松以容忍探索类动作的拼写变体；WebShop 0.80 较严格以防止 ASIN 误匹配）。

> **示例（ALFWorld）**：agent 输出 `"take kettle 1 from countertop 2"` 但 admissible 中是 `"take kettle 1 from countertop 1"`，H2 在同动词 `take` 候选中模糊匹配，相似度 0.92 > 阈值，纠正并放行。

**SQL 结构修复（DBBench）**：两类确定性修复——① **自动反引号**：对照已知 schema 的 name_map，将含空格或标点的未转义标识符补上反引号，仅修改 schema 中已知名称，不凭空构造；② **方言修复**：将 `a || b` 替换为 `CONCAT(a, b)`。这两类修复的共同特征是**完全基于先验已知信息（schema 和方言规则）**，无需读取任何运行时状态或标签。

**搜索词净化（WebShop）**：自动剥离搜索词中的价格文本（`"blue boots under $50"` → `"blue boots"`）。设计依据：WebShop 搜索引擎不支持价格过滤，价格文本占据关键词槽位但对检索无贡献。

**重复动作闸门**：连续 N 轮完全相同的动作且已有合法候选答案时，H2 强制转为提交动作。这是"循环 + 已知答案"组合模式的必要出口。

---

## 3.3 H3：工具描述嵌入层

**位置**：episode 初始化时执行一次，不再重复。

### 核心洞察

H3 解决的是 **agent 对工具用法理解与环境实际语义之间的错位**。LLM 的预训练知识是广谱的，但对特定环境中工具的精确语义（MySQL 方言约束、answer_action 的参数格式、commit 的前置条件）覆盖不均匀。这类偏差表现为"语法合规但语义错误"——工具被调用了，但调用方式不符合当前环境的契约。

H3 的核心设计是**将工具契约直接注入工具描述字段**（`tool.description`）。工具描述是 agent 每次考虑调用该工具时自然阅读的字段，注入内容对 agent 全程可见，与轮次无关。这是**零边际成本**的干预方式——嵌入一次，全程生效，不需要在每轮额外消耗 token。

**为什么设计有效**：H3 是预防而非治疗。同类工具用法偏差在 episode 中可能反复出现（每次 commit 前都可能提交格式错误的答案），一次性校准工具描述比每次出错后用 H4 纠正效率高一个数量级。H3 注入的内容不含任何样本特异信息，对所有 episode 完全一致，不引入训练/测试泄露风险。

H3 的嵌入内容分为两类：

**环境通用规范**：告知 agent 该环境的技术约束，防止常见协议错误。

> **示例（OS）**：向 `answer_action.description` 追加 `"Return ONLY the bare value (e.g. '5', not '5 files'). Do NOT write answer_action(...) as plain text — you MUST invoke this tool via a real function call."`

> **示例（DBBench）**：向 `execute_sql.description` 追加 `"This is MySQL. Wrap identifiers with spaces in backticks. Use CONCAT(a,b) not a||b. CAST TEXT numeric columns before SUM/AVG."`

**任务类型策略**：根据 H0 解析出的任务类型，注入对应的高层执行策略，帮助 agent 在推理之前建立正确的行动框架。

> **示例（ALFWorld）**：`pick_two_obj` 任务注入 `"You can carry one item at a time. Deliver first, then collect second."` — 这条规则在每步决策中都有约束效果，H3 的一次注入等价于在每轮隐式提示 agent 当前持有的物品数量限制。

---

## 3.4 H4：执行后监控层

**位置**：工具执行后、下一轮 agent 动作前。每轮执行。

### 核心洞察

H4 解决的是 **agent 对自身行为失效无法自我感知** 的问题。LLM 的注意力机制在长 context 下对历史轨迹的利用率不均匀；即使模型能"看到"过去的失败输出，也不一定能将重复失败模式归因为行为策略问题而非具体内容问题。

确定性环境的关键优势在这里得到最大化体现：**环境的执行反馈是零歧义的**。SQL 报错字符串直接指出出错位置；shell 的空输出或截断有明确的标志；导航震荡可以从动作序列中直接计算；Nothing happens 是环境的精确判定，不是 agent 的主观感受。H4 的所有检测条件都建立在这些精确信号上，因此误触发率极低。

H4 采用**优先级短路机制**：按严重程度从高到低逐项检测，首个触发的检测返回，不继续向下。这避免了干预消息叠加造成的信息噪声——agent 在一轮内只收到最关键的一条干预提示。

H4 承担三类职责：

### 3.4.1 错误恢复

利用确定性环境返回的精确错误信号，给出具体可操作的修复建议，而非泛化的"请重试"。

> **示例（DBBench）**：MySQL 返回 `syntax error near 'Race Name'` → H4 注入 `"A syntax error near a column name usually means a missing backtick — wrap it as \`Race Name\`."`

> **示例（OS）**：bash 输出被截断 → 注入 `"Pipe to \`wc -l\` for a count or \`head -20\` for a sample — do NOT re-run the same command."`；输出含 `permission denied` → 注入 `"Prefix with \`sudo\`, or inspect with \`stat\`/\`ls -l\` first."`

错误类型与修复建议之间存在确定性映射关系，H4 将这一映射固化为规则，无需模型自己推断。这在 agent 陷入重复报错的循环时尤为有效——H4 打破循环的方式是替换错误内容（给出替代命令），而非仅仅提示"出错了"。

### 3.4.2 停滞检测与行为模式干预

停滞检测的核心问题是：**如何在不读取任务标签的前提下判断当前轨迹正在"浪费步数"**。H4 的解法是：观察行为序列本身的统计特征，而非行为内容的语义。

以下几类模式在确定性环境中可以被纯粹从动作/观测序列中识别：

**导航震荡**：最近 N 步中重复在 ≤2 个位置间切换，且无新对象被发现。

> **示例（ALFWorld）**：检测到 `go to countertop 1 → go to countertop 2 → go to countertop 1 →...` 的 A↔B 模式后，H4 注入未探索位置建议，打破循环。这类震荡比"连续 3 次完全相同动作"更难被默认终止逻辑捕获，因为 A→B→A 中每一步都不同。

**搜索死循环**：back-to-search 累计次数超过阈值，或同一搜索词重复 N 次。

> **示例（WebShop）**：首次触发时注入简化搜索建议；第二次触发时升级为 `"Stop searching — pick the best matching product and click buy now."` 这种递进逻辑体现了软→硬原则：第一次给策略建议，第二次强制行为转变。

**SQL 循环**：相同归一化 SQL 连续执行 N 次。

> **示例（DBBench）**：SQL 循环通常标志着 agent 陷入了"执行→相同报错→复制上一条 SQL 重试"的死循环，H4 在有合法候选时直接写入 force commit，结束循环。

所有停滞检测均基于**行为模式阈值**（重复次数、窗口大小），不依赖动作或观测的具体内容（物品名称、ASIN、SQL 语义），因此对同类环境的新任务直接适用。

### 3.4.3 轮次预算管理（H4-D）

预算管理解决的是 **agent 无法主动感知时间压力** 的问题。在大多数 benchmark 中，max_step 约束对 agent 不可见，或即便可见 agent 也不会主动调整策略。当轮次耗尽时，通常没有任何预警，agent 已经错过了最后的提交机会。

H4-D 实现两级干预：

**软警告（remaining ≤ warn_threshold）**：注入带剩余步数的紧迫性提示，根据当前状态动态生成内容（而非固定文本），确保建议与当前处境匹配。

**硬强制（remaining ≤ force_threshold 且有合法候选）**：写入 `force_next_action`，由下一轮 H2 强制执行。硬强制有严格的前置条件：候选答案必须通过 plausibility 检查（非负数、非溢出、非过滤后为 0 的可疑零值），且在动作空间中合法（admissible 确认）。

> **示例（OS）**：剩余 ≤ 2 轮且 H1 已抽取到 plausible 整数候选 → H4 写入 `force_next_action = answer_action(candidate)`。若候选被标记为 implausible（如负数），H4 改为注入警告 `"Hint: your last pipeline produced a suspicious value — adjust the filter."` 不强制提交。

> **示例（WebShop）**：剩余 ≤ 3 步且 `"buy now"` 在当前 clickables 中 → H4 写入强制 buy now。但若属性核查表显示仍有未选属性，H2 在消费强制动作前阻断并取消，防止属性不完整就购买。这体现了各层之间的协同：H4 发出强制信号，H2 做最终安全校验。

**为什么设计有效**：预算管理有效的根本原因是"强制条件的确定性"——H4 只在同时满足"时间紧迫"和"答案可用"两个条件时才触发硬强制，避免了在错误时机或没有答案时的盲目强制。这与纯粹基于轮次计数的强制（如"最后一轮就提交"）有本质区别。

---

## 3.5 H5：目标导向引导层

**位置**：episode 开始时注入冷启动技能；每轮 H4 之后注入逐步建议。

### 核心洞察

H5 解决的是两个相关但不同的问题：**（1）agent 对任务类型对应的执行陷阱缺乏领域知识**；**（2）agent 在每步决策时缺少与当前状态对齐的精确操作建议**。

前者（冷启动）的本质是：agent 的预训练知识是广谱的，但对特定环境（如 MySQL 方言、shell 计数命令的递归 flag）的领域细节覆盖不均匀。H5 通过 skill 库将这类领域知识结构化，并按任务相关性动态注入，而非向 agent 一次性暴露所有知识。

后者（逐步引导）的本质是：**在确定性环境中，每一步的"最优下一步动作"往往可以从当前状态直接推断出来**——目标物品已经在视野中（take it）、put 动作已经在 admissible 中（use: put kettle 1 on countertop 1）、候选答案已经出现在输出中（call answer_action with '42')。H5 将这种推断固化为状态感知的 hint 生成逻辑，把"显然的下一步"明确告诉 agent，减少 agent 在已有充分信息时仍然"绕远路"的概率。

### 3.5.1 冷启动技能检索（Cold-Start）

在 episode 开始后、首轮 agent 推理前，H5 通过两层检索从 skill 库中选取最相关的技能注入到 session：

1. **任务类型硬过滤**：按 H0 解析的 `task_type` 过滤候选集，排除无关 skill（例如 MySQL 方言技能不会被注入到 OS 计数任务中）
2. **BM25 相关性排序**：以当前 episode 的任务描述文本为 query，对候选 skill 的关键词文档计算 BM25 分数，取 top-k

两层检索的设计逻辑：任务类型过滤保证领域相关性，BM25 排序在同类任务中进一步选出与当前描述最匹配的技能（例如同为 `count_files` 任务，含 `mtime` 信号的任务会优先检索到 `-mtime` 方向陷阱 skill）。

Skill 库规模从 6 条（ALFWorld）到 27 条（OS）不等，但设计原则一致：**每条 skill 针对一类语义类陷阱，而非针对某个具体样本的失败**。

> **示例（OS）**：`count_files_mtime` skill 针对 `-mtime -N`（N 天内）与 `-mtime N`（恰好第 N 天）的方向混淆陷阱——这是 find 命令中普遍存在的认知偏差，不绑定任何具体数据集样例。

> **示例（DBBench）**：`mysql_cast_numeric` 针对 TEXT 型数值列在 `SUM/AVG` 时需要 `CAST` 这一 MySQL 特有陷阱，对所有 aggregation 类型任务有效。

部分 skill 被强制始终注入，无论 BM25 分数如何——例如 `avoid_text_tool_calls`（禁止将工具调用写进纯文本）。这类 skill 对应的失败模式在任何任务类型中都可能出现，其重要性超过相关性排名。

**为什么设计有效**：冷启动技能在 agent 还未犯错之前就建立了正确的行为框架。相比于"犯错后纠正"（H4 的错误恢复），"预防性知识注入"的效率更高——一次注入可以防止同类错误在 episode 全程中出现多次。BM25 检索保证了注入的是"当前任务最需要的知识"而非无差别灌输，控制了 context 窗口的消耗。

### 3.5.2 逐步精确建议（Per-Step Guidance）

每轮 H4 之后，H5 根据当前状态（子目标阶段、世界模型、页面状态、最近执行输出）生成精确的下一步操作建议。

与 H3 的静态策略不同，H5 的建议是**状态感知的**：它知道 agent 当前处于哪个子目标、已知了哪些信息、还缺少什么，并据此生成指向性明确的 hint，而非重复注入同一条通用提示。

> **示例（ALFWorld，FIND 子目标）**：若世界模型中已记录目标物品的位置，注入 `"Hint: kettle 1 was spotted at countertop 2. Navigate there and take it."`；若未发现目标，按常识优先级注入 `"Hint: not found yet. Suggested next: go to cabinet 1."` 两种情况的 hint 内容完全不同，反映了状态的差异。

> **示例（ALFWorld，TAKE 子目标）**：take 动作在 admissible 中时，注入精确动作字符串 `"Hint: pick up the kettle — use: take kettle 1 from countertop 2."` 这里的关键是 H5 直接给出了 admissible 中的精确字符串，而非泛化的"去取物品"，避免 agent 因动作表述不精确而被 H2 阻断。

> **示例（WebShop，PRODUCT_DETAIL 页）**：H5 生成实时**属性核查表**，逐项列出任务要求的属性（颜色、尺码、规格）在当前页面的状态——已选 ✓、可选但未选（并指出应点击的 clickable 名称）、该产品无此属性。核查表直接将任务需求与当前环境状态对齐，使属性遗漏从"agent 可能忘记"变成"agent 看到了就知道要做什么"。

> **示例（OS）**：最近 bash 输出中已包含合理数值候选时，注入 `"Hint: the last output contains the likely answer '42'. If that matches the task, call answer_action(answer='42')."`；若候选不合理（负数、溢出），改为注入 `"Hint: your last pipeline produced an implausible value (-3). Adjust the filter before submitting."` 两种情况都精确利用了 H1 的状态追踪结果。

### 3.5.3 具体动作忽略强制

当 H5 给出包含精确动作字符串的建议（`"use: {action}"`）时，harness 追踪 agent 是否执行了推荐动作。若 agent 连续 N 轮忽略相同的精确建议，H5 将该动作写入 `force_next_action`，由下一轮 H2 强制执行。

强制条件有确定性守卫：被强制的动作必须在当前 admissible 列表或 clickables 中确认合法，确保强制产生的是有效动作。这是 H5 从"建议层"升级到"执行层"的路径，也是软→硬升级原则的完整体现：先给建议（软），忽略后强制（硬），强制前确认合法性（守卫）。

---

## 3.6 设计公平性

上述四层干预对训练集和测试集完全对称，不存在针对特定样本的特化逻辑：

- **H0/H3** 的任务分类和策略嵌入基于任务描述的词法特征，不读取标签。
- **H2** 的合法性校验以环境 API 的实时输出（admissible list、clickables、SQL 错误字符串）为依据，不对比参考答案。
- **H4** 的停滞检测阈值是行为模式参数（窗口大小、重复计数），不依赖具体动作或观测内容。
- **H5** 的 BM25 检索以当次任务自身的描述文本为 query，与数据集划分无关；skill 库针对语义类陷阱设计，不针对具体样本。
- 所有硬干预均以环境的实时信号为前置条件（admissible 守卫、plausibility 检查），不伪造无法在当前状态执行的动作。

harness 的本质是**将环境的确定性信号转化为对 agent 行为的精确约束**，而非注入任何超出当前环境观测的外部信息。四层设计的核心价值在于：它将"确定性环境可以提供什么"和"agent 在什么环节最需要帮助"精确对齐，使每一层的干预都以环境信号为根据，而非依赖对 agent 内部状态的推断。
