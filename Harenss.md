# Harness 通用设计：四层干预框架（Method 章节）

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
| **H5** 目标导向引导 | episode 开 | 通用提升 | 技能检索注入 |

四层之间分工明确，协同工作：H3 在 episode 开始时"设防"，H2 在每轮执行前"守门"，H4 在执行后"诊断"，H5 注入经验。

## 3.2 需要保障的性质（包括但不限于）
根据上面提到的设计原理，我们需要重点观察：
- 任务中哪一层的harness效果最好，是否有没用/副作用的harness？我们期望的是不同层级互相合作，都有一定作用。
- harness之间是否有重复冗余，甚至冲突？如果有需要调整到合适的层级，不同层级应该是独立，但又可以有一点协作关系。
- harness分级是否符合上述的设计原理？在符合设计原理的前提下，根据实验结果，尽可能逐层优化harness以进一步提升性能，可以两两对比来看，例如w/ h2和w/o h2即可查看h2效果。
- 除了指标外，你需要挑选具有代表性的轨迹进行完整分析。harness应当具有一定鲁棒性，因为我们是在训练集开发harness，最终要到测试集测试的。

## 3.3 OS 实验结果分析（2026-04-26，v5 → v6）

### 三组对比实验

| 实验 | H2 | H3 | H4 | H5 | acc |
|---|---|---|---|---|---|
| harness（v5）| ✓ | ✓ | ✓ | ✓ | 0.59 |
| no-H4 | ✓ | ✓ | ✗ | ✓ | 0.57 |
| no-H5 | ✓ | ✓ | ✓ | ✗ | **0.64** |

关键发现：**关闭 H5 后准确率反而上升 +5%**，说明 v5 的 H5 存在净负效应。H4 则有小幅净正效应（+2%）。

### H5 根因分析（净 -5）

**回退 13 案例 vs 帮助 8 案例**

1. **`home_dir_file_glob` 关键词过宽**（7 个回退）：keywords 包含 "home"/"txt"/"text"/"files" 等通用词，导致几乎所有 home-dir 文件任务都会注入该 skill，而真正的错误（如缺少 `-i` flag）无法被覆盖。H4 的 `no_such_file` 恢复已能响应式处理 glob 错误。

2. **"Ignore case sensitivity" 未被解析**（3 个回退：IDX 31/83/809）：`_CASE_INSENS_SIGNALS` 只包含 "ignoring case"，不匹配 "Ignore case sensitivity" 写法，导致 `case_sensitive=None` 而非 `False`，`case_insensitive_hint` 无法被 context-driven override 强制注入。

3. **低分注入无阈值**（2 个回退：IDX 426/663，BM25=5.36）：无关 skill 在 top-1 条件下依然注入。

### H4 根因分析（净 +2，7 个回退中部分为噪声）

**真实 bug**：
- IDX 245/425：zero-nudge 中 "run `ls` to verify" 指令在答案真为 0 时触发级联探索（mtime filter 任务），最终提交错误非零答案
- IDX 663：`find: warning:` 单行输出未被 `_ERROR_RE` 识别为错误，`candidate_string_answer` 被设为警告文本后被提示提交
- IDX 788：`find: invalid mode` 未被 `_ERROR_RE` 捕获，zero-nudge 触发而非错误恢复路径
- IDX 952：round-1（第二次 bash）得到候选数字后直接提示 "submit it"，agent 提交了错误中间结果

**评估噪声**：部分回退案例（IDX 297/402 等）中相同命令在不同 docker 实例中返回不同结果（temperature=0.7 + 独立容器），实为评估噪声而非 harness 逻辑问题。

### v6 修复措施

**H5 修复**（预期将净效益从 -5 转正）：
1. `h5_score_threshold=6.0`：低于阈值的 skill 不注入，屏蔽 IDX 426/663 类低质注入
2. context-driven override：`case_sensitive is False` 时强制将 `case_insensitive_hint` 注入为 top-1，覆盖 BM25 排名更高但无关的 skill（修复 IDX 31/83/809）
3. `_CASE_INSENS_SIGNALS` 增加 "ignore case" + regex fallback，覆盖 "Ignore case sensitivity" 等写法
4. `home_dir_file_glob` 关键词缩减至 ["glob","wildcard","star","asterisk"]，避免过度匹配

**H4 修复**：
1. `_ERROR_RE` 新增 "invalid mode"/"invalid option"/"find: warning:"/"grep: warning:" 等，覆盖 IDX 788/663 错误类型
2. zero-nudge 去除 "run `ls`" 指令，改为软提示（"if correct, 0 is valid"），防止 IDX 245 类级联错误探索
3. `candidate_string_answer` 增加 length guard（≤120 chars）和 tool-prefix 过滤（find:/grep: 等），防止 IDX 663 类工具警告被提示为答案
4. 验证窗口从 round=0 延伸至 round≤1（前两次 bash 都触发 "verify before committing"），防止 IDX 952 类 round-1 提前提交

---

## 3.4 WebShop 实验结果分析（2026-04-26，v6 初版）

### 三组对比实验

| 实验 | H2 | H3 | H4 | H5 | success@1 | avg_reward |
|---|---|---|---|---|---|---|
| no-H2 | ✗ | ✓ | ✓ | ✓ | **36** | **0.688** |
| no-H3 | ✓ | ✗ | ✓ | ✓ | **36** | 0.680 |
| harness（全开）| ✓ | ✓ | ✓ | ✓ | 34 | 0.665 |

关键发现：**H2 和 H3 均为净负效应**。harness 全开反而最差；关闭 H2 后性能最优（avg_reward +0.023）。H2 有 10 个 reward 回退（5 个 success 回退），H3 有 8 个 reward 回退（4 个 success 回退）。

### H2 根因分析

**IDX 4374（62 步死循环，reward=0 vs no-H2=1）**：  
`_instruction_grounded_matches` 把 "banana + almond butter" 和 "peanut butter" 标记为 IG 要求，原因是两者都含 "butter"，与任务关键词 "sunflower **butter** and chocolate protein bars" 发生误匹配。agent 已正确选中 "sunflower butter + chocolate"，但 IG 检查仍反复 block buy-now（"please click 'banana + almond butter'"）。IG block 无每 episode 上限，导致 60 步死循环直到 task_limit_reached。

**IDX 9096（21 步，reward=0.667 vs no-H2=1）**：  
product page 有 4 个 color 选项：matte black、metallic blue、metallic green、**metallic gun metal**。`extract_attribute_options` 将前三个分到 "color" 组，但 "metallic gun metal" 因无 word-boundary color token 被归入 "other"。agent 选中 "metallic gun metal" 后，H2 defensive guard 发现 "color" 组有选项但未选中任何 → 触发 block（"select a color"）。agent 受 repeat-click gate 阻止（已点击同一选项两次），无法重选 metallic gun metal，只能点其他颜色 "metallic green" → 以错误颜色购买。

**IDX 6548（reward=0.5 vs no-H3=1）**、**IDX 8209（同）**：  
nail polish "c08" 和 brush "greenwithoutlogo" 均位于 page 的 "color" 区域，但 `extract_attribute_options` 把它们归为 "other"（无 color/size token 匹配）。IG check 无法匹配（短字符串/无 word-boundary overlap），defensive guard 只检查 "color"/"size" 组（均为空）→ buy-now 不被 block → agent 未选属性就购买，得到 partial reward。

**IDX 6530（noise）**：两次运行选了不同产品（stochastic），与 H2 逻辑无关。

### H3 根因分析

大多数 H3 回退（IDX 6548/8209/6237）根本原因是 H2 未拦截无属性 buy-now：H3 click_action 描述末尾 "then click 'buy now'" 产生 last-instruction bias，小模型跳过属性选择直接购买；但若 H2 正确 block，影响可被消除。IDX 3953 为 stochastic noise（两次运行恰好选了不同产品）。

H3 描述中 "then click 'buy now'" 位于末尾，在 small model 上等价于"click 'buy now'"的直接指令（last-instruction bias）。

### v6 修复措施

**H2 修复**（核心）：

1. **观测文本 section 解析**（修复 IDX 9096/6548/8209）：新增 `_extract_attr_with_obs(observation, clickables)`，优先从 observation 的 `[SEP]`-delimited 区段头（"color"/"size"/"flavor name" 等）判定选项分组，而非单纯依赖 token 匹配。"metallic gun metal"、"c08"、"greenwithoutlogo" 等原先被错分至 "other" 的选项，现在通过观测文本中的 `color [SEP]` 头正确归入 "color" 组。

2. **IG 关键词覆盖过滤**（修复 IDX 4374）：在 `_buy_now_precheck` 的 IG check 后增加 coverage filter：若 IG option 的所有 meaningful overlap 关键词已存在于当前已选属性中，则跳过该 IG option。"banana + almond butter" 的 overlap 关键词 "butter" 已被 "sunflower butter + chocolate"（已选）覆盖 → 从 ig_matches 中过滤 → buy-now 不再 block。

3. **`_total_buy_now_blocks` 安全上限**（safety net）：每 episode 最多 block buy-now 4 次，超过后强制放行，防止任何原因导致的无限循环。

4. **defensive guard 扩展**：当 agent 在当前 product page 尚未选中任何属性时，defensive guard 也检查 "other" 组的选项（而非仅 "color"/"size"），捕获通过观测解析仍无法识别的 unclassified 选项。

**H3 修复**：

将 click_action 描述改为先强调属性选择，消除 last-instruction bias：  
旧："...select ALL required attributes... then click 'buy now'."  
新："On a product page, select ALL required attributes... BEFORE clicking 'buy now'. Check each option group..."

**H5 修复**：
1. 新增 `h5_score_threshold=3.0`（WebShop skill 库较小，阈值低于 OS 的 6.0），防止低相关 skill 注入。

---

## 3.5 WebShop 实验结果分析（2026-04-26，v6 第二轮）

### 三组对比实验

| 实验 | H2 | H3 | H4 | H5 | success@1 | avg_reward | avg_steps |
|---|---|---|---|---|---|---|---|
| no-H2 | ✗ | ✓ | ✓ | ✓ | **42** | **0.7131** | 12.6 |
| no-H3 | ✓ | ✗ | ✓ | ✓ | 41 | 0.7006 | 13.7 |
| harness（全开）| ✓ | ✓ | ✓ | ✓ | **42** | 0.7098 | 14.0 |

关键发现：**v6 修复后 H2 基本达到中性**（success@1 与 no-H2 持平，avg_reward 仅差 -0.003），**H3 显示小幅正效应**（去掉 H3 后 success@1 -1，avg_reward -0.009）。代价是 H2 引入额外步数开销（+1.4 steps vs no-H2），说明仍有少量无效 block 残余。

### H2 残余回退根因分析（第二轮）

第二轮新增 H2 回退案例（与 no-H2 对比，reward 下降）集中在以下两类：

**防御性 block 误触发（IDX 1447）**：  
agent 正确选中 scent 选项 "rainforest"（归属 "other" 组），但防御性 guard 仍对 "size" 组触发（"size" 组有选项列表，但 `selected_attributes["size"]=""` ）。根因：`any_selected` 检查（`any(v for v in state.selected_attributes.values())`）为 True（rainforest 已选），但 `any_selected` 只保护了 "other" 扩展路径，未保护 color/size 的循环检查，导致 size 组假阳性 block → agent 被迫选择无关 size 选项 "get naked" → 以错误属性组合购买。

**其他案例（IDX 4880/6743/9542）**：分析后确认为随机噪声（不同 episode、temperature=0.7 下模型选择了不同产品路径），与 H2 逻辑无关。

### H3 残余回退根因分析（第二轮）

IDX 3335/3953 两个 H3 回退案例：不同 episode 与 no-H3 对比恰好选择了不同候选产品，属于 stochastic 噪声，非 H3 逻辑问题。

### v6 第二轮修复

**修复 IDX 1447 防御性 block 误触发**：  
将 `any_selected` 保护范围扩展至整个防御性 guard（不仅是 "other" 扩展），**当 agent 在当前页已选中任何属性时，跳过全部防御性检查**：

```python
# 修复前：any_selected 只保护 "other" 扩展，color/size 循环仍无条件触发
if not to_click and self._defensive_buy_blocks < 2:
    any_selected = any(v for v in state.selected_attributes.values())
    for group in ("color", "size"):   # 即使 any_selected=True 也会执行
        ...
    if not any_selected:  # 仅保护 "other"
        ...

# 修复后：整个防御性 guard 被 any_selected 保护
if not to_click and self._defensive_buy_blocks < 2:
    any_selected = any(v for v in state.selected_attributes.values())
    if not any_selected:   # 完全未选才触发防御
        for group in ("color", "size"):
            ...
        other_def_opts = ...
        if other_def_opts and not sel:
            ...
```

设计原理：防御性 guard 的目的是捕获"agent 完全未选任何属性就尝试 buy-now"的情况。一旦 agent 选中了任何一个选项，说明它已识别到当前 product page 的属性结构并作出了合理选择，不应再被 guard 干扰。
---

## 3.6 WebShop 实验结果分析（2026-04-26，v6 第三轮）

### 三组对比实验

| 实验 | H2 | H3 | H4 | H5 | success@1 | avg_reward | avg_steps |
|---|---|---|---|---|---|---|---|
| harness（全开）| ✓ | ✓ | ✓ | ✓ | 40 | 0.6998 | 13.7 |
| no-H2 | ✗ | ✓ | ✓ | ✓ | 41 | 0.7046 | 12.3 |
| no-H3 | ✓ | ✗ | ✓ | ✓ | **42** | **0.7096** | 13.3 |

三轮累计结论：**H3 持续净负效应**（三轮均为负/中性），**H2 中性偏负**（回退案例经分析后部分为随机噪声，但有真实 bug）。

### H3 净负根因（最终定论）

H3 通过 `patch_webshop_tool_descriptions` 在 `search_action` 描述中追加 "Do NOT include price or budget words"，导致 agent **不在搜索词中包含价格**。无 H3 时，agent 包含价格，H2 strip 后结果略有不同（如保留 "under" 等词），两种搜索词在统计上导向不同的产品选择。由于 WebShop 搜索结果对关键词敏感，这种差异最终表现为随机但系统性的 reward 变化。H3 的 `click_action` 描述修改则几乎无可测量效果（对产品页行为已有 H4 step guidance 覆盖）。

**决策**：删除 H3（config 设 `h3: false`）。H3 的工具描述嵌入对 WebShop 无益，且会引入搜索词构造的系统性偏差。

### H2 第三轮回退根因分析

| IDX | harness | no-H2 | no-H3 | 根因 |
|---|---|---|---|---|
| 4161 | 0.75 | 1.00 | 0.75 | 随机噪声（不同 budget 上限：$60 vs $50）|
| 9519 | 0.50 | 1.00 | 1.00 | 随机噪声（不同任务 goal：$50 vs $40）|
| 4564 | 0.75 | — | 1.00 | H3 导致不同产品选择 → 选了过预算产品 → H4 price 循环 |
| 6789 | 0.50 | 0.75 | 0.75 | H3 改变搜索词 → 不同产品 → H2 IG 误报 |
| 1447 | 0.05 | 0.08 | — | 防御性 guard 误触发（已修复）|
| 4880/9542 | — | — | — | 随机噪声（与 H2 逻辑无关）|

**关键发现**：H2 的真实回退（非随机噪声）全部与 H3 引起的搜索词变化有关（H3 + H2 相互作用），或是已修复的 bug。**移除 H3 后，H2 的已知真实回退全部消除**。

### v6 第三轮修复

**1. 禁用 H3**（`h3: false`）：
- 根本原因：H3 搜索描述改变 agent 搜索词构造，引入系统性随机性，净效益三轮均为负
- 实施：config `webshop-env_train.h3: false`（之前实验的 no-H3 实际上已验证此配置最优）

**2. 删除 H2 IG（Instruction-Grounded）检查**：
- 根本原因：IG check 用任务关键词与 product page "other" 组选项做词汇匹配，误报率高
  - IDX 4374："butter" 匹配多种 butter 口味（已有 coverage filter 但仍有残余）
  - IDX 6789："basket" 匹配 "fun birthday gift basket"，覆盖过滤无效
  - IDX 6789 中 H4 attribute checklist 已正确识别属性，IG check 造成冲突
- 实施：从 `_buy_now_precheck` 中移除 IG check block（保留其在 step_guidance 中的信息性展示）
- 保留：color/size/spec 的 parsed-requirement check（可靠，基于 H0 明确解析）+ 防御性 guard（已修复）

**3. H4 price-over-budget 改为每 ASIN 仅触发一次**：
- 根本原因：每轮重复触发 → agent 试图 search_action → H2 阻断（产品页无搜索框）→ 死循环
- 实施：新增 `_price_warned_asins: set`，已警告过的 ASIN 不再重复触发
- 同时将消息改为 "Click 'back to search' and find a cheaper option"（更明确的动作指引）
