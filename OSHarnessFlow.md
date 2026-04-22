# OS Harness 触发架构（按一次推理流程）

本文档描述当前仓库里 **已实现** 的 OS (Linux shell) harness，在一次正常推理中会在什么阶段触发、触发哪些能力。设计参考已有的 ALFWorld / WebShop harness，并针对 OS 任务的失败模式新增了 **H1（Shell 状态跟踪）**。

> **层级说明**：对外暴露的干预层为 **H2~H5** 四层。预算管理归入 H4（post-step 监控）；答案归一化归入 H2（rescue 时立即归一化 + eval 时兜底归一化）；H0（任务解析）和 H1（Shell 状态跟踪）为初始化/内部状态层，不对 session 注入消息。

## 设计动机

Linux shell 是一个**命令确定性**环境：相同命令在相同文件系统状态下返回相同输出，shell 的 exit code 和 stdout 是精确的执行反馈，没有感知噪声。这种确定性使 harness 能够以"零歧义"的方式解读每一步执行结果，并做出精准干预。

> **关键洞察**：Baseline 分析表明，约 **70% 的失败不是 shell 推理能力不足，而是工具调用协议违规**——agent 把 `answer_action({"answer":"5"})` 写进了纯文本而非真正调用工具。这意味着模型实际上已经得出了正确答案，只是输出格式违反了 function-calling 协议。H2 的 rescue parser 专门恢复这类"意图正确、格式错误"的输出，是整个 harness 中**杠杆最高的单一模块**。

OS harness 的四层干预对应 shell agent 的四类系统性问题：**H2** 恢复协议违规的工具调用（rescue）并守卫危险命令，**H3** 在 episode 开始时注入 shell 最佳实践，**H4** 监控截断/报错/循环并管理轮次预算，**H5** 提供任务类型感知的具体命令模板。每一层的触发条件均来自 shell 执行的客观信号（exit code、stdout 内容、轮次计数），不依赖模型的主观推断。

## 0. 开关层（是否启用）

任务配置入口在 `configs/tasks/os.yaml`：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `enabled` | bool | false | 总开关，false 时其余开关全部无效 |
| `h2` | bool | true | 动作前置校验：文本注入工具 rescue + 安全过滤 + 重复 bash 闸门 |
| `h3` | bool | true | 工具描述嵌入（shell 策略 + 工具使用格式提示） |
| `h4` | bool | true | 执行后监控：截断 / 报错 / 空输出 / 循环干预 + 预算管理 |
| `h5` | bool | true | 目标导向指导：BM25 skill 库 + 分步 hint |
| `h5_top_k` | int | 2 | cold-start 阶段注入的 skill 数量 |

### 全量配置参数参考

`OSHarnessConfig` 的完整默认值（`src/server/harness/os_interaction.py`）：

| 参数 | 默认值 | 所属层 | 说明 |
|------|-------|-------|------|
| `h2_repeat_bash_block_after` | 2 | H2 | 相同 bash 连续 N 次且已有候选答案时强制回答 |
| `h2_text_only_streak_force` | 2 | H2/H4 | 连续 N 轮无 tool call 后给出硬性提示 |
| `h4_stall_window` | 3 | H4 | 最近 N 轮相同命令 + 相同输出 → 循环干预 |
| `h4_empty_output_threshold` | 2 | H4 | 空输出累计 N 次 → 放宽过滤提示 |
| `h5_hint_max_words` | 40 | H5 | 每步 hint 词数上限 |
| `h5_top_k` | 2 | H5 | cold-start BM25 skill 检索返回数 |
| `h5_cold_start_max_words` | 50 | H5 | 每条 cold-start skill 词数上限 |
| `h6_warn_threshold` | 3 | H4 | 剩余轮数 ≤ N 时注入软告警（预算子逻辑，归入 H4） |
| `h6_force_threshold` | 2 | H4 | 剩余轮数 ≤ N 且有候选答案时硬性强制 answer_action（预算子逻辑，归入 H4） |
| `h7_max_strip_units` | 1 | H2 | 尾部单位词剥离层数（归一化子逻辑，归入 H2） |

---

## 背景：为何要 harness

### v0 基线（harness 全关）

OS baseline（Qwen3-4B-Instruct，100 个训练样本，`outputs/os/2026-04-18-20-47-05`）表现：

- **accuracy 46%**（46/100 通过）
- **39% task_limit_reached**（回合用完）
- **2% task_error**

在 54 个失败样本里：

| 失败形态 | 数量 / 54 | 证据 |
|---|---|---|
| Agent 把 `answer_action({"answer":"X"})` 当作 **纯文本** 输出，而不是真的调用 tool | **38** | idx 24, 34, 200, 210, 928, … |
| Agent 在同一条 `bash_action` 上循环 ≥ 2 轮 | 35 | idx 34 (×4), 200 (×4), 928 (×4) |
| tool 输出里其实已经有正确答案，但 agent 没提交 | 38 | 同上 |
| Agent 真的调了 `answer_action` 但答案语义错误 | 13 | idx 842 等 |

**结论：约 70% 的失败是 tool 调用协议违规，而不是 shell 推理能力不足。** 纯靠一个"把文本里的 `answer_action(...)` 提取出来合成真正工具调用"的 rescue parser，就预计能收回 ≈ 20pp 的 accuracy。剩余空间在经典 OS 推理陷阱（递归 / 大小写 / `-type f` / `-mtime` 方向 / 答案格式）以及 8 轮预算压力。

### v1 实测（`outputs/os/2026-04-18-21-27-00`） — harness 全开

- **accuracy 56%**（+10pp vs v0，但未达 60% 目标）
- **task_limit_reached 2%**（-37pp，循环问题基本修掉）
- **average_history_length 17.87**，max 23

v1 残留 44 个失败样本的深度抽样（5 条代表性轨迹 + 全量 runs.jsonl 统计）揭露：

| v1 残留失败形态 | 数量 / 44 | 根因 |
|---|---|---|
| rescue regex 只匹配 `answer_action({"answer":"X"})` JSON 风格，但 Qwen3-4B 实际输出是 Python kwarg `answer_action(answer='5')` 或 positional `answer_action(5)` | **38** | H2 regex 漏洞 |
| 预算强制提交把语义可疑候选（负数、溢出、过滤后为 0）提前提交 | 12 | 强制提交缺少 plausibility 过滤 |
| `total size ... human-readable` 被 H0 错分为 count_files（如 idx 113, 790） | 4 | H0 shape 识别不全 |
| awk 平均数任务踩除零 / 换行分隔陷阱 | 3 | 缺少专门 skill |
| harness_trace 审计信息在 runs.jsonl 里全部丢失 | 全部 | agentrl HTTP 层剥离 `TaskSampleExecutionResult.result` |

### v2 实测（`outputs/os/2026-04-19-10-35-30`）— 56% → 61%

v2 实测 61%（+5pp），task_limit_reached 6%（-33pp vs v0），平均轨迹长度 10.92（v1 的 17.87 大幅压缩）。

v2 残留 39 个失败的深度分析揭露：

| v2 残留失败形态 | 数量 / 39 | 根因 |
|---|---|---|
| XML `<tool_call>` 格式未被 rescue 捕获（Qwen3 在某些 episode 使用 JSON 格式）| **6** | H2 缺少 XML regex |
| Lint 过激：agent 已有正确候选时 lint 触发推偏答案 | **4** | idx 24, 720, 809, 450 |
| size 小数格式 `50.0K` 被评估器 `int()` 报 ValueError | **2** | idx 113, 682 |
| 0 是正确答案但 implausible 规则阻止提交 → task_limit | **1+** | idx 200 |
| String 答案任务（日期）的数字子串 "-02" 被误判为负数 → implausible | **1** | idx 350 |
| bash 逻辑本身算错（harness 无法干预） | **25** | 其余 completed failures |

### v3 目标

v3 针对以上所有可修复根因加固，架构统一为 H0~H5 主层级，并将预算与归一化纳入主流程。目标 accuracy ≥ 68%。

### 训练集分布（training.json，n ≈ 1000）— 仅用作设计参考，不进入代码

- 93% 答案是 **bare integer**；4.7% 字符串；2.3% 人类可读 size。
- 93% 查询任务 / 7% 副作用任务（mutation）。
- 70%+ 是计数任务（`find … | wc -l`，`sort -u | wc -l`，`grep -c`）。
- 参考解法 2–4 个 pipe，91% 用到 `wc`，66% 用到 `grep`，43% 用到 `find`。
- 常见陷阱：递归 vs 非递归，大小写，`-type f` 是否过滤目录，`-mtime -N`（N 天内）vs `-mtime N`（恰好第 N 天）。

---

## 一、整体流水线（单次 sample 内）

```
init                 : H0 task parse ──▶ H3 tool 描述补丁（一次性）──▶ H5 cold-start skill
每轮（agent → env）  :
  [before tool exec] H2 rescue parser + safety + duplicate-bash 闸门
                         ↳ rescue answer_action → 立即归一化
  [container 执行]
  [decode 后]        H1 state 更新 ──▶ H4 post-step 监控（含预算） ──▶ H5 step guidance
commit               : eval 前答案归一化兜底
```

> **实现对齐说明**：预算 warn/force 在 bash 后由 H4 的 `post_step_monitor(remaining_rounds)` 触发；归一化在 rescue 与 eval 两个时机共用，不单列层级。

---

## 二、H0 — 任务解析（`OSTaskContext`）

**触发**：`OSHarnessRuntime.init_task(description)`，一次 sample 初始化时调用一次。

**作用**：将任务描述用规则/正则解析成结构化上下文。所有判定都是**常识级词法 / 正则**，**不使用训练集统计量**（测试集安全）。

| 字段 | 取值 | 来源信号 |
|---|---|---|
| `task_type` | `count_files`, `count_lines`, `count_matches`, `count_unique`, `largest`, `smallest`, `list`, `read_content`, `system_info`, `sum_size`*, `average`*, `mutate`, `other` | 任务文本关键词 |
| `answer_shape` | `integer`, `string`, `size`, `path`, `None` | 基于 task_type + "name/filename/path" / "size" 等信号 |
| `target_path` | 如 `~/foo`, `/var/log`, `~` | 任务文本里的 path 模式 |
| `extension_filter` | 如 `log`, `txt`, `py` | 引号 `".log"` 或 `.log extension/files` |
| `recursive` | True / False / None | `subdirectories` / `recursively` → True |
| `case_sensitive` | True / False / None | `ignoring case` → False |
| `time_filter_days` | int / None | `last N (days\|hours\|weeks)` |
| `size_filter_bytes_gt`/`lt` | int / None | `greater/less than N (KB\|MB\|GB)` |

> **v2 新增**（标注 `*`）：
>
> - `sum_size`：匹配 `total size`, `sum of sizes`, `combined size`, `disk usage`。若任务里同时出现 `human[- ]readable` → `answer_shape=size` 且 cold-start 注入 `du -sh`/`numfmt --to=iec` 模板；否则 shape=`integer`。**在 count 正则前先判定 sum_size**，否则会被错分为 `count_files`（v1 bug: idx 113/790）。
> - `average`：匹配 `average|mean` **且** 后跟定量名词（age/value/size/length/count/number/score/time/rate/price/weight/height/duration/latency/salary/temperature）**或** 被 `calculate|compute|find` 等动词前缀。正则收紧是为了避免误伤 "on average, files are 10KB"（该句里 task 其实是 count）。shape=`integer` 或 `size`。

---

## 三、H1 — Shell / 对话状态（`OSShellState`，本 harness 新增）

**触发**：`update_state_after_bash()` 在每次 bash 执行、输出 decode 后被调用。

**作用**：给 H2 / H4 / H5 提供决策输入，不对 session 注入消息。

字段：

- `bash_history: List[str]` — 历史 bash 脚本，用于重复检测。
- `last_output_raw / truncated / was_empty / had_error` — 最近一轮 bash 的状态位。
- `last_output_numeric_candidates` — 用 `extract_numeric_candidates` 从输出里抽出的可能答案（优先级：纯整数行 → `N total` 行 → 最后一行末尾整数）。
- `candidate_numeric_answer` — 当前最合理的 integer 候选答案。
- `candidate_string_answer` — 单行输出（largest / smallest 的 filename）。
- `candidate_implausible`（**v2 新增**）— `bool`，最近一次 bash 产生的候选是否被 `is_plausible_numeric_candidate` 判定为语义可疑（负数、溢出、`_uint64` wrap、>1e12、或过滤任务上命中 0）。该位被 H5 与 H4(预算子逻辑) 联合消费，用来**拒绝提交**而不是**拒绝记录**。
- `empty_output_streak` — 连续空输出次数。
- `text_only_streak` — 连续无 tool call 次数。
- `rescue_hits`、`answered` — 审计字段。

---

## 四、H2 — Action Gate（含 rescue parser，**最高杠杆层**，v3 新增 XML 格式）

**触发**：`_handle_agent_action` 解析 `session.action()` 的结果后。

优先级顺序：

1. **Force-action 消费**：若之前的 H4 写入了 `harness_runtime.force_next_action`，本轮直接合成该 tool_call 来喂回 env，跳过 agent 原输出。`harness_trace.h2` 记录 `reason="force_consumed"`。

2. **Rescue parser（v2 大幅扩展，覆盖 v0 38/54 + v1 残留 38/44）**：当 `tool_calls == []` 但 `content` 里嵌入了以下模式之一，就合成真正的 tool_call。v2 补了 Qwen3-4B 实际最常输出的 Python kwarg / positional / bare 三种风格：

    | 匹配模式（regex 名） | 示例 | 合成调用 |
    |---|---|---|
    | `_RESCUE_ANSWER_JSON_RE` | `answer_action({"answer":"5"})` | `answer_action(answer="5")` |
    | `_RESCUE_ANSWER_KWARG_RE`（**v2**） | `answer_action(answer='5')` / `answer_action(answer="5")` | `answer_action(answer="5")` |
    | `_RESCUE_ANSWER_POSITIONAL_RE`（**v2**） | `answer_action(5)` / `answer_action("5")` | `answer_action(answer="5")` |
    | `_RESCUE_ANSWER_BARE_RE`（**v2**，MULTILINE） | 行首 `answer_action 60` | `answer_action(answer="60")` |
    | `_RESCUE_BASH_KWARG_RE`（**v2**） | `bash_action(script='find …')` | `bash_action(script=...)` |
    | `_RESCUE_FINISH_KWARG_RE`（**v2**） | `finish_action(thought='done')` | `finish_action(thought=...)` |
    | `_RESCUE_TOOL_CALL_XML_RE`（**v3**） | `<tool_call>{"name":"bash_action","arguments":{"script":"..."}}</tool_call>` | 任意 action |
    | ReAct 风格 `Act: answer(X)` | / | `answer_action(answer="X")` |
    | ReAct 风格 `Act: bash` + ```` ```bash … ``` ```` | / | `bash_action(script=...)` |

    XML 格式优先最先匹配（unambiguous，含 name+full arguments），其余按 strict→loose 顺序。XML 格式修复了 idx 760 整条 task_limit（Qwen3-4B 在该 episode 全程输出 `<tool_call>...</tool_call>` 而 rescue 未命中）。

    **rescue → 立即归一化（v3）**：当 rescue 命中 `answer_action` 时，立即在 `task.py` 中对 `answer` 值调用 `normalize_answer`，保证提交值与 eval-time 归一化一致。归一化事件记录在 H2 trace 中。

    **解析顺序（strict→loose）**：JSON > kwarg > positional > bare > ReAct；优先 answer/finish > bash（一旦 agent 已经写出答案，宁可提交也不要再跑命令）。

    **False-positive 防御**：捕获到的 answer 值如果命中 `{"answer", "action", "value"}`（常见的 kwarg 名本身被误捕），直接丢弃该命中尝试下一个模式。这避免了 `answer_action(answer=answer)` 这种病态输出被错解析成 `answer="answer"`。

    `harness_trace.h2` 记录 `reason="rescued_text_embedded"` 和具体命中的 regex 名。

3. **无 tool call 且无可救**：注入 `"No executable tool calls found. Please call a tool instead"`；连续 ≥ 2 轮无 tool call 时追加 `"you MUST invoke a function. Never write answer_action(...) as plain text"`。

4. **bash 安全过滤**（`_is_dangerous_bash`）：阻断 `rm -rf /`、fork bomb、`mkfs`、`dd of=/dev/`、`shutdown`、`reboot`、`chmod -R 777 /`。容器本身是沙箱，但这些命令会浪费 round 预算。

5. **重复 bash 闸门**：如果最近 `h2_repeat_bash_block_after=2` 次 bash 完全相同，**并且** 已经抽到 integer 候选答案，**并且** task 是 integer-shape，**并且** 还没提交答案，**并且**（**v2 新增**）候选**不是** implausible → 设置 `force_next_action = answer_action(candidate)` 并阻断本次 bash 执行。这是 35 episode 循环失败的主要修复；v2 补的 implausibility 守门避免了 "循环 + 语义可疑候选" 被硬塞进答案。

---

## 五、H3 — 工具描述补丁（静态）

**触发**：`OSInteraction.__init__` 中，`patch_os_tool_descriptions(tools)` 在 `enabled and h3_enabled` 时对 `self.tools` 做一次 deepcopy + append。测试集/训练集完全一致（无 split-specific 文本）。

- `bash_action.description` 追加：
  > Prefer one concise pipeline per turn. For counting files, use `find DIR -type f ... | wc -l` (add `-type f` to exclude directories). For counting unique values, pipe to `sort -u | wc -l`. For matching lines, prefer `grep -c PATTERN FILE`.

- `answer_action.description` 追加：
  > Return ONLY the bare value (e.g. '5', not '5 files' or 'The answer is 5'). Do NOT write `answer_action(...)` as plain text — you MUST invoke this tool via a real function call. Plain-text invocations are rejected.

- `finish_action.description` 追加：
  > Use `finish_action` only for mutation tasks (create/delete/chmod/...). For a question with a numeric or string answer, use `answer_action`.

---

## 六、H4 — Post-Step 监控 + 预算管理

**触发**：`_execute_bash_command` 里，`session.inject(tool_message)` 之后、下一轮开始之前。接收 `remaining_rounds = round_limit - current_round`。

按优先级短路返回（⑤ 预算为并入逻辑）：

| 优先级 | 触发条件 | 注入提示 |
|---|---|---|
| ⓪ 截断 | `[truncated because the output is too long]` 出现 | "Refine the command — pipe to `wc -l` for a count, `head -20` for a sample, or narrow the filter. Do NOT re-run the same command." |
| ① command not found | 输出含 `command not found` | "Try an alternative (ss instead of netstat, ip a instead of ifconfig) or check PATH." |
| ① no such file | 输出含 `no such file or directory` | "Try `ls ~` or widen the path — the expected directory may be under your home." |
| ① permission denied | 输出含 `permission denied` | "Prefix with `sudo` if you are root, or inspect with `stat` / `ls -l` first." |
| ② count 任务空输出 | `empty_output_streak ≥ 1` 且 task 属于计数族 | "The filter may be too strict — drop the extension/time/size constraint one step at a time." |
| ③ bash 循环 | 最近 `h4_stall_window=3` 轮同一条命令 | 若有 integer 候选且非 implausible：提交提示；否则调用 `_first_turn_hint` 给具体替代命令 |
| ④ 连续无 tool call | `text_only_streak ≥ 2` | "You must call a tool. Never embed `answer_action(...)` in plain text." |
| ⑤ 预算 force | `remaining-1 ≤ h6_force_threshold=2` 且有 plausible integer 候选 | 硬性强制：`force_next_action = answer_action(candidate)`，下轮 H2 消费 |
| ⑤ 预算 warn | `remaining-1 ≤ h6_warn_threshold=3` | 软告警；implausible 候选 → 专门的"implausible，请换方案"提示；无候选 → 通用催促 |

预算事件和错误恢复统一记录在 `harness_trace.h4`，统一 post-step 触发点，减少循环体分支复杂度。

**设计取舍**：预算检查和错误恢复都属于同一类 post-step 决策，统一放在 H4 可以减少触发点和循环体分支复杂度。

---

## 七、H5 — Skill 库 + 分步指导

### 7a. Cold-start BM25 skill 检索

**触发**：`_judge` 刚注入完 user 消息后，当 `h5_enabled` 时调用 `cold_start_skill_hints()`。

- **27 条 skill**（`OS_SKILLS`，v2 由 20 → 27），每条有 `task_types` 过滤标签 + `keywords` BM25 文档。
- 两层检索：先按 `task_ctx.task_type` 过滤候选集 → BM25 对 `raw_description` 打分 → 取 top-`h5_top_k`（默认 2）。
- 始终强行追加 `avoid_text_tool_calls` skill（针对 v0 的 38/54 失败模式），替换最后一个位。
- 注入示例：

  ```
  Harness skill hints for this task:
  - To count files with a given extension, use `find DIR -type f -name '*.EXT' | wc -l`. …
  - NEVER write `answer_action(...)` or `bash_action(...)` in plain text — you MUST invoke the tool via a real function call.
  ```

skill 库设计（按 task_type 分组）：

| id | task_types | 覆盖问题 |
|---|---|---|
| `count_files_by_ext` | count_files | 扩展名过滤 + `-type f` |
| `count_files_mtime` | count_files | `-mtime -N` vs `-mtime N` 方向 |
| `count_files_size` | count_files | `-size +Nk` / `-size -Nm` |
| `count_subdirs` | count_files | `find DIR -mindepth 1 -type d` |
| `count_nonrecursive`（**v2**） | count_files | `find DIR -maxdepth 1 -type f \| wc -l` — 专治单层目录误当作递归 |
| `recursion_hint` | 多个计数族 | `find`/`grep -r` vs 非递归 `ls`/`grep` |
| `exclude_dirs_hint` | count_files | `-type f` 排除目录 |
| `count_lines_matching` | count_matches | `grep -c PATTERN FILE` |
| `count_lines_total` | count_lines | `wc -l` 用法 |
| `count_unique_field` | count_unique | `sort -u \| wc -l` |
| `extract_field_then_unique`（**v2**） | count_unique / average | `awk -F: '{print $1}' \| sort -u` / 先抽字段再去重 |
| `grep_time_window`（**v2**） | count_matches / count_unique | `grep -E '2024-(01\|02)'` 时间窗过滤（awk 月份比较是陷阱） |
| `files_containing_pattern`（**v2**） | count_files / count_matches | `grep -rl PATTERN DIR \| wc -l` — 文件数 vs 匹配行数区分 |
| `hidden_files`（**v2**） | count_files / list | `find DIR -name '.*' -type f` / `ls -A` — 专治 `ls` 默认跳过 dotfile |
| `case_insensitive_hint` | 多个 | `grep -i` / `find -iname` |
| `find_largest` / `find_smallest` | largest / smallest | `-printf '%s %p\n' \| sort` |
| `sum_sizes` | count_files / system_info | `du -sh` / `du -b` |
| `human_readable_total_size`（**v2**） | sum_size | `du -sh DIR` 或 `find … -printf '%s\n' \| awk '{s+=$1} END {print s}' \| numfmt --to=iec` — 专治 size 任务被当计数题 |
| `awk_safe_average`（**v2**） | average | `awk '{s+=$1; n++} END {if(n) printf "%.2f\n", s/n}'` — 避免除零 / 尾部换行导致多计一个空字段 |
| `system_disk_free` / `system_mem` / `process_count` | system_info | `df` / `free` / `ps` |
| `answer_format_reminder` | ALL | 只写 bare value |
| `avoid_text_tool_calls` | ALL | 严禁文本嵌入 tool call（永远注入） |
| `no_repeat_when_done` | ALL | 已有答案就提交别再跑命令 |
| `truncation_handling` | ALL | 截断时如何 refine |

7 条 v2 新增 skill 的选型原则：**每条对应一个语义类陷阱**，不是对某个失败 idx 的回溯。`hidden_files` 来自 v1 idx 样本里多次出现的 "用 `ls \| wc -l` 漏掉 dotfile"；`grep_time_window` 来自时间窗过滤被写成 awk 字符串比较的典型错误。不绑定具体 dataset 样例。

### 7b. Step guidance（分步 hint，v2 四层优先级）

**触发**：`_execute_bash_command` 在 H4 之后，`step_guidance(round_num, round_limit)`。

分支优先级（**v2 新增 0 和 2，重排原有分支**）：

0. **semantic-gap lint（v2 新增）**：若 `task_ctx` 里有 `extension_filter` / `recursive` / `case_sensitive` 的强信号但最近一条 bash 没有对应 flag（`-name`/`-iname`/`-r`/`-type f`），调用 `bash_semantic_gaps(ctx, bash)` 产出一条具体 lint，例如：
    - task 里有 `.log` 但 bash 用 `ls` 无 `*.log` → `"Hint: filter by extension with 'find ~/foo -type f -name "*.log"'."`
    - task 说 `recursively` 但 bash 用 `ls DIR` 无 `-R` / `find` → `"Hint: use 'find DIR -type f' to recurse into subdirectories."`
    - task 说 `ignoring case` 但 bash 用 `grep` 无 `-i` → `"Hint: add '-i' for case-insensitive match."`
    - task 要 `files` 但 bash 没 `-type f` → `"Hint: add '-type f' to exclude directories."`
1. **promote numeric candidate（修订）**：`answer_shape == integer` 且 `candidate_numeric_answer` 存在且**非 implausible** 且未提交 → `"Hint: the last output contains the likely answer '{N}'. If that matches the task, call answer_action(answer='{N}')."`
2. **implausibility warn（v2 新增）**：`candidate_implausible=True` → `"Hint: your last pipeline produced an implausible value ({N}). Adjust the filter before submitting."`；**不** 合成 answer。
3. **promote string candidate**：largest/smallest 任务且输出恰好一行 → `"Hint: if '{name}' is the target, call answer_action with just that value."`
4. **round 0 且无 bash 历史** → 根据 `task_ctx` 调用 `_first_turn_hint` 生成具体的首条命令模板：
    - `count_files` + `ext=log` + `target=~/foo` → `` Hint: try `find ~/foo -type f -name '*.log' | wc -l`. ``
    - `count_matches` → `` grep -c PATTERN FILE `` 或 `` grep -rh PATTERN … | wc -l ``
    - `largest` → `` find … -printf '%s %p\n' | sort -rn | head -1 ``
    - `sum_size`（**v2**）→ `` du -sh DIR `` 或 `` find … -printf '%s\n' | awk … | numfmt --to=iec ``
    - `average`（**v2**）→ `` awk '{s+=$1; n++} END {if(n) printf "%.2f\n", s/n}' ``

- 硬性约束：`h5_hint_max_words=40`。
- 去重：`self._last_hint` 缓存，相同 hint 本轮不重复注入。

### 7c. is_plausible_numeric_candidate（v2 新增辅助）

纯语义判定，不查任何训练集：

- reject `value < 0`、`value == 0 on filter task`（过滤任务里零答案几乎总是说明过滤写错了）
- reject `value > 1e12`（uint64 溢出 / wrap 的典型表征）
- reject `str(value)` 长度 > 12（绝大多数合法答案 ≤ 6 位数）

---

## 八、H2 归一化子逻辑（答案归一化）

答案归一化是 H2 的一部分，在 rescue 时和 eval 时各触发一次：

**触发**（双路径）：
1. **Rescue 时**（`task.py::_handle_agent_action`）：rescue 提取到 `answer_action` 后立即归一化，记录到 `harness_trace.h2[trigger="rescue_normalize"]`。
2. **Eval 时**（`_evaluate_answer` 中，`config.match["strip"]` 之后）：兜底归一化，确保 rescue 后未经过归一化的答案也被处理。

`normalize_answer(value, answer_shape)` 根据 H0 的 shape 做条件处理：

- 去前缀：`"the answer is "`, `"answer: "`, `"count: "`, `"total: "` 等。
- 去外层引号。
- **integer shape**：
  - strip 末尾 `.` 
  - 去尾部单位词：`files`, `lines`, `bytes`, `matches`, `occurrences`, `directories`, `processes`, `users`, `entries`, `items`, `IPs`, `addresses`
  - 若还不是纯整数，取正则 `\d+` 的最后一个匹配
- **size shape**：
  - 将 `"42.5 MB"` 折叠成 `"42.5MB"`（空格归一）
  - **v3 新增**：将 `"50.0K"` → `"50K"`、`"10.0M"` → `"10M"`（去 `.0` 小数）。根因：评估器 `size-match.py` 调用 `int("50.0")` 会抛 ValueError，导致答案被判错；awk 的 `printf "%.1f"` 格式产生此小数；`du -ch ... | tail -1` 给出整数格式，是推荐替代
- **string shape**：去尾部 `.`、尾部换行
- **path shape**：去尾 `/` 和 `.`
- **shape 为 None**：**passthrough**（避免误伤 mutation 或 system_info 未分类场景）

**返回**：`(normalized, mutated)`；`mutated=True` 时记录归一化前后的值（可在 trace 中查找）。

关键通用性：所有规则不针对具体样本，而是针对常见自然语言余量（"5 files"、"The answer is 5"、"42.5 MB" 等）。

---

## 九、harness_trace（审计）

每个 sample 的 `TaskSampleExecutionResult.result["harness_trace"]` 结构：

```
{
  "h2": [{round, reason, blocked, action_name}, ...],   # rescue / force / normalize 事件
  "h3": [{applied}],
  "h4": [{round, audit_reason, recovery_prompt, budget_branch}, ...],  # 监控 + 预算事件
  "h5": [{id, text, trigger, token_cost}, ...]
}
```

> 注：代码内部兼容字段 `h6`（预算强制）和 `h7`（归一化）在论文层面已归并入 `h4` 和 `h2`，对外审计以 `h2 / h4` 为准。

使用：

1. 审计 H2 rescue 是否只在 `tool_calls` 为空时触发。
2. 确认 H4 预算 force 只在有候选且非 implausible 时触发。
3. 确认 H2 归一化在 shape=None 时完全 passthrough。

### v2 持久化机制（重要）

v1 阶段曾发现：在 `runs.jsonl` 里 `harness_trace` 全部丢失。根因是 agentrl 的 HTTP worker/controller 协议在传递 `TaskSampleExecutionResult` 时**只保留 `status` + `result.answer` / `result.detail`**，其他字段被剥离。这是 agentrl 上游包的行为，不应修改。

v2 做法（`src/server/tasks/os_interaction/task.py::_persist_harness_trace`）：在 agent loop 的 3 个退出点（task_limit_reached / eval-error / 正常完成）**把 harness_trace 作为带标签的 user message 注入到 session**：

```python
session.inject(ChatCompletionUserMessageParam(
    role='user',
    content='[HARNESS_TRACE_V1]\n' + json.dumps(harness_trace, ensure_ascii=False),
))
```

该消息随 `api_messages` → `openai_messages` 一起被保留进 `runs.jsonl`。下游审计只需 `grep -A1 HARNESS_TRACE_V1` 就能恢复完整 trace。
- **注意**：注入必须在 agent 最后一次 action 之后，否则会被当作 user turn 喂回给 LLM 浪费一次推理。
- 这个 pattern 对 alfworld / webshop 同样可复用（它们的 trace 也被上游剥离）。

---

## 十、文件落点

| 文件 | 改动 |
|---|---|
| `src/server/harness/os_interaction.py` | **新建**。`OSHarnessConfig`, `OSHarnessRuntime`, `OSTaskContext`, `OSShellState`, `OS_SKILLS`, `patch_os_tool_descriptions`, `rescue_tool_call_from_text`, `normalize_answer`, safety filter |
| `src/server/harness/__init__.py` | 新增 OS 导出 |
| `src/server/tasks/os_interaction/task.py` | 接线 H0–H5 + 预算/归一化流程；`harness_trace` 注入到 result |
| `configs/tasks/os.yaml` | `default.parameters` 默认关闭、`os-env_train.parameters` 开启 |
| `extra/docker-compose.yml` | 无改动（已 mount `src/server/harness` 和 `src/server/tasks/os_interaction`） |

---

## 十一、泛化约束与迁移性

**测试集隔离（不看标签）**：

- H0 解析全部是常识级词法/正则，**没有任何训练集统计量**，未见过的任务形态退化到 `task_type=other`（harness 等价于部分关闭，不会产生错误干预）。
- H5 skill 库 BM25 分数基于当次任务自己的 `raw_description`，无 reference-answer 查表。
- 预算 force 只在 H1 抽到具体 integer 候选时触发；**永远不伪造 string 答案**。
- 归一化流程针对常见自然语言修饰（"5 files" / "The answer is 5" / "42.5 MB"），不做 per-sample 调参。
- 所有阈值是**行为阈值**（轮数、重复次数、字符长度），不是针对特定任务的特化值。

**向其他 shell/代码执行场景的迁移性**：

rescue parser 的核心逻辑——从纯文本内容中识别并恢复嵌入的工具调用——是 **function-calling 场景的通用问题**，不限于 shell 任务。任何使用 function-calling 协议的 LLM agent 都可能出现"把工具调用写进文本"的失败模式；rescue parser 的正则覆盖策略（JSON / kwarg / positional / bare / XML 五种格式）可直接迁移到其他 agent 框架。H4 的错误恢复逻辑（截断 / 路径不存在 / 权限拒绝 / 循环检测）是 Linux shell 通用的，对其他 shell benchmark（ShellBench、BashBench 等）同样适用。

---

## 十二、验证路径

1. **单元级 smoke** — `OSTaskContext` 对 10 条手挑样本（含 idx 24, 34, 200, 210, 928, 842）解析正确；v2 补的正则回归："On average, files are 10KB. Count how many..." 不能被误归为 `average`。
2. **Rescue replay** — 对 38 个 text_embedded_tool 失败样本的 assistant `content` 喂给 `rescue_tool_call_from_text`，目标 ≥35/38 成功合成 `answer_action`；v2 追加 14 条 kwarg/positional/bare 测试样本应 100% 命中。
3. **端到端 100 样本**（`os-env_train`）— 与 baseline 对比 `overall.json`：
   - v0 → v1 → v2：46% → 56% → 61%（已达成）
   - **v3 目标**：accuracy ≥ 68%；`task_limit_reached` 保持 ≤ 5%；原有通过样本无回归。
4. **Trace 抽样** — 10 个样本的 `harness_trace` 审计（v2 走 `[HARNESS_TRACE_V1]` 注入通道）：H2 rescue 只在 tool_calls 为空时触发，预算 force 只在有候选且非 implausible 时触发，归一化流程在 shape=None 时 passthrough。
5. **容器热重载** — `docker compose -f extra/docker-compose.yml restart os_interaction-env_train` 不重 build 即生效（已在 docker-compose.yml 中 mount `src/server/harness` 和 `src/server/tasks/os_interaction`）。

---

## 十三、v1 范围外

- 多步规划（把一个任务拆成若干子任务）。skill 库只给单 pipeline 模板。
- 多值答案的输出解析（答案 shape ≥ 93% 是单值）。
- 容器级资源监控 / 超时（wrapper 已有）。
- 任何会泄露 train/test 边界的信息（per-task override、记忆化答案、dataset-specific 阈值）。
