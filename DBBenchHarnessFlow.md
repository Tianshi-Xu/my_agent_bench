# DBBench Harness 架构文档 (v2)

## 背景

DBBench 任务要求 LLM 通过调用 `execute_sql` 和 `commit_final_answer` 两个工具，操作一个 MySQL 数据库来回答问题。

| 版本 | 准确率 |
|---|---|
| Baseline（无 harness） | 33/99 = **33.3%** |
| v1（harness 首次启用） | 43/100 = **43%** |
| v2 目标 | ≥ **58%** |

---

## v1 → v2 失败分析 & 修复

v1 运行后分析 57 个失败样本，发现以下根因：

| 根因 | 数量 | v2 修复 |
|---|---|---|
| **Evaluator bug**：`_clean_mysql_result` 对多列元组返回 `[]` 而非 `None`，导致多列 SELECT ground truth 始终为空集 | 21 (SELECT 全军覆没) | `result_processor.py`：空时返回 `None` 让 fallback 生效 |
| **Agent 答案格式错误**：13/21 SELECT 提交扁平 cells（"a","b","c"）而非行元组（"('a','b','c')"） | 13 | H3/H5 指令改为"每行一个 tuple repr"；`gate_commit` 新增 flat-cell regroup |
| **Agent 全部行合并成单字符串**："('a','b'),('c','d')" → 需拆分 | 3 | `normalize_answers_list` 新增 bare tuple sequence unpack |
| **Mutation 提交门限过低**：UPDATE/INSERT 任务 H2 只拦截 1 次，之后放行（agent 只做 SELECT 不做 UPDATE） | ~5 | `h2_mutation_commit_block_limit=3`；H4 新增 SELECT-only-on-mutation 探针 |
| **INSERT 编造值**：agent 凭空猜测要插入的数据 | ~4 | H5 `insert_value_order_matches_columns` 技能加强："读题目原文，禁止猜测" |
| **numeric_strip bug**："1900s" 被错误截断为 "1900" | 1-2 | `_normalize_scalar_numeric` 加空格guard |
| **NULL agg 在 SELECT 任务无法触发** H4 ③ | 1-2 | H4 ③ 扩展：当 last_sql 含 SUM/AVG/COUNT 时也触发 |

---

## Harness 的作用是：在不读取 test 标签、不作逐样本特化的前提下，系统性地修补失败根因：

| 根因 | 数量（baseline 66 个错误中） |
|---|---|
| SQL 语法 / MySQL 方言错误（列名未加反引号、`\|\|` 非法等） | 24.8% SQL 调用出错（71/286） |
| 答案格式错配（把多列结果拼成英文句子、附加单位词等） | ~12 例 |
| SQL 逻辑错误（过滤太严、数值列未 CAST、缺少 WHERE 等） | ~15 例 |

---

## 文件结构

```
src/server/harness/dbbench.py          # 核心逻辑（H0–H5）
src/server/harness/__init__.py         # 导出 DBBenchHarnessConfig / Runtime 等
src/server/tasks/dbbench/task.py       # 接线：在推理流水线中调用 harness
configs/tasks/dbbench.yaml             # 开关配置
extra/docker-compose.yml               # harness 目录挂载
```

---

## 层级总览

```
初始化（每个样本一次）
  H0  任务解析          ── 从 entry 读取 task_type / answer_shape / target_table
  H_SCHEMA 列名映射     ── 复现 _sanitize_identifier，建 name_map + Schema Card
  H3  静态 Prompt 补丁  ── 追加 MySQL 方言提示到工具描述 + system prompt
  H5  冷启动技能        ── BM25 检索 2 条最相关技能注入 user message

主循环（每轮一次）
  [动作前]
    H2  force 消费      ── 若上轮 H4 写入了强制动作，直接合成 commit
    H2  Rescue Parser   ── 从纯文本中拯救嵌入的工具调用（XML / kwarg / JSON / fence）
  
  [execute_sql 路径]
    H2  SQL Gate        ── 安全过滤 + 自动反引号 + 方言修复 + 重复 SQL 强制提交
    DB  execute(sql)    ── 真正执行
    H1  状态更新        ── 错误类型分类、候选答案提取、各类 streak 计数
    H4  Post-Step 监控  ── 语法错 / 未知列 / 空结果 / SQL 循环 + 预算 warn/force
    H5  逐步引导        ── 候选提示 / lint / 首轮模板 / 异常零值警告
  
  [commit_final_answer 路径]
    H2  Commit Gate     ── 拦截过早提交 / mutation 未执行
    H2  Answer 归一化   ── 去单位词、None→0、千位逗号、解包 stringified list
    评估               ── DBResultProcessor.compare_results（不改动）

退出点（共 3 处）
    [HARNESS_TRACE_V1] 注入 openai_messages（4 个退出点均覆盖）
```

> **合并政策**：H6（预算管理）并入 H4，H7（答案归一化）并入 H2。对外只暴露 `h2 / h3 / h4 / h5` 四个开关。

---

## H0 — 任务解析

**触发时机**：每个样本初始化时（`init_task(entry)`），只执行一次。

**输入**：`entry['type']`、`entry['description']`、`entry['table']`、`entry['sql']['query']`（仅读列数，不读值）。

**输出**：`DBTaskContext`

| 字段 | 说明 |
|---|---|
| `task_type` | SELECT / INSERT / UPDATE / DELETE / counting / ranking / aggregation-* / comparison / other |
| `answer_shape` | scalar_int / scalar_float / scalar_str / multi_row_single_col / multi_row_multi_col / hash |
| `target_table_sanitized` | 第一张表的入库名 |
| `target_cols_sanitized` | 第一张表各列的入库名 |
| `mentions_like` | description 含 contains/includes 等词 → True |
| `case_insensitive` | description 含 ignoring case 等词 → True |
| `expected_insert_cols` | INSERT 任务的列数（来自 std_sql 结构，不读值） |

**泛化性**：只读 `type[0]` 和词法，不读 label。

---

## H_SCHEMA — 列名截断映射

**触发时机**：H0 之后，系统消息发送前。

**背景**：`task.py::_sanitize_identifier` 将列名截断至 64 字符（含去除 `\n\t\r` / strip / 去重后缀），agent 从题目描述中照抄完整列名时就得到 "Unknown column" 错误。

**实现**：

1. 对 entry 中每个 table / column 重跑 `sanitize_identifier`（与 task.py 完全镜像），建立 `name_map: {lowercase → sanitized}`。
2. 标记被改写的列名（raw ≠ sanitized）。
3. 生成 **Schema Card** 注入 user prompt（列名含截断时附加 `/* truncated from N chars */`）：

```
[SCHEMA HINT] MySQL tables (always wrap names with spaces/punct in backticks):
- `Election Results` (`District`, `Incumbent`, `Party`, `Status`)
- `Seattle School Info` (`Location Denotes location of scho` /* truncated from 122 chars */)
Note: names longer than 64 chars were truncated on import. Reference columns EXACTLY as shown above.
```

---

## H1 — Session State

**触发时机**：每次 `execute_sql` 执行结束后调用 `update_state_after_sql`。

**追踪字段**：

| 字段 | 用途 |
|---|---|
| `sql_history` / `sql_history_raw` | 归一化后的 SQL 列表 + 原始文本，用于重复检测 |
| `last_error_kind` | syntax / unknown_col / unknown_table / timeout / null_agg / empty / ok |
| `candidate_answer` | 最近一次成功执行的 DB 输出（标量 str 或 list[str]） |
| `candidate_implausible` | 候选来自 None/空结果时为 True，阻止 H4 强制提交 |
| `error_streak / empty_streak / loop_streak` | 错误 / 空结果 / 重复 SQL 的连续计数 |
| `mutation_attempted` | INSERT/UPDATE/DELETE 任务：是否已成功执行变更 SQL |
| `commit_blocks_used` | H2 commit gate 已阻止次数（防死锁） |

---

## H2 — Action Gate（最高杠杆层）

**触发时机**：每轮 agent 动作返回后，在工具真正执行之前拦截。

### 2a. Rescue Parser

从 assistant 纯文本 content 中拯救嵌入的工具调用，优先级：XML `<tool_call>` → commit JSON/kwarg → execute_sql JSON/kwarg → ` ```sql ``` ` fence。

命中时注入 `harness_trace["h2"]["reason": "rescued_text_embedded"]`。

### 2b. Force Action 消费

若上轮 H4 写入了 `force_next_action`（仅 `commit_final_answer`），本轮跳过 `session.action()` 直接走 commit 路径。

### 2c. SQL 安全过滤

屏蔽 `DROP DATABASE`、`SHUTDOWN`、`GRANT ALL` 等 DDL。

### 2d. 自动反引号（auto_backtick_sql）

- 对照 H_SCHEMA 的 `name_map`，找出 SQL 文本中**未被反引号/引号包围**、且**含空格/点/标点**的已知标识符，自动补上反引号。
- 例：`SELECT Race Name FROM t` → `SELECT \`Race Name\` FROM t`
- 只修改 **schema 中已知的**名称，绝不凭空构造标识符。

### 2e. 方言修复（dialect_fix_sql）

`a || b` 在 SELECT 上下文 → `CONCAT(a, b)`（MySQL 不支持 `||` 作字符串拼接）。保守 2 元形式，避免误改 boolean OR。

### 2f. 重复 SQL 强制提交

相同归一化 SQL 连续出现 N 次（默认 2）且有合法候选答案 → 强制转化为 `commit_final_answer`，不再重跑。

### 2g. Commit Gate + Answer 归一化（原 H7）

**阻止条件**：
- 未运行任何 `execute_sql` 就提交（非 mutation，最多阻止 1 次）
- mutation 任务未执行变更 SQL 就提交（最多阻止 `h2_mutation_commit_block_limit=3` 次，比非 mutation 高）
- 空 answers 且非 mutation 任务（最多阻止 1 次）

**答案归一化规则**（按 `answer_shape` 分支）：

| 形状 | 规则 |
|---|---|
| scalar_int / scalar_float | 去尾部单位词（games/wins/records…）、去千位逗号、None/null/""→"0"、去尾点；仅当值含空格时提取末尾数字（避免 "1900s"→"1900"） |
| scalar_str | 去前缀短语（"the answer is …"）、去首尾引号、去尾点 |
| multi_row_single_col | 若 agent 把整个列表装入单个字符串（`"[('1948',),...]"`）则解包展开 |
| multi_row_multi_col | ① 解包 `"[(...),(...)]"` 包装字符串 → 每行保留 tuple repr；② 解包无括号裸 tuple 序列 `"('a','b'),('c','d')"` → 逐行拆分；③ 若 agent 提交扁平 cells 且 `last_result_col_count` 已知，按列数分组重建 tuple repr |
| hash（INSERT/UPDATE/DELETE） | **完全透传**，不修改 |

命中时记录 `harness_trace["h2_commit"]`。

---

## H3 — 静态 Prompt / 工具描述补丁

**触发时机**：任务类初始化时（`__init__`）执行一次。

**execute_sql.description 追加**：
> This is MySQL. Wrap identifiers with spaces/dots/punctuation in backticks. Use `CONCAT(a, b)` not `a || b`. Cast TEXT numeric columns with `CAST(col AS DECIMAL(20,6))` before SUM/AVG.

**commit_final_answer.description 追加**：
> For multi-row multi-column SELECT, put each row's values as separate strings in `answers`. Do NOT reformat rows into English sentences. For INSERT/UPDATE/DELETE, run the mutation SQL successfully BEFORE committing.

**system prompt 追加**：
> This database is MySQL. Column names longer than 64 characters were truncated on import; use the schema card shown below as the source of truth. A syntax error near a column name is almost always an un-backticked identifier.

---

## H4 — Post-Step 监控（含预算管理）

**触发时机**：每次 `execute_sql` 执行后、下一轮 agent 动作前。

入参 `remaining_rounds = max_round - 已完成轮次`。

**优先级短路表**（越靠前越优先）：

| 优先级 | 条件 | 注入内容 | 误伤防护 |
|---|---|---|---|
| ⓪ 语法错 | `last_error_kind == syntax` | 解析 `near 'X'` 提示反引号 | 仅在 MySQL 错误字符串含 `syntax` 时触发 |
| ① 未知列 | `last_error_kind == unknown_col` | 建议 `DESCRIBE \`table\`` | 仅在错误字符串含 `Unknown column` 时触发 |
| ② 未知表 | `last_error_kind == unknown_table` | 建议 `SHOW TABLES` | 仅在错误字符串含 `doesn't exist` 时触发 |
| **②.5** **mutation 只跑 SELECT** | mutation 任务 + ≥2 SQL 历史 + 全是 SELECT + 无 mutation_attempted | 强提示：执行 INSERT/UPDATE/DELETE SQL | 仅 mutation 任务类型触发 |
| ③ NULL 聚合 | `last_error_kind == null_agg` 且 task 为 aggregation-* **或** last_sql 含 SUM/AVG/COUNT | 建议 CAST + 提交 '0' | 扩展：SELECT 任务中使用聚合函数也触发 |
| ④ 连续空结果 | `empty_streak >= 2` 且 SELECT/counting/comparison | 建议 LIKE / 放宽过滤 | 需要 **连续** N 次才触发（默认 2） |
| ⑤ SQL 循环 | 最近 3 条 SQL 相同 | 有合法候选则强制提交，否则给 hint | 候选必须非 implausible |
| ⑥ **预算 warn**（原 H6） | `remaining <= 3` | 软提示：提交候选或调整策略 | 有候选且 non-implausible 时才给具体候选 |
| ⑦ **预算 force**（原 H6） | `remaining <= 2` 且候选合法 | 写入 `force_next_action` → 下轮 H2 消费 | 候选必须 `candidate_implausible == False` |

预算事件记录在 `harness_trace["h4"]["budget_branch"]`，方便审计时过滤预算类动作。

---

## H5 — Skill 库 + 逐步引导

### 5a. 冷启动技能（Cold-Start Skills）

BM25 检索（query = entry description，doc = skill keywords + text），按 `task_type` 预过滤后取 top-k（默认 2）。强制包含 `avoid_text_tool_calls`。

完整技能库（20 条）：

| id | 适用 task_type | 要点 |
|---|---|---|
| `backtick_identifiers` | ALL | 空格/点/标点列名必须反引号 |
| `mysql_dialect_concat` | SELECT/comparison/counting | `CONCAT(a,b)` 代替 `\|\|` |
| `mysql_cast_numeric` | aggregation-*/ranking | TEXT 列 SUM/AVG/ORDER 前 CAST |
| `describe_first_on_error` | ALL | 出错时先 DESCRIBE 看真实列名 |
| `like_contains_word` | SELECT/counting/comparison | contains → `LIKE '%X%'` |
| `or_any_of` | SELECT/counting | A 或 B → `LIKE '%A%' OR LIKE '%B%'` |
| `select_star_preserves_shape` | SELECT | 多列用 SELECT \*，原样提交 |
| `count_rows_basic` | counting | `SELECT COUNT(*) FROM … WHERE` |
| `ranking_order_by_limit` | ranking | `ORDER BY CAST(col AS SIGNED) LIMIT 1` |
| `aggregation_cast_sum` | aggregation-SUM/AVG | CAST 后 SUM，NULL → '0' |
| `aggregation_avg_denominator` | aggregation-AVG | 排除空行 WHERE col != '' |
| `insert_value_order_matches_columns` | INSERT | 值顺序对齐列顺序 |
| `update_needs_where` | UPDATE/DELETE | 无 WHERE 改全表 → hash 必错 |
| `commit_raw_db_output_for_multirow` | SELECT | 多行多列：原样提交，不拼英文句子 |
| `commit_numeric_bare` | counting/aggregation-* | 只提交数字，不提交 "5 games" |
| `none_or_null_means_zero` | aggregation-*/counting | None → '0' |
| `mutation_must_execute` | INSERT/UPDATE/DELETE | 先执行 SQL，再 commit |
| `no_repeat_when_done` | ALL | 有结果立即 commit，别重跑 |
| `avoid_text_tool_calls` | ALL | 不要在文本中写工具调用 |
| `truncation_handling` | ALL | 列名截断，用 schema card 里的名字 |
| `date_format_passthrough` | SELECT/ranking/comparison | 日期原样匹配 |

### 5b. 逐步引导（Step Guidance）

每轮最多 40 词，优先级：

| 优先级 | 条件 | 内容 |
|---|---|---|
| 0 | 有候选且 non-implausible（H4 未主动触发） | "last result suggests … call commit_final_answer" |
| 1 | 无候选，最近 SQL 含 semantic gap | lint：缺 LIKE / 缺 LOWER / 缺 ORDER BY 等 |
| 2 | 首轮（round_num=0，无 SQL 历史） | 按 task_type 给模板 SQL |
| 3 | 候选 implausible 且 counting/aggregation | "zero/None suspicious — relax filter" |

---

## 配置参数（configs/tasks/dbbench.yaml）

```yaml
harness:
  enabled: false                # true 时开启所有层；dbbench-env_train 覆盖为 true
  h2: true                      # H2 action gate（含 answer 归一化）
  h3: true                      # H3 tool/prompt 静态补丁
  h4: true                      # H4 post-step 监控（含预算管理）
  h5: true                      # H5 技能库 + 逐步引导
  h5_top_k: 2                   # 冷启动 BM25 取 top-k
  h2_repeat_sql_block_after: 2  # 相同 SQL N 次 → 强制提交
  h4_stall_window: 3            # 循环检测窗口
  h4_empty_threshold: 2         # 连续 N 次空结果触发 ④
  h4_budget_warn_threshold: 3   # remaining ≤ N → 软提示
  h4_budget_force_threshold: 2  # remaining ≤ N + 候选合法 → 强制
  h5_hint_max_words: 40         # 每条 hint 最大词数
```

`dbbench-std` 不开启 harness（`enabled: false`），保持评测公平性。

---

## 审计字段（[HARNESS_TRACE_V1]）

harness_trace 在每个样本结束时（含 COMPLETED / TASK_LIMIT_REACHED / TASK_ERROR / CANCELLED 共 4 个退出点）作为带标签的 user message 注入 openai_messages，格式与 OS/WebShop/ALFWorld harness 一致，可通过正则从 runs.jsonl 提取：

```json
{
  "h0": [{"task_type": "counting", "answer_shape": "scalar_int"}],
  "h2": [{"round": 2, "reason": "sql_rewritten", "hits": [...]}],
  "h2_commit": [{"round": 5, "action": "allow", "normalize": {...}}],
  "h3": [{"applied": true}],
  "h4": [{"round": 3, "reason": "syntax_error", "budget_branch": null}],
  "h5": [{"id": "count_rows_basic", "trigger": "cold_start"}]
}
```

---

## 泛化性自检

- **H0**：只读 `type[0]` + 题目词法，不读 label。
- **H_SCHEMA**：只读 `entry['table']`，不读 label。
- **H2 auto_backtick**：只看 SQL 文本 + schema_map 已知名称，不对比 std_sql。
- **H2 commit 归一化**：处理 LLM 常见冗余格式（单位词/千位逗号/None）；不做 label 反查。
- **H4 错误恢复**：基于 MySQL 错误字符串解析（`near 'X'` / `Unknown column`），不读 label。
- **H4 预算 force**：候选来自 agent 自己执行的 SQL 返回值，不读 label。
- **H5 BM25**：检索 query = 样本 description，doc = skill 自身文本；不读 label。

---

## 预期提升目标

| 指标 | Baseline v0 | v1 实测 | v2 目标 |
|---|---|---|---|
| 总体 acc | 33.3% | 43% | ≥ 58% |
| SELECT | 0/22 = 0% | 1/22 ≈ 5% | ≥ 40% |
| counting | 1/6 = 17% | 1/6 ≈ 17% | ≥ 40% |
| SQL 语法错率 | 24.8% | ~10% | ≤ 8% |
| task_limit_reached | 3.0% | 3.0% | ≤ 5% |

---

## 热重载

修改 harness 后重启对应容器即可生效（无需重建镜像）：

```bash
docker compose -f extra/docker-compose.yml restart dbbench-env_train
```

`src/server/harness` 已挂载为 volume，重启后新代码立即生效。
