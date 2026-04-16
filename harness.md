# AgentBench Harness Implementation Notes (Structure Only)

This document is for a new LLM/engineer to quickly understand the current repository structure and where harness integration should be attached. It intentionally does **not** define concrete harness policies yet.

## 1) End-to-end execution flow

- Run command: `python -m src.assigner -c <assignment_yaml>`
- Main scheduler: `src/assigner.py`
  - Parses assignment config, dispatches samples, handles retries, writes outputs.
- Per-sample client entry: `src/client/task.py` -> `TaskClient.run_sample(...)`
  - Talks to controller (`/start_sample`, `/interact`, `/cancel`)
  - Normalizes controller responses and returns `TaskClientOutput`.
- Task worker execution (inside task services):
  - WebShop: `src/server/tasks/webshop/task.py`
  - ALFWorld: `src/server/tasks/alfworld/task.py`

## 2) Key extension points for future harness

If we need per-sample/per-step harness behavior, these are the primary hook locations:

1. Global sample-level hook (task-agnostic):
   - `src/client/task.py` -> `TaskClient.run_sample`
2. WebShop task loop:
   - `src/server/tasks/webshop/task.py` -> `WebShop.sync_start_sample`
3. ALFWorld task loop:
   - `src/server/tasks/alfworld/task.py` -> `ALFWorld.sync_start_sample`
   - `src/server/tasks/alfworld/task.py` -> `ALFWorld.alfworld_run`

## 3) Config layering (what controls what)

- Task behavior (dataset range, sampling):
  - `configs/tasks/webshop.yaml`
  - `configs/tasks/alfworld.yaml`
- Assignment behavior (agent-task mapping, concurrency, output dir):
  - `configs/assignments/*.yaml`
- Runtime services and code mapping:
  - `extra/docker-compose.yml`

## 4) Docker and code effectiveness rules

- Changes under `src/server/tasks/*` are worker-side code:
  - Require recreating the corresponding worker container to take effect.
- Changes under `src/client/*` or `src/assigner.py` are local runner-side code:
  - No worker restart needed; rerun assigner command is enough.

## 5) Recommended boundaries (without deciding harness policy)

When starting implementation, keep this separation:

- Put generic harness abstractions in a dedicated module, e.g.:
  - `src/server/harness/` (new folder)
- Keep task-specific invocation in each task file:
  - `webshop/task.py`, `alfworld/task.py`
- Keep output traces in sample result payload (for offline analysis), e.g.:
  - `TaskSampleExecutionResult.result["harness_trace"]`

## 6) Current observability status

- Client side already has per-sample start logs in assigner terminal:
  - `[ASSIGNER][SAMPLE_START] task=<...> index=<...>`
- WebShop worker side has per-sample start logs in container logs:
  - `[MOUNT_CHECK][SAMPLE_START] webshop index=<...>`

These logs are useful to verify future harness hooks are triggered at the intended layer.

# Tau-bench Harness 文档

## 各 Domain 评测指标

每个 domain 的最终 reward 由若干**组件**相乘得到（任意组件为 0 则总分为 0）。各组件由 task 的 `evaluation_criteria.reward_basis` 字段声明，不同 domain 的默认组合不同。

### 组件类型说明

| 组件 | RewardType | 含义 | 实现 |
|------|-----------|------|------|
| **DB** | `DB` | 仿真结束后将 agent 产生的 DB 变更与 gold actions 重放结果对比，状态一致则为 1.0 | `EnvironmentEvaluator` |
| **ENV_ASSERTION** | `ENV_ASSERTION` | 在预测环境上运行一组**程序化断言函数**（如 `assert_mobile_data_status()`），全部通过为 1.0 | 同上 |
| **COMMUNICATE** | `COMMUNICATE` | 检查 agent 消息中是否出现了 task 要求的关键数值 / 字符串（精确子串匹配） | `CommunicateEvaluator` |
| **NL_ASSERTION** | `NL_ASSERTION` | 由 LLM judge 对对话进行自然语言断言判断（如"agent 应告知用户退款金额为 $X"） | `NLAssertionsEvaluator` |
| **ACTION** | `ACTION` | 检查 agent 是否调用了正确的工具 + 参数（按 gold actions 列表匹配） | `ActionEvaluator` |

> 当 task 的 `communicate_info` 或 `nl_assertions` 字段为空时，对应组件自动返回 1.0（视为通过），不发起任何检查或 LLM 调用。

---

### 各 Domain 默认指标

#### Airline（50 tasks）

```
reward_basis = ["DB", "COMMUNICATE"]   ← 100% 任务
reward = DB_reward × COMMUNICATE_reward
```

- **DB**：订单、航班、会员信息等数据库状态是否与 gold 一致
- **COMMUNICATE**：agent 是否在对话中明确说出了 task 指定的关键值（如补偿金额、退款日期）

**实际效果**：50 个任务中有 44 个 `communicate_info` 为空 → COMMUNICATE 组件自动通过，**绝大多数任务事实上只考核 DB**。余下 6 个任务要求 DB 和 COMMUNICATE 同时通过。

---

#### Retail（114 tasks）

```
reward_basis = ["DB", "NL_ASSERTION"]  ← 112/114 任务
reward_basis = ["DB"]                  ← 2/114 任务（task 33、34）

reward = DB_reward × NL_ASSERTION_reward
```

- **DB**：订单状态、用户地址、支付信息等是否与 gold 一致
- **NL_ASSERTION**：由 LLM judge 评判 agent 是否满足自然语言断言（通常是"agent 应告知用户某个金额/结果"）

**实际效果**：114 个任务中有 74 个 `nl_assertions` 为空 → NL_ASSERTION 自动通过，**约 65% 的任务事实上只考核 DB**。另外 40 个任务要求 DB 和 NL_ASSERTION 同时通过（这 40 个任务是通过率的主要瓶颈）。

> **与 airline 的关键区别**：airline 的 COMMUNICATE 是精确子串匹配（确定性、无 LLM 开销）；retail 的 NL_ASSERTION 由 LLM judge 调用（非确定性，有额外 API 费用）。框架默认使用 `EvaluationType.ALL_WITH_NL_ASSERTIONS`，因此 NL_ASSERTION **已在标准评测流程中自动启用**，无需额外参数。

---

#### Telecom（2285 tasks，含 full split）

```
reward_basis = ["ENV_ASSERTION"]           ← 2253/2285 任务
reward_basis = ["ACTION", "ENV_ASSERTION"] ← 32/2285 任务

reward = ENV_ASSERTION_reward
      （或 ACTION_reward × ENV_ASSERTION_reward）
```

- **ENV_ASSERTION**：调用领域定义的断言函数（如 `assert_mobile_data_status`、`assert_internet_speed`），检验环境的**计算状态**而非原始 DB 字段值——比 DB 比较更灵活，可覆盖"流量速度恢复正常"等复合状态
- **ACTION**（少数任务）：检验 agent 是否调用了正确的工具（如 `transfer_to_human_agents` with 正确 reason）

> Telecom **不使用 DB 直接对比**，全部依赖 ENV_ASSERTION，因此 harness H2 规则也无法依赖 DB 字段做精确推断，须直接查询领域环境状态。

---

#### Banking_knowledge（97 tasks）

```
reward_basis = ["DB"]      ← 88/97 任务
reward_basis = ["ACTION"]  ←  9/97 任务

reward = DB_reward（或 ACTION_reward）
```

- **DB**：知识库检索后的 DB 状态（通常是"agent 是否正确地查到了答案并写入了某个字段"）
- **ACTION**（少数任务）：检验 agent 是否执行了正确的工具调用（如在账户争议场景下调用 `transfer_to_human_agents`）

---

### 汇总对照表

| Domain | 主要组件 | 辅助组件 | 实际考核重点 | 是否需要 LLM judge |
|--------|---------|---------|------------|------------------|
| airline | DB | COMMUNICATE（多数任务为空） | 几乎只考 DB | 否 |
| retail | DB | NL_ASSERTION（40% 任务非空） | DB + NL 双重考核 | **是**（NL_ASSERTION） |
| telecom | ENV_ASSERTION | ACTION（极少数） | 环境断言 | 否 |
| banking_knowledge | DB | ACTION（少数） | DB | 否 |

---

## 数据集划分

各 domain 的任务集由 `data/tau2/domains/<domain>/split_tasks.json` 管理。**Harness 规则和 H3 提示仅基于 `train` split 设计**，`test` split 在开发阶段严格保密，不作任何分析或定向优化。

| Domain | train | test | base（全集） | 备注 |
|--------|------:|-----:|------------:|------|
| airline | 30 | 20 | 50 | — |
| retail | 74 | 40 | 114 | — |
| telecom | 74 | 40 | 114 | 另有 `small`（20）和 `full`（2285）split |
| banking_knowledge | **60** | **37** | 97 | 按任务类型分层抽样（seed=42）；`split_tasks.json` 位于 `data/tau2/domains/banking_knowledge/` |

> **重要**：所有针对 train 任务的分析（failure case 挖掘、规则设计、H3 提示撰写）均基于 `--split train` 的运行结果。`test` split 只用于最终评估，不作 cherry-picking。

### Banking_knowledge 数据集特征与 Token 成本

**数据集结构**（`data/tau2/domains/banking_knowledge/`）：
- **97 个任务**（`tasks/task_001.json` ～ `task_097.json`，非连续编号）
- **698 个文档**（`documents/`），平均 247 tokens/篇，最大 1,942 tokens；覆盖信用卡产品、支票账户、借记卡、欺诈政策、支持流程等类别
- **transactional DB**（`db.json`）：用户账户、信用卡、交易、争议等状态

**每任务的 token 估算**（相较 telecom）：

| 指标 | banking_knowledge | telecom |
|------|------------------:|--------:|
| 用户 instruction 平均 tokens | **806**（max 1,704） | **265**（max 324） |
| 系统 policy 基础 tokens | ~1,170（bm25 变体） | ~1,310 |
| 每任务所需文档数 | **平均 9.9**（max 30）| 无文档检索 |
| 所需文档 tokens 之和 | **平均 3,766**（max 10,367）| 0 |
| 预估对话轮数 | **5～8 轮**（知识 Q&A，无复杂多步操作） | **10～20 轮**（含 Hard persona 循环） |
| 预估单次任务总 prompt tokens | **~15,000～25,000** | **~25,000～35,000** |

**结论**：banking_knowledge 每任务 prompt 总量**与 telecom 相当或略少**。但有两个关键差异：
1. **文档检索成本**：使用 `bm25` 等在线检索变体时，每次 `retrieve_knowledge` 调用返回若干文档（avg 247 tokens/篇），随对话积累上下文；使用 `golden_retrieval` 变体则将所需文档全量写入 policy（avg 2,557 tokens），上下文固定但 prompt 起点更大。
2. **对话轮数**：banking 对话较短（5～8 轮），telecom 可因 Hard persona 达 20+ 轮，所以 telecom 累计 token 开销更高。

**推荐**：成本优先可先用 `golden_retrieval`（上下文固定、无检索 API 调用）跑基准；之后用 `bm25` 或 embedding 变体做检索能力对比。

---

## Harness 分层（H2～H5）与运行流程

Harness 在 **assistant 侧工具链**（`Environment.tools`，即 domain 的 `ToolKitBase` 子类）上叠加多层能力。各层可独立开关，常见组合为 H2+H3，或 H2+H3+H4+H5（Retail / Telecom）。

### 各层职责

| 层 | 名称 | 触发时机 | 实现位置 | 机制概要 | 上下文 / Token |
|----|------|----------|----------|----------|----------------|
| **H2** | 前置校验（Pre-validation） | 每次 `use_tool` **执行原工具之前** | `HarnessedToolKitMixin` + 各域 `harness_rules` | `HarnessRule.check(db, **kwargs)` 违反政策则 `raise ValueError` → 环境转为 `ToolMessage(error=True)`，**不执行写操作** | 不增加 system prompt；仅在拦截时多一轮错误回复 |
| **H3** | 工具描述嵌入 | **构造 `tools[]` 时**（每个 LLM 请求的 tools 参数都会带上完整 schema） | `h3_tools.py` 中各域 `H3*ToolDescriptionMixin` | 将短政策要点 **append 到目标工具的 docstring**，经 `Tool` 解析进入 OpenAI schema | **每次请求**重复计入 tools 段总长；应控制每条 hint 篇幅 |
| **H4** | 工具响应注释（Post-annotation） | 工具 **成功返回之后**（与 H2 同一条 `use_tool` 路径） | `HarnessedToolKitMixin` + 各域 `harness_annotators` | `HarnessAnnotator.annotate(db, result, **kwargs)` 返回非空字符串时，拼接到序列化后的 tool 结果末尾 | 仅增加**该轮** tool 消息的 token |
| **H5** | System prompt 增强 | **`build_agent` 一次**（每 task 构建 agent 时） | `runner/build.py` 调用 `h5_policy.py` / `skills.py` / `policy_rag.py` | ① **关键词**或 **BM25** 向 policy 末尾追加片段；② **Skill**：按任务描述 BM25 取 top-k 条技巧，**prepend** 到 policy 前部；③ **`retrieve_policy` 工具模式**：缩短 system、由 agent 按需检索规则段（与 ①② **互斥**） | 注入段落越长，长对话中越易被后续 turns 稀释注意力 |

**设计对照**（摘自 `skills.py` 注释）：H2 按工具、可阻塞、运行时；H3 按工具、被动、每次 API；H4 按工具结果、提示性、响应后；H5 按任务、策略向、构建时一次。

**Replay 与安全**：H2/H4 逻辑**只读 DB 与本次调用的 kwargs/result**，不读对话历史，保证评估器 `set_state` 重放时行为一致。

**User tools**：用户模拟器侧的 `user_tools` **不经过** harness mixin；H2/H3/H4 **仅作用于 assistant `tools`**（Telecom 中诸如 `check_vpn_status` 等若在 user 侧执行则不会被 H4 注释）。

### 端到端流程（从构建到一次 tool call）

```
┌── build_environment(domain, harness_enabled, harness_h3, harness_h4, …)
│     → 选择具体 Toolkit 类（如 H3H4HarnessedRetailTools）
│
├── build_agent(…)
│     ├─ tools ← environment.get_tools()（已是带 H3 patch 的 toolkit 实例）
│     ├─ domain_policy ← environment.get_policy()
│     └─ H5（若 harness_h5 且非 harness_h5_rag_tool）:
│           • Skill: retrieve_skills → format_skills_block → 拼到 policy 前
│           • 或关键词 / policy RAG: get_policy_suffix → 拼到 policy 后
│
└── 每轮 LLM 请求
      tools[] 中含 H3 扩展描述
            │
            ▼
      Agent 产生 tool_call → Environment 路由到 toolkit.use_tool
            │
            ├─ H2: for rule in harness_rules[tool_name]: rule.check(…)
            │        └─ 失败 → 返回 error ToolMessage，原工具不执行
            │
            ├─ 执行领域原工具 → result
            │
            └─ H4: 拼接各 annotator.annotate(…) 非空结果到 result 字符串
```

### 各域是否实现 H4 / H5 Skill 库

| Domain | H2 | H3 | H4 | H5（policy 关键词 / RAG / retrieve_policy） | H5（`skills.py` DOMAIN_SKILLS） |
|--------|:--:|:--:|:--:|:---------------------------------------------:|:------------------------------:|
| **airline** | ✅ | ✅ | ✅（5 个 Annotator） | ✅ | ✅（5 条 Skill） |
| **retail** | ✅ | ✅ | ✅ | ✅ | ✅（10 条 Skill） |
| **telecom** | ✅ | ✅ | ✅ | ✅ | ✅（12 条 Skill） |
| **banking_knowledge** | ❌ | ❌ | ❌ | ❌ | ❌ |

---

## 各 Domain Harness 内容一览

以下为**速查表**；规则与提示的逐条说明见后文「H2 规则详情」「H3」「H5」等节。

### Airline（`src/tau2/harness/airline.py` + `h3_tools.py`）

| 层 | 内容摘要 |
|----|----------|
| **H2** | **A1～A17**，挂载工具：`cancel_reservation`、`update_reservation_flights`、`book_reservation`、`update_reservation_baggages`、`update_reservation_passengers`、`send_certificate`（详见下文分工具列表） |
| **H3** | `_AIRLINE_H3_HINTS`：与上列写工具及 `transfer_to_human_agents` 等关键工具的描述尾部追加政策要点 |
| **H4** | `H4AirlineAnnotationMixin`：`get_user_details`（多预订摘要）、`get_reservation_details`（支付总额分解）、`cancel_reservation`（精确退款额）、`update_reservation_flights`（净支付/退款额）、`book_reservation`（总收费确认） |
| **H5** | `h5_policy.py` 关键词表、`--harness-h5-rag` policy 分段 BM25、`--harness-h5-rag-tool` 的 `retrieve_policy`；**`AIRLINE_SKILLS`**（5 条：`communicate_exact_amounts`、`verify_correct_reservation`、`one_certificate_per_reservation`、`round_trip_include_all_legs`、`check_cancellation_eligibility`） |

**Toolkit 类**：`HarnessedAirlineTools`、`H3AirlineTools`、`H3HarnessedAirlineTools`、`H4AirlineTools`、`H4HarnessedAirlineTools`、`H3H4HarnessedAirlineTools`（由 `airline/environment.py` 按 `harness_enabled`/`harness_h3`/`harness_h4` 组合选用）。

---

### Retail（`src/tau2/harness/retail.py` + `h3_tools.py`）

| 层 | 内容摘要 |
|----|----------|
| **H2** | **R1～R19**，覆盖 `cancel_pending_order`、`modify_pending_order_items`、`modify_pending_order_payment`、`exchange_delivered_order_items`、`return_delivered_order_items`（含空列表、列表长度、重复 item、跨品类、礼品卡余额等） |
| **H3** | `_RETAIL_H3_HINTS`：上述写流程工具 + `modify_pending_order_address`、`modify_user_address`、`transfer_to_human_agents` 等 |
| **H4** | `H4RetailAnnotationMixin.harness_annotators`：`get_order_details`（订单状态/运单/总额等）、`get_product_details`、`get_item_details`、`exchange_delivered_order_items`、`return_delivered_order_items`、`cancel_pending_order`、`modify_user_address`、`modify_pending_order_address` |
| **H5** | 同 Airline 的 policy 模式 + **`skills.py` → `RETAIL_SKILLS`**（10 条，如 `multi_order_scan`、`communicate_exact_amounts`、`no_fake_success` 等） |

**Toolkit 类**：`H4RetailTools`、`H4HarnessedRetailTools`、`H3H4RetailTools`、`H3H4HarnessedRetailTools` 等（由 `retail/environment.py` 按 `harness_enabled` / `harness_h3` / `harness_h4` 组合选用）。

---

### Telecom（`src/tau2/harness/telecom.py` + `h3_tools.py`）

| 层 | 内容摘要 |
|----|----------|
| **H2** | **T1～T6** 业务规则 + **T7～T13** 等：`CustomerIDValidationRule`（T8）、`LineOwnershipRule`（T9）、`ResumeLineStatusCheck`（T10）、`BillOwnershipRule`（T11）、`DisableRoamingAlreadyDisabledRule`（T12）、`PhoneNumberFormatRule`（T13）、`EnableRoamingAlreadyEnabledRule`（T7）等；挂载 `get_customer_by_phone`、`send_payment_request`、`resume_line`、`refuel_data`、`suspend_line`、`enable_roaming`、`disable_roaming` |
| **H3** | `_TELECOM_H3_HINTS`：`send_payment_request`、`resume_line`、`refuel_data`、`suspend_line`、`enable_roaming`、`transfer_to_human_agents` |
| **H4** | `H4TelecomAnnotationMixin`：`get_customer_by_phone`、`get_data_usage`、`get_details_by_id`、`get_bills_for_customer`、`send_payment_request`、`enable_roaming` |
| **H5** | 同 Airline 的 policy 模式 + **`skills.py` → `TELECOM_SKILLS`**（12 条，如 `data_quota_check`、`abroad_enable_roaming`、`mms_wifi_calling`、`sim_pin_plus_billing`、`hard_persona_loop_break` 等） |

**Toolkit 类**：`HarnessedTelecomTools`、`H3HarnessedTelecomTools`、`H4HarnessedTelecomTools`、`H3H4HarnessedTelecomTools` 等（由 `telecom/environment.py` 按标志位选用）。

---

### Banking_knowledge（`src/tau2/domains/banking_knowledge/`）

Harness 层均**未实现**。该 domain 的核心挑战是**知识检索**（从 698 个文档中找到正确依据）而非状态操作校验，因此不设 H2/H3/H4/H5。

| 层 | 内容摘要 |
|----|----------|
| **H2** | ❌ 无 |
| **H3** | ❌ 无 |
| **H4** | ❌ 无 |
| **H5** | ❌ 无 |

**核心差异：`retrieval_variant` 参数**（取代 harness 层控制检索能力）：

| 变体 | 机制 | 特点 |
|------|------|------|
| `bm25`（默认） | BM25 关键词检索；agent 调用 `retrieve_knowledge(query)` | 检索结果动态累积，上下文随对话增长 |
| `golden_retrieval` | 任务所需文档直接写入 policy（oracle，上限） | policy 固定为 avg 2,557 tokens；无检索调用 |
| `full_kb` | 全量 698 篇文档写入 policy | policy 极大，不推荐实际使用 |
| `grep_only` | 精确字符串匹配检索 | 较 BM25 精确但召回有限 |
| `qwen_embeddings` / `openai_embeddings` | 语义向量检索 | 需向量服务；质量通常优于 BM25 |
| `*_reranker` | 检索后 rerank 二阶段 | 精度更高但延迟增加 |
| `qwen_embeddings_grep` / `bm25_grep` | 向量/BM25 + grep 融合 | 实验变体 |

**运行方式**（`retrieval_config` 通过 CLI `--retrieval-config` 或 `TextRunConfig.retrieval_config` 传入，`eval_harness.py` 目前不直接支持此参数，需用主 CLI 或 Python API）：
```bash
# 主 CLI：使用 golden_retrieval（上界基准，文档全量内联）
tau2 run --domain banking_knowledge --retrieval-config golden_retrieval \
    --agent-llm <model> --user-llm <model>

# 主 CLI：使用 bm25 在线检索
tau2 run --domain banking_knowledge --retrieval-config bm25 \
    --agent-llm <model> --user-llm <model>
```

Python API：
```python
config = TextRunConfig(
    domain="banking_knowledge",
    retrieval_config="golden_retrieval",   # 或 "bm25"
    ...
)
```

---

## 快速使用

```bash
# 只开 H2（运行时校验）
tau2 run --domain airline --harness-enabled --agent-llm <model> --user-llm <model>

# 只开 H3（工具描述嵌入）
tau2 run --domain airline --harness-h3 --agent-llm <model> --user-llm <model>

# H2 + H3（推荐组合：运行时拦截 + 描述层提醒）
tau2 run --domain airline --harness-enabled --harness-h3 --agent-llm <model> --user-llm <model>

# H2 + H3 + H5（全量：运行时拦截 + 描述层提醒 + 系统提示注入）
tau2 run --domain airline --harness-enabled --harness-h3 --harness-h5 \
  --agent-llm <model> --user-llm <model>

# Retail / Telecom：再加 H4（工具返回尾部注释）
tau2 run --domain retail --harness-enabled --harness-h3 --harness-h4 --harness-h5 \
  --agent-llm <model> --user-llm <model>

# Airline：H2 + H3 + H4 + H5（新增 H4 支付注释 + H5 Skill 库）
tau2 run --domain airline --harness-enabled --harness-h3 --harness-h4 --harness-h5 \
  --agent-llm <model> --user-llm <model>

# Banking_knowledge：无 harness，通过主 CLI 指定检索变体
tau2 run --domain banking_knowledge --retrieval-config golden_retrieval \
  --agent-llm <model> --user-llm <model>
```

批量对比评测脚本 `scripts/eval_harness.py`：对应标志为 `--harness`（H2）、`--h3`、`--h4`、`--h5`（及 `--failed-from` / `--task-indices` 等任务筛选）。

Python API：

```python
from tau2.data_model.simulation import TextRunConfig

config = TextRunConfig(
    domain="airline",
    harness_enabled=True,   # H2
    harness_h3=True,        # H3
    harness_h4=False,       # H4（airline / retail / telecom 均已实现）
    harness_h5=True,        # H5
    ...
)
```

---

## 架构

### 核心接口（`src/tau2/harness/base.py`）

```python
class HarnessRule(Protocol):
    tool_name: str
    def check(self, db: DB, **kwargs) -> None:
        """拦截时 raise ValueError，自动转为 ToolMessage(error=True)"""

class HarnessAnnotator(Protocol):
    tool_name: str
    def annotate(self, db: DB, result: Any, **kwargs) -> str | None:
        """成功返回后追加短注释；返回 None 表示不追加"""

class HarnessedToolKitMixin:
    harness_rules: dict[str, list[HarnessRule]] = {}
    harness_annotators: dict[str, list[HarnessAnnotator]] = {}

    def use_tool(self, tool_name: str, **kwargs) -> Any:
        for rule in self.harness_rules.get(tool_name, []):
            rule.check(self.db, **kwargs)   # 拦截则 raise
        result = super().use_tool(tool_name, **kwargs)
        # H4：对每个 harness_annotators[tool_name] 调用 annotate，将非空字符串
        #     追加到序列化后的 tool 返回值（见 base.py 完整实现）
        ...
```

（省略号处逻辑见 `src/tau2/harness/base.py` 中 `HarnessedToolKitMixin.use_tool`。）

### Replay 安全性

**H2**：规则**只读取 DB 与本次调用的 kwargs**，不依赖对话历史。**H4**：annotator **只读取 DB、kwargs 与本次 tool 的 result**，无副作用。评估器在 `set_state()` 重放轨迹时，DB 与调用参数一致，因此拦截与注释结果与仿真时相同，不会破坏评估。

---

## H2 规则详情

### Airline Domain（`src/tau2/harness/airline.py`）

激活方式：`--domain airline --harness-enabled`

**`cancel_reservation`**
- **A1/A2** `CancelFlightRule`：双重校验
  - 任意航段 `status ∈ {flying, landed}` → 阻止，要求转人工
  - 以下 4 条至少满足 1 条才可取消，否则逐条诊断后拦截：
    1. 订单在过去 24 小时内创建（`created_at` vs `SIMULATION_TIME`）
    2. 航空公司取消了该预订内的某个航班
    3. 舱位为 `business`
    4. 已购买旅行险（且原因涵盖健康/天气）
  - basic_economy 命中时额外提示"可先升舱再取消"

**`update_reservation_flights`**
- **A3** `BasicEconomyFlightChangeRule`：
  - `basic_economy` 同舱位改航班 → 拦截
  - `basic_economy` 升舱同时换不同航班 → 拦截；唯一合法路径：同航班号升舱，或 cancel+rebook
- **A5** `NoOriginDestChangeRule`：第一段 origin 和最后一段 destination 不可变；round_trip 忘提交返程 → 专项提示
- **A6** `CabinChangeOnFlyingReservationRule`：任意航段已飞/着陆时禁止改舱
- **A12** `FlightUpdateNoCertificateRule`：改航班支付方式只能是 gift_card / credit_card，不接受 certificate

**`book_reservation`**
- **A8** `BookReservationPaymentRule`：≤ 1 certificate，≤ 1 credit_card，≤ 3 gift_card
- **A9** `MaxPassengersRule`：每笔预订最多 5 人
- **A10** `FlightStatusBookableRule`：所有航班必须为 `available`
- **A11** `BaggageAllowanceRule`：`nonfree_baggages = total − free_allowance`（按会员等级×舱位计算）
- **A15** `BookReservationPaymentTotalRule`：`sum(payment amounts) = sum(flight.price[cabin]) × num_pax`；自动给出正确总价，防止 agent 在错误数字上死循环

**`update_reservation_baggages`**
- **A7** `NoBaggageDecreaseRule`：行李数只增不减
- **A16** `BaggageAllowanceUpdateRule`：`nonfree_baggages = total − free_allowance`（与 A11 逻辑相同，覆盖更新路径；从 reservation 中读取会员/舱位）

**`update_reservation_passengers`**
- **A13** `NoPassengerCountChangeRule`：乘客人数在预订后永久锁定，即使人工客服也不能修改

**`send_certificate`**
- **A4** `CertificateEligibilityRule`：regular 会员 + 无险 + 无 business 舱 → 拒绝补偿
- **A17** `CertificateFlightStatusRule`：所有活跃预订中无任何航班处于 `delayed` / `cancelled` 状态 → 拦截；防止无真实延误/取消时的幻象补偿
- **A14** `CertificateAmountCapRule`：金额必须为 $50 正整数倍；上限 = $100 × 最大预订乘客数（最多 $500）

---

### Retail Domain（`src/tau2/harness/retail.py`）

激活方式：`--domain retail --harness-enabled`

**`cancel_pending_order`**
- **R1** `CancelPendingOrderRule`：只允许 `pending`；**单独文案**：`cancelled`（不可再取消）、`processed`、`pending (item modified)`；delivered 等提示退货流程
- **R5** `CancelReasonRule`：reason 必须为 `'no longer needed'` 或 `'ordered by mistake'`，其他措辞一律拦截并列出合法选项

**`modify_pending_order_items`**
- **R2** `ModifyPendingOrderRule`：只允许 `pending`；`pending (item modified)` → 拦截（每单仅可改一次）；**单独文案**：`cancelled` / `delivered` / `processed` / `exchange requested` / `return requested`
- **R17** `NonEmptyItemIdsRule`：`item_ids` 不得为空
- **R18** `NonEmptyNewItemIdsRule`：`new_item_ids` 若传入则不得为空
- **R19** `ItemIdsNewItemIdsParityRule`：两列表长度必须一致
- **R8** `ModifyItemsProductTypeRule`：每个 new_item_id 必须是同一商品的不同变体（variant）；尝试跨商品类型修改（如 shirt → shoes）→ 拦截，并指明原始商品类型与目标商品类型
- **R10** `ModifySameItemRule`：new_item_id 不能与被替换的 old_item_id 相同；相同则拦截并提示先用 `get_product_details` 查找其他变体
- **R16** `ValidateNewItemIdExistsRule`：每个 `new_item_id` 须在商品目录中真实存在；拦截伪造 ID，提示通过 `get_product_details` 查询

**`modify_pending_order_payment`**
- **R6** `ModifyPaymentSameMethodRule`：新支付方式必须与当前不同；相同 → 拦截
- **R7** `ModifyPaymentGiftCardBalRule`：若切换至礼品卡，余额必须覆盖订单总额；余额不足 → 拦截并给出金额差值

**`exchange_delivered_order_items`**
- **R3** `ExchangeDeliveredOrderRule`：只允许 `delivered`；**`pending` 状态时错误消息明确提示改用 `modify_pending_order_items`**；`exchange requested` / `return requested` → 拦截
- **R17** / **R18** / **R19**：同 modify 路径（非空 + 列表长度一致）
- **R9** `ExchangeItemsProductTypeRule`：与 R8 逻辑相同，适用于换货路径；跨商品类型换货 → 拦截
- **R11** `ExchangeSameItemRule`：new_item_id 不能与被换出的 old_item_id 相同；相同则拦截并提示查找其他变体
- **R16** `ValidateNewItemIdExistsRule`：同 modify 路径，`new_item_id` 须为真实商品 ID
- **R13** `ValidateExchangeItemsExistRule`：item_ids 必须存在于该订单
- **R15** `NoDuplicateItemIdsRule`：item_ids 中不能出现超出订单实际数量的重复项

**`return_delivered_order_items`**
- **R4** `ReturnDeliveredOrderRule`：只允许 `delivered`；**`pending` 状态时提示改用 `cancel_pending_order`**；退款须为原支付方式或礼品卡
- **R17** `NonEmptyItemIdsRule`：`item_ids` 不得为空
- **R14** `ValidateReturnItemsExistRule`：item_ids 必须存在于该订单
- **R15** `NoDuplicateItemIdsRule`：防止重复 item_id

---

### Telecom Domain（`src/tau2/harness/telecom.py`）

激活方式：`--domain telecom --harness-enabled`；H4：`--harness-h4`（需与环境里 `harness_h3`/`harness_enabled` 组合，见 `telecom/environment.py`）。

**`get_customer_by_phone`**
- **T13** `PhoneNumberFormatRule`：号码须为 `NXX-NXX-XXXX`（10 位 + 短横线），拒绝带国家码等格式，避免必然查单失败

**`send_payment_request`**
- **T8** `CustomerIDValidationRule`：客户 ID 格式与存在性
- **T11** `BillOwnershipRule`：`bill_id` 须属于该客户
- **T1** `SendPaymentRequestOverdueRule`：账单须为 `OVERDUE`
- **T6** `SendPaymentOneAtATimeRule`：同一客户同时仅允许一张 `AWAITING_PAYMENT` 账单

**`resume_line`**
- **T8** / **T9** 同上（客户 ID、线路归属）
- **T10** `ResumeLineStatusCheck`：线路当前须为 `Suspended` 才可 resume
- **T2** `ResumeLineEligibilityRule`：无 `OVERDUE` 账单且合同未过期

**`refuel_data`**
- **T8** / **T9**
- **T3** `RefuelDataLimitRule`：单次 ≤ 2 GB
- **T5** `RefuelDataActiveLineRule`：线路须为 `Active`

**`suspend_line`**
- **T8** / **T9**
- **T4** `SuspendLineStatusRule`：线路须为 `Active` 才可暂停

**`enable_roaming`**
- **T8** / **T9**
- **T7** `EnableRoamingAlreadyEnabledRule`：账户漫游已开时拦截重复调用，引导设备侧开关

**`disable_roaming`**
- **T8** / **T9**
- **T12** `DisableRoamingAlreadyDisabledRule`：已关闭时拦截无意义调用

---

### Telecom H4（`--harness-h4`，与 H2/H3 叠加）

挂载于 `H4TelecomAnnotationMixin.harness_annotators`：

| 工具 | Annotator 作用（摘要） |
|------|------------------------|
| `get_customer_by_phone` | `CustomerPhoneLineAnnotator`：多线路时标出与来电号码一致的 `line_id` |
| `get_data_usage` | `DataQuotaAnnotator`：配额用尽时提示 `refuel_data` |
| `get_details_by_id`（Line） | `LineStatusAnnotator`：漫游关、配额用尽、暂停/合同到期等工作流提示 |
| `get_bills_for_customer` | `BillsOverdueAnnotator`：标出逾期账单与付款链路 |
| `send_payment_request` | `PaymentWorkflowAnnotator`：提醒后续 `make_payment` → `resume_line` 等 |
| `enable_roaming` | `EnableRoamingResultAnnotator`：区分「已开启」与「新开启」，提醒设备 toggle |

---

## H3：工具描述政策嵌入

实现文件：`src/tau2/harness/h3_tools.py`

激活方式：`--domain airline --harness-h3`（可与 `--harness-enabled` 和/或 `--harness-h5` 自由组合）

### 为什么工具描述"每次都能看见"

LLM API 的请求体分为两个独立部分：

| 部分 | 位置 | 随对话增长的变化 |
|------|------|----------------|
| `messages[]` | 对话历史（含 system prompt） | 越来越长，早期内容被推至注意力尾部 |
| `tools[]` | 独立参数，不在 messages 里 | **每次请求重新发送**，始终以完整权重进入注意力 |

因此把 policy 约束写进工具描述，可以确保无论对话已经多长，这些约束**永远处于模型的"视野中心"**，不会因为长对话而被"遗忘"。

### 覆盖范围

**Airline 关键工具 H3 约束清单**（实现于 `_AIRLINE_H3_HINTS`）

| 工具 | 嵌入的主要约束 |
|------|--------------|
| `cancel_reservation` | 取消 4 条件（任一满足）；已飞→转人工；basic_economy 升舱提示 |
| `update_reservation_flights` | **升降舱均合法**（含降至 basic_economy）；basic_economy 限制仅针对改航班号；不可改 origin/dest；round_trip 须含返程；支付限制 |
| `book_reservation` | 支付上限（1 cert / 1 CC / 3 GC）；最多 5 人；航班必须 available；免费行李计算公式 |
| `update_reservation_baggages` | 只增不减；不得主动加用户未请求的行李 |
| `update_reservation_passengers` | 人数不可变；只可改信息不可增减人 |
| `send_certificate` | 资格条件（silver/gold/business/insurance）；金额 $50 倍数；取消 $100×人/延误 $50×人；上限 $500 |
| `transfer_to_human_agents` | 列出所有在 scope 内的操作（含降舱/升舱/取消等），防止 premature transfer；仅允许转人工：已飞/着陆航段需取消 |

**Retail 关键工具 H3 约束清单**（实现于 `_RETAIL_H3_HINTS`）

| 工具 | 嵌入的主要约束 |
|------|--------------|
| `cancel_pending_order` | 只允许 `pending`；reason 只能是两个固定选项（含映射示例）；**退款目标锁定为原支付方式，不可重定向到礼品卡** |
| `modify_pending_order_items` | pending 换货走本工具；status / ONCE；item_ids 与 new_item_ids（`get_product_details`）；确认后同轮出 tool call |
| `modify_pending_order_payment` | 新旧方式须不同；单一支付；礼品卡余额须足额 |
| `modify_pending_order_address` | 只限 `pending`/`pending (item modified)`；**提醒需分别调用 `modify_user_address` 来同步用户默认地址** |
| `exchange_delivered_order_items` | delivered only；order/item/new_item 规则与多订单每单调用；确认后同轮 tool call |
| `return_delivered_order_items` | delivered / pending 分流；每单批量；退款方式限制；确认后同轮 tool call |
| `modify_user_address` | 只更新用户默认地址；**MANDATORY：调用后 MUST 检查 pending orders 并同步地址（强制，不可跳过）** |
| `transfer_to_human_agents` | 列出所有在 scope 内操作（含通过 `get_order_details` 查找订单地址），仅撤单/新订单/跨用户才需转人工 |

**Telecom 关键工具 H3 约束清单**（实现于 `_TELECOM_H3_HINTS`）

| 工具 | 嵌入的主要约束 |
|------|--------------|
| `send_payment_request` | 必须先确认账单为 OVERDUE；每次只能一张 Awaiting Payment；付款后验证状态变为 PAID |
| `resume_line` | 需零逾期账单 + 合同未过期；恢复后提示用户重启设备 |
| `refuel_data` | 线路需 Active；最多 2 GB/次；先确认数据已超限再补充 |
| `suspend_line` | 线路需 Active；$5/月保留费；最长 6 个月；需用户确认 |
| `enable_roaming` | 免费；账户侧 + 设备侧两步；若提示已开启则仅设备 toggle |
| `transfer_to_human_agents` | 须**显式 tool call**，不可仅在对话里写「转人工」；合同到期 / SIM PIN 需到店等场景 |

### 扩展到其他 Domain

创建新的 mixin 继承 `H3ToolDescriptionMixin`，设置 `_h3_hints` 即可：

```python
from tau2.harness.h3_tools import H3ToolDescriptionMixin

class H3RetailToolDescriptionMixin(H3ToolDescriptionMixin):
    _h3_hints = {
        "cancel_pending_order": "POLICY: Only pending orders can be cancelled...",
        ...
    }
```

---

## H5：System Prompt 动态增强

实现文件：`src/tau2/harness/h5_policy.py`（关键词 / 构建时 RAG）、`src/tau2/harness/policy_rag.py`（分段与检索）、`src/tau2/harness/skills.py`（Retail / Telecom 的 **Skill 库** + BM25 检索，由 `build.py` 在 `harness_h5=True` 且非 RAG-tool 模式时注入）

### 模式三：`retrieve_policy` 工具（**默认关闭**，可随时开关）

**CLI**：`tau2 run ... --harness-h5-rag-tool`  
**Config**：`harness_h5_rag_tool=True`（可选配合 `harness_h5_rag_top_k`）

行为：

1. **系统 prompt 只保留结构化部分**：标题、角色与行为准则、**Domain Basic**（用户/航班/订单等数据模型说明）。
2. **不注入完整 policy**，也不在构建时追加 RAG 片段。
3. 向 agent 工具列表追加 **`retrieve_policy(query)`**：仅对**程序性规则**段落（`## Book flight`、`## Modify flight` 等）做 BM25 检索，按需返回 top-k 段。

与模式一/二**互斥**：若开启 `harness_h5_rag_tool`，`build_agent` **优先**走工具模式，不再执行 H5 关键词/RAG 后缀注入。

无 `## Domain Basic` 的 domain（如 mock）：前缀尽量保留标题行，规则库为全文。

---

H5 构建时注入支持两种子模式，通过 `--harness-h5-rag`（或 `harness_h5_rag`）切换：

### 模式一：关键词匹配（默认）

根据任务 `description` 中的关键词，将预置 policy 片段追加到系统提示末尾。

```python
suffix = get_policy_suffix(domain="airline", task_description=task.description)
domain_policy = base_policy + "\n" + suffix
```

**Airline 关键词映射表**（仅基于 train split 设计，不泄露 test 数据）

| 关键词 | 注入内容 |
|--------|---------|
| `cancel`, `refund`, `取消`, `退款` | 取消航班完整条款（4个条件 + 已飞转人工） |
| `basic economy`, `change flight` | Basic economy 改航班限制说明 |
| `compensat`, `certificate`, `补偿` | 补偿资格条件（会员/保险/商务舱） |

**Retail / Telecom** 关键词同理（见 `h5_policy.py`）。

### 模式二：Policy RAG 检索（`--h5-rag`）

**实现文件**：`src/tau2/harness/policy_rag.py`

将 policy 中**程序性规则部分**（排除 preamble + `## Domain Basic`）按 `##` 分段，BM25-lite 对任务描述打分，检索 top-k 段**追加**到完整 policy 末尾：

```python
# h5_policy.py
suffix = get_policy_suffix(
    domain="airline",
    task_description=task_desc,   # reason_for_call + task_instructions
    use_rag=True,
    policy_text=full_policy_md,
    rag_top_k=3,
)
```

```bash
# CLI 用法
uv run python scripts/eval_harness.py --split train --harness --h5-rag --h5-rag-top-k 3
```

**优势对比**

| 特性 | 关键词模式 | RAG 模式 |
|------|-----------|---------|
| 需要维护关键词表 | 是（需人工设计） | 否（自动） |
| 跨域通用性 | 各 domain 独立 | 任意 domain |
| 检索精度 | 关键词匹配 | BM25 语义相关度 |
| 无外部依赖 | ✅ | ✅（纯 Python） |
| 可解释性 | 高（显式映射） | 中（评分透明） |

### 设计约束

- 模式一/二：在**完整 policy 文本**末尾**追加**片段（不替换前文）
- 模式三：**替换**为「结构化前缀 + `retrieve_policy` 使用说明」，规则由工具返回
- RAG 检索使用 `task.user_scenario.instructions`（`reason_for_call` + `task_instructions`）作为 query，信息更丰富
- H5 不涉及 DB 操作，与 replay 安全性无关
- `policy_rag.PolicyRetriever` 使用 `@lru_cache` 缓存，同 domain 多次调用只构建一次

---

## 覆盖指标

**Airline H2 规则（17条）**

| 规则ID | 工具 | 覆盖点 |
|--------|------|--------|
| A1/A2 | `cancel_reservation` | 已飞拦截 + 4条件资格校验（含逐条诊断） |
| A3 | `update_reservation_flights` | basic_economy 同舱改航班；升舱+换航班须 cancel+rebook |
| A4 | `send_certificate` | 资格校验（regular+无险+非商务拦截） |
| A5 | `update_reservation_flights` | 不允许改 origin/destination/trip_type；round_trip 缺返程提示 |
| A6 | `update_reservation_flights` | 已飞航段不允许改舱 |
| A7 | `update_reservation_baggages` | 行李只增不减 |
| A8 | `book_reservation` | 支付方式上限（≤1 cert, ≤1 CC, ≤3 GC） |
| A9 | `book_reservation` | 乘客上限 5 人 |
| A10 | `book_reservation` | 航班状态必须为 available |
| A11 | `book_reservation` | 免费行李额度校验（nonfree = total − free_allowance） |
| A12 | `update_reservation_flights` | 改航班支付不能用证书 |
| A13 | `update_reservation_passengers` | 乘客人数不可变更 |
| A14 | `send_certificate` | 补偿金额上限（$50 倍数，≤ $100×最大乘客数） |
| A15 | `book_reservation` | 支付总额 = 各航班票价 × 乘客数；直接给出正确总价 |
| A16 | `update_reservation_baggages` | 免费行李额度校验（update 路径，与 A11 镜像） |
| A17 | `send_certificate` | 活跃预订中须存在真实 delayed/cancelled 航班，否则拦截 |

**Retail H2（19条，R1~R19）**

| 规则ID | 工具 | 覆盖点 |
|--------|------|--------|
| R1 | `cancel_pending_order` | 非 pending 拦截；**单独提示**：已 `cancelled` / `processed` / `pending (item modified)`；delivered 等提示改用 return |
| R2 | `modify_pending_order_items` | 已改过一次拦截；**单独提示**：`cancelled` / `delivered` / `processed` / `exchange requested` / `return requested` |
| R3 | `exchange_delivered_order_items` | 非 delivered 拦截；**pending 时主动提示改用 modify_pending_order_items** |
| R4 | `return_delivered_order_items` | 非 delivered 拦截；**pending 时主动提示改用 cancel_pending_order**；退款须为原支付方式或礼品卡 |
| R5 | `cancel_pending_order` | reason 必须为两个合法选项之一 |
| R6 | `modify_pending_order_payment` | 新支付方式须不同于当前 |
| R7 | `modify_pending_order_payment` | 礼品卡余额须 ≥ 订单总额 |
| R8 | `modify_pending_order_items` | 不可跨商品类型修改（须同一 product_id 下的 variant） |
| R9 | `exchange_delivered_order_items` | 不可跨商品类型换货（须同一 product_id 下的 variant） |
| R10 | `modify_pending_order_items` | new_item_id 不能与 old_item_id 相同；拦截并提示查询变体 |
| R11 | `exchange_delivered_order_items` | new_item_id 不能与 old_item_id 相同；拦截并提示查询变体 |
| R12 | `modify_pending_order_items` | item_ids 必须确实存在于该订单；列出可用 item_id 辅助纠错 |
| R13 | `exchange_delivered_order_items` | item_ids 必须确实存在于该订单；列出可用 item_id 辅助纠错 |
| R14 | `return_delivered_order_items` | item_ids 必须确实存在于该订单；列出可用 item_id 辅助纠错 |
| R15 | exchange / return / modify | item_ids 不能超出订单中该 item 的实际数量；防止重复 item_id 导致"幻象批量"操作 |
| R16 | modify / exchange | new_item_ids 必须在商品目录中真实存在；拦截如 `X_new_expensive` 等伪造 ID，并提示通过 get_product_details 查询 |
| R17 | modify / exchange / return | `item_ids` 不得为空列表 |
| R18 | modify / exchange | `new_item_ids` 若传入则不得为空列表 |
| R19 | modify / exchange | `len(item_ids)` 必须等于 `len(new_item_ids)`（避免 zip 静默截断） |

**Telecom H2（T1~T13 等，按工具聚合）**

| 规则ID | 工具 | 覆盖点 |
|--------|------|--------|
| T1 | `send_payment_request` | 账单须为 OVERDUE |
| T2 | `resume_line` | 无逾期账单 + 合同未到期 |
| T3 | `refuel_data` | 单次补充 ≤ 2 GB |
| T4 | `suspend_line` | 线路须为 Active 才可暂停 |
| T5 | `refuel_data` | 线路须为 Active 才可 refuel |
| T6 | `send_payment_request` | 每客户同时仅一张 AWAITING_PAYMENT |
| T7 | `enable_roaming` | 已开启则拦截，引导设备侧 |
| T8 | 多个写工具 | `CustomerIDValidationRule` |
| T9 | 多个写工具 | `LineOwnershipRule`（line 属于客户） |
| T10 | `resume_line` | 当前须为 Suspended |
| T11 | `send_payment_request` | `BillOwnershipRule` |
| T12 | `disable_roaming` | 已关闭则拦截 |
| T13 | `get_customer_by_phone` | 电话号码格式 `NXX-NXX-XXXX` |

---

## False Positive 验证

对 airline **train split（30条）和 test split（20条）** 全部执行了 gold actions 顺序重放测试：

| Split | Write gold actions | False positive |
|-------|--------------------|---------------|
| train（30 tasks） | 53（含新 6 条规则） | **0** |
| test（20 tasks） | 23 | **0** |

验证方法：对每个 task，按顺序在同一 DB 实例上执行所有 gold actions，确认 harness 不拦截任何合法操作。顺序执行是关键（Task 32 中先升舱使 cabin 变为 economy，后续改航班时新规则已不触发）。

```bash
# 复现验证（参考 scripts/eval_harness.py）
uv run python scripts/eval_harness.py --trials 1 --num-tasks 30 --save-to fp_check
```

---

## 扩展：添加新规则

1. 在对应的 `src/tau2/harness/<domain>.py` 中新增一个 rule 类：

```python
class MyNewRule:
    tool_name = "my_tool"

    def check(self, db: MyDB, arg1: str, **_: Any) -> None:
        # 只读 DB，不依赖对话历史
        if some_condition(db, arg1):
            raise ValueError("清晰的拒绝原因，帮助 agent 理解并纠正")
```

2. 注册到 `Harnessed<Domain>Tools.harness_rules`：

```python
class HarnessedMyDomainTools(HarnessedToolKitMixin, MyDomainTools):
    harness_rules = {
        "my_tool": [MyNewRule()],
        ...
    }
```

3. 若涉及时间判断，使用 domain 的仿真时间常量（如 `SIMULATION_TIME`），禁止使用 `datetime.now()`。

4. 在 train split 上验证无 false positive 后再部署到 test 评估。

---

## 自动化失败分析流水线

实现文件：`scripts/harness_failure_report.py`

### 流程概览

```
①  运行评测（可选，也可复用已有结果）
         ↓
②  加载 results.json，提取 reward < 1 的失败仿真
         ↓
③  每个失败任务：构造对话摘要 → 调用 Analyst LLM 分类
         ↓
④  聚合分类结果 → 生成 Markdown 报告
```

### 使用的 LLM

| 角色 | 默认模型 | 说明 |
|------|----------|------|
| **Agent LLM**（被评测对象） | `openai/Qwen/Qwen3-4B-Instruct`（本地 vLLM） | 4B 小模型，通过 `http://localhost:30000/v1` 接入 |
| **User LLM**（用户模拟器） | `openai/deepseek-v3-2-251201` | DeepSeek-V3，通过 LiteLLM 路由 |
| **Analyst LLM**（失败分析） | `openai/deepseek-v3-2-251201` | 同 User LLM；可用 `--analyst-llm` 替换 |

> 通过 `--analyst-llm` 可指定任何 LiteLLM 支持的模型用于失败分析，例如 `openai/gpt-4.1` 获得更高分析精度。

### 失败分类体系（10 类）

| 代码 | 含义 |
|------|------|
| `POLICY_HALLUCINATION` | Agent 捏造了不存在的 policy 规则 |
| `PREMATURE_TRANSFER` | 不应转人工却提前转人工 |
| `WRONG_TOOL_ARGS` | 工具正确但参数错误（金额、ID 等） |
| `WRONG_TOOL_CHOICE` | 调用了错误工具或跳过了必需工具 |
| `NO_HARNESS_RECOVERY` | Harness 已拦截但 agent 未能自我恢复 |
| `INCOMPLETE_TASK` | 完成了部分任务但遗漏了必要步骤 |
| `WRONG_DB_STATE` | 未核查 DB 状态就行动（基于错误假设） |
| `USER_INSTRUCTION_IGNORED` | 忽略或误解了用户的明确指令 |
| `CONFIRM_MISSING` | 执行写操作前未获得用户明确确认 |
| `OTHER` | 不符合上述任何类别 |

### 报告内容

生成 `<run_dir>/failure_report/report.md`，包含：

1. **总体通过率** — passed / total
2. **失败类别分布表** — 各类别计数与占比
3. **Harness 可介入比例** — 多少失败可通过 H2 / H3 / BOTH 预防
4. **具体改进建议** — Analyst LLM 为每个失败给出可直接实现的 harness 规则建议
5. **每任务失败详情表** — task_id、分类、置信度、根因摘要
6. **完整 triage JSON**（可供程序二次处理）

### 快速用法

```bash
# 测试（3 个任务，快速验证脚本正确性）
uv run python scripts/harness_failure_report.py \
    --domain airline --split train --harness --h3 --num-tasks 3

# 完整训练集分析
uv run python scripts/harness_failure_report.py \
    --domain airline --split train --harness --h3

# 复用已有评测结果（跳过评测，直接分析）
uv run python scripts/harness_failure_report.py \
    --run-dir data/simulations/eval_train_harness_h3_20260407_201921

# Retail / Telecom 域
uv run python scripts/harness_failure_report.py \
    --domain retail --split train --harness --h3
uv run python scripts/harness_failure_report.py \
    --domain telecom --split train --harness --h3

# 控制分析成本：只 triage 前 N 个失败
uv run python scripts/harness_failure_report.py \
    --domain airline --split train --harness --h3 --max-triage 10
```

---

## 文件索引

```
src/tau2/harness/
├── __init__.py          # 模块入口（导出 HarnessRule、HarnessedToolKitMixin、H3 mixins）
├── base.py              # HarnessRule / HarnessAnnotator + HarnessedToolKitMixin（H2+H4 管道）
├── airline.py           # A1~A17 + Harnessed / H3 / H3Harnessed（无 H4）
├── retail.py            # R1~R19 + H4RetailAnnotationMixin + H3/H4/H3H4 组合类
├── telecom.py           # T1~T13 等 + H4TelecomAnnotationMixin + H3/H4/H3H4 组合类
├── h3_tools.py          # H3ToolDescriptionMixin + 各域 _*_H3_HINTS + _append_hint_to_tool
├── skills.py            # H5 Skill 库（Retail / Telecom）+ retrieve_skills / format_skills_block
├── h5_policy.py         # H5 关键词映射 + get_policy_suffix()（支持 RAG 模式）
└── policy_rag.py        # PolicyRetriever（BM25-lite）+ get_policy_chunks()

src/tau2/domains/<domain>/environment.py   # harness_enabled / harness_h3 / harness_h4（retail、telecom）
src/tau2/domains/airline/utils.py          # SIMULATION_TIME 常量
src/tau2/runner/build.py                   # harness_* 传环境 + H5（policy / skills / rag_tool）注入 agent
src/tau2/data_model/simulation.py          # harness_enabled / harness_h3 / harness_h4 / harness_h5 等
src/tau2/cli.py                            # --harness-enabled / --harness-h3 / --harness-h4 / --harness-h5

scripts/eval_harness.py                    # 对比评测脚本（airline / retail / telecom）
scripts/harness_failure_report.py          # 自动化失败分析流水线（run → triage → report）
```
