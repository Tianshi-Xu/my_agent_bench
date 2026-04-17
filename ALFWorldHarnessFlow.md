# ALFWorld Harness 触发架构（按一次推理流程）

本文档描述当前仓库里 **已实现** 的 ALFWorld harness，在一次正常推理中会在什么阶段触发、触发哪些能力。

## 0. 开关层（是否启用）

任务配置入口在 `configs/tasks/alfworld.yaml`：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `enabled` | bool | false | 总开关，false 时其余开关全部无效 |
| `h2` | bool | true | 前置校验（Action Gate） |
| `h3` | bool | true | 工具描述嵌入 |
| `h4` | bool | true | 流程监控（世界模型 + 子目标状态机 + 停滞检测） |
| `h5` | bool | true | 目标导向指导注入 |
| `h6` | bool | true | 步数预算管理 |
| `h5_top_k` | int | 1 | H5 冷启动最多注入几条技能 |

### 全量配置参数参考

`ALFWorldHarnessConfig` 的完整默认值（`src/server/harness/alfworld.py`）：

| 参数 | 默认值 | 所属层 | 说明 |
|------|-------|-------|------|
| `action_similarity_threshold` | 0.55 | H2 | 同动词候选相似度阈值，低于此值视为 invalid |
| `never_block_prefixes` | `("go to ", "open ", "close ")` | H2 | 这些前缀动作即使 invalid 也不阻断（探索类动作优先） |
| `invalid_block_after` | 2 | H2 | 连续 N 次 invalid 后阻断 |
| `h2_empty_turn_threshold` | 2 | H2 | 连续 N 轮空 action 后注入提醒 |
| `h3_max_words` | 20 | H3 | 工具描述 hint 词数上限 |
| `h4_stall_window` | 4 | H4 | 停滞检测滑动窗口大小（轮数） |
| `h4_min_rounds_before_stall` | 8 | H4 | 最少轮数后才启用停滞检测（冷启动保护） |
| `h4_soft_intervention_rounds` | 2 | H4 | 进入硬干预前的最大软干预次数 |
| `h4_post_put_grace` | 4 | H4 | PUT 成功后宽限轮数，期间不触发停滞检测 |
| `h4_nothing_happens_window` | 4 | H4 | "Nothing happens" 循环检测观测窗口 |
| `h4_nothing_happens_threshold` | 3 | H4 | 触发干预所需的最少 "nothing happens" 次数 |
| `h5_top_k` | 1 | H5 | 冷启动最多注入的技能条数 |
| `h5_cold_start_max_words` | 35 | H5 | 冷启动每条技能词数上限 |
| `h5_step_hint_max_words` | 25 | H5 | 每步 hint 词数上限 |
| `h6_warn_threshold` | 7 | H6 | 剩余步数 < N 且在 FIND 时注入紧急提示 |
| `h6_force_threshold` | 4 | H6 | 剩余步数 < N 且条件满足时强制 PUT |
| `h6_warn_threshold_multistep` | 12 | H6 | pick_two_obj / pick_cool 的更高预警阈值 |

---

## 层级设计总览

```
Episode 开始: H0 → H3 → H5(cold-start)
每轮循环:
  agent 输出后  → H2 (Action Gate)
  环境执行后    → H4 (WorldModel + SubgoalSM + stall detection)
               → H5 (per-step guidance)
               → H6 (budget check)
```

### 泛化性设计原则

- **_SEARCH_PRIORITY** 基于家居常识排序，不使用训练集统计。
- **H5 step_guidance** 始终以"建议"语气注入，不强制（H6 硬干预除外）。
- 所有 observation/action 解析依赖 ALFWorld 环境 API，在 train/test split 之间完全一致。
- WorldModel 每个 episode 从零构建，无跨 episode 记忆。

---

## 1. Episode 开始前（初始化阶段）

### H0 触发：任务解析

**位置**：`ALFWorldHarnessRuntime.init_task(init_prompt, initial_admissible)`

在 episode 的 init_prompt 注入 session 后立即调用（一次性）：

- 从任务描述中解析 `TaskContext`：
  - `task_type`：6 种任务类型（见下表）
  - `target_type`：目标物品类型（如 `"kettle"`）
  - `destination_type`：目标容器类型（如 `"countertop"`）
  - `transformation`：所需变换（`"clean"` / `"heat"` / `"cool"` / `None`）
  - `count`：需要的物品数量（1 或 2）
  - `subgoal_chain`：根据任务类型确定的子目标序列
- 初始化 `WorldModel`，用初始 admissible 中的 `go to X` 命令填充 `unvisited` 列表
- **初始目标扫描**（P2-2）：若初始 admissible 中存在 `take {target_type}` 动作，立即设置 `world.target_found=True, target_location="starting location"`

**任务类型检测**支持两种 look_at_obj 表述：
- `"examine the X with the desklamp"` → `look_at_obj`
- `"look at X under the desklamp"` → `look_at_obj`（ALFWorld 的另一种任务格式）

**子目标链定义**：

| 任务类型 | 子目标序列 |
|---------|----------|
| pick_and_place | FIND → TAKE → GOTO_DEST → PUT |
| pick_clean_then_place | FIND → TAKE → GOTO_SINK → CLEAN → GOTO_DEST → PUT |
| pick_heat_then_place | FIND → TAKE → GOTO_MICROWAVE → HEAT → GOTO_DEST → PUT |
| pick_cool_then_place | FIND → TAKE → GOTO_FRIDGE → COOL → GOTO_DEST → PUT |
| look_at_obj | GOTO_LAMP → USE_LAMP → FIND → **TAKE** → EXAMINE |
| pick_two_obj | FIND → TAKE → GOTO_DEST → PUT → FIND → TAKE → GOTO_DEST → PUT |

> **look_at_obj 注意**：`examine X with desklamp` 仅在持有目标物品且灯已开启时才出现于 admissible。因此子目标链中 FIND 后必须经过 TAKE，再进行 EXAMINE。

**H0 不向对话注入任何内容**，只建立内部数据结构，供后续层消费。

---

### H3 触发：工具描述嵌入

在 `ALFWorld.__init__` 阶段（类初始化时执行一次）：

- 调用 `patch_take_action_tool_description(tools, hint)` 向 `take_action` 工具描述追加任务提示
- `build_h3_hint()` 根据任务类型返回对应策略 hint：

| 任务类型 | 嵌入描述 |
|---------|---------|
| pick_and_place | `"Prioritize: locate target → take → navigate to destination → put."` |
| pick_two_obj | `"You can carry one item at a time. Deliver first, then collect second."` |
| pick_clean/heat/cool_then_place | `"Locate target → take → go to [appliance] → transform → go to destination → put."` |
| look_at_obj | `"Go to desklamp → use it → find target object → take it → examine with desklamp."` |

---

### H5 冷启动触发：任务类型技能注入

`init_task` 执行后立即调用 `cold_start_skill_hints()`：

- **两层检索**：
  1. **分类过滤**：从 `ALF_SKILLS` 中筛选 `task_types` 包含当前任务类型的技能
  2. **BM25 排序**：对过滤后的候选按 `task_ctx.task_goal`（原始任务描述）打分，取前 `h5_top_k` 名
- 将检索到的技能文本注入 session（`"Harness skill hints:\n- ..."`），每条不超过 `h5_cold_start_max_words` 词

典型映射结果（top_k=1）：

| 任务类型 | BM25 top-1 技能 |
|---------|----------------|
| pick_and_place | `pickup_then_deliver` |
| pick_two_obj | `pickup_then_deliver` |
| pick_clean/heat/cool_then_place | `state_changing_actions_only` |
| look_at_obj | `examine_with_lamp` |

`break_loops_early` 的 `task_types=[]`，永远不会出现在冷启动结果中，仅由 H4 停滞恢复注入。

### 技能库（ALF_SKILLS）

`ALF_SKILLS` 定义在 `alfworld.py` 末尾，共 6 条：

| ID | task_types | 用途 |
|----|-----------|------|
| `pickup_then_deliver` | pick_and_place, pick_two_obj | 基础取放流程：定位 → 取 → 导航 → 放 |
| `state_changing_actions_only` | pick_clean/heat/cool_then_place | 变换流程：取 → 变换器 → 变换 → 目标 → 放 |
| `two_object_staging` | pick_two_obj | 提醒 agent 每次只能持一个物品，先交付再取第二个 |
| `examine_with_lamp` | look_at_obj | 先去灯台开灯，再找目标物品，**取起后**用灯检查 |
| `explore_unseen_first` | 除 look_at_obj 外所有类型 | 搜索时优先访问未探索的容器/位置 |
| `break_loops_early` | _（空，不用于冷启动）_ | 仅由 H4 硬/软停滞恢复注入，提示换到其他未探索位置 |

> `examine_with_lamp` 明确包含 TAKE 步骤（"take it"），与 look_at_obj 子目标链 `FIND → TAKE → EXAMINE` 一致。

---

## 2. 每轮：agent 输出后、环境执行前

### H2 触发：Action Gate（前置校验）

`pre_validate_action(raw_action, admissible)` 每轮执行：

1. **H4 强制动作优先**：若 H4 设置了 `force_next_action`，但 agent 当前动作是 task-critical 动词（`take/put/clean/heat/cool/use/examine`）且在 admissible 中，则放弃 force，优先执行 agent 的关键动作。
2. **精确匹配**：若 raw_action 在 admissible 中，直接通过。
3. **动词前缀限定匹配**：相似度匹配只在与 agent 动作同动词的候选中进行。若无同动词候选，返回 invalid，**不做跨动词替换**（防止 take↔put 混淆）。
4. **相似度阈值**：`action_similarity_threshold = 0.55`，仅接受高置信度的拼写/格式修正。
5. **连续无效阻断**：`invalid_block_after = 2`，连续 2 次 invalid 后阻断并注入提醒。

**空轮次检测**（P2-1，在 `post_step_monitor` 中执行）：
- 追踪 `empty_turn_count`（连续空 action 次数）
- 达到 `h2_empty_turn_threshold = 2` 后注入：`"Harness: you must call take_action... Current goal: {sg} ({tt})."`

**`never_block_prefixes`**：`go to / open / close` 开头的动作即使 invalid 也**不阻断**——这类动作是探索行为，阻断会锁死搜索；其代价（错误动作不执行）远小于误阻断。

**返回字段**：`action / canonicalized / blocked / reason / raw_action`

---

## 3. 每轮：环境执行后（拿到 observation 后）

### H4 触发：流程监控

`post_step_monitor(raw_output, final_action, observation, admissible)` 每轮执行，包含三个子系统：

#### H4-A：世界模型更新（WorldModel）

每步 observation 解析后更新：

| 字段 | 更新逻辑 |
|------|---------|
| `inventory` | `"you pick up"` → 设置持有物；`"you put ... in/on"` → 清空 |
| `visited` / `unvisited` | 从 `go to X` 动作维护；admissible 中新出现的 `go to X` 加入 unvisited |
| `object_at` | 从 observation 提取 `"word number"` 格式对象，记录到当前位置 |
| `target_found` / `target_location` | 一旦观测到目标物品类型，标记并记录位置 |
| `placed_count` | 每次成功 PUT 后递增 |
| `placed_locations` | 每次成功 PUT 后记录当前位置（P1-2 + P3：供 FIND hint 过滤，且目标发现时排除） |
| `lamp_location` | USE_LAMP 成功时记录当前位置（L1）；EXAMINE hint 用此引导 agent 回到灯台 |

**`find_put_action(dest_type, admissible)`** 只返回 dest_type 完全匹配的 put 动作，**无 fallback**（P0-1）。原 fallback（返回任意 put 动作）会导致 agent 将物品放入错误容器，随后在正确目的地产生"Nothing happens"死循环。

#### H4-B：子目标状态机推进（SubgoalSM）

每步根据 observation + admissible 推进子目标索引：

| 当前子目标 | 转移条件 |
|----------|---------|
| FIND | `find_take_action(target)` 在 admissible 中（所有任务类型统一，含 look_at_obj） |
| TAKE | `"you pick up"` 在 observation 中 |
| GOTO_SINK/MCW/FRG | 对应变换动作（clean/heat/cool）出现在 admissible |
| CLEAN/HEAT/COOL | `"you clean/heat/cool"` 在 observation 中 |
| GOTO_DEST | `find_put_action(dest)` 在 admissible 中，或 `"you put"` 在 observation 中（后者直接跳 2 步） |
| PUT | `"you put"` 在 observation 中 |
| GOTO_LAMP | `find_use_lamp_action()` 在 admissible 中 |
| USE_LAMP | `"you turn on"` 或 `"you switch on"` 在 observation 中 |

**PUT 后重置**（P0-2）：当 PUT 执行后（或 GOTO_DEST 跳过 PUT 时），若下一个子目标是 FIND（即 pick_two_obj 第二周期），自动重置 `world.target_found=False, world.target_location=None`，防止 agent 导航回已清空的取物位置。此逻辑对任何未来包含多次 PUT→FIND 循环的任务类型同样适用。

PUT 执行后自动设置 `post_put_grace`（默认 4 轮），宽限期内不触发停滞检测。

#### H4-C：停滞检测与干预

**标准停滞**（原有）：满足 `enough_rounds`（已运行 ≥ `h4_min_rounds_before_stall` 轮）AND `repeated_action`（最近 `h4_stall_window` 轮 action 完全相同）AND `no_new_objects`（observation 无新对象），宽限期或子目标 DONE 时跳过。

干预升级：首先执行最多 `h4_soft_intervention_rounds = 2` 次**软干预**（注入文本提示），之后切换为**硬干预**（设置 `force_next_action`，绕过 agent 决策）。

**"examine without lamp" 循环检测**（L1，新增）：
- 条件：当前子目标为 EXAMINE，持有物品，最近 ≥ 2 个 observation 含 `"nothing special"`
- 原因：`examine X`（不含 desklamp）给出 "There's nothing special" 且不完成任务；agent 不知道要回到灯台
- 干预：注入 `"Harness: 'examine {tt}' without the desklamp does nothing. Go to {lamp_location} — you must be AT the lit desklamp."` （若 `lamp_location` 已记录）
- `find_examine_action` 已修改为**只匹配含 "desklamp" 的动作**，防止 hint 指向错误的 plain examine

**"Nothing happens" 循环检测**（P1-1，原有）：
- 维护 `last_observations` 滑动窗口（`h4_nothing_happens_window = 4`）
- 条件：最近 N 个 observation 中 ≥ `h4_nothing_happens_threshold = 3` 个含 `"nothing happens"`，且 `world.inventory is None`
- 干预：注入 `"Harness: you are not holding anything — put actions have no effect. Find and pick up the {tt} before trying to place it."`
- 触发后清空观测窗口，避免重复触发

> 所有配置参数（窗口大小、阈值等）见第 0 节全量参数表。

---

## 4. 每轮：H4 之后

### H5 触发：每步目标导向指导

`step_guidance(current_round, max_step, admissible)` 每轮执行：

基于当前子目标 + 世界模型，生成精确的下一步建议（**始终为 Hint 语气，不强制**）：

| 当前子目标 | 条件 | 注入内容 |
|----------|------|---------|
| FIND（非 look_at_obj） | target 已知 | `"Hint: {tt} was spotted at {loc}. Navigate there and take it."` |
| FIND（非 look_at_obj） | target 未知 | `"Hint: not found yet. Suggested next: go to {priority_loc}."` （排除 `placed_locations`） |
| FIND（look_at_obj） | target 已知 | `"Hint: {tt} was spotted at {loc}. Navigate there and take it."` |
| FIND（look_at_obj） | target 未知 | `"Hint: searching for {tt}. Try: go to {priority_loc}."` |
| TAKE | take 动作在 admissible | `"Hint: pick up the {tt} — use: {take_action}."` |
| GOTO_SINK/MCW/FRG | 变换动作不在 admissible | `"Hint: go to {appliance} to {transform} the {item}."` |
| CLEAN/HEAT/COOL | 变换动作在 admissible | `"Hint: {transform} the {item} — use: {transform_action}."` |
| GOTO_DEST 或 PUT | put 动作在 admissible | `"Hint: place the {item} — use: {put_action}."` |
| GOTO_DEST 或 PUT | put 动作不在 admissible | `"Hint: go to {dt} to place the {item}."` |
| GOTO_LAMP | — | `"Hint: find the desklamp in this room and go to it."` |
| USE_LAMP | lamp 动作在 admissible | `"Hint: turn on the lamp — use: {lamp_action}."` |
| EXAMINE | `examine X with desklamp` 在 admissible | `"Hint: use exactly: {examine_action}."` |
| EXAMINE | 持有物品，lamp_location 已知 | `"Hint: go to {lamp_location} first — 'examine {tt} with desklamp' only works when you are at the lit lamp."` |
| EXAMINE | 持有物品，lamp_location 未知 | `"Hint: find the lit desklamp and go to it — then use 'examine {tt} with desklamp'."` |

**FIND hint 逻辑（优先级由高到低）**：
1. `world.target_found` 已定位 → 直接给出位置
2. `find_known_target_location(tt)`：扫描 `object_at` 中已观测到的同类目标，排除 `placed_locations` → 直接命中（**pick_two_obj 第二周期关键优化**）
3. `ordered_unvisited`：按 commonsense 排序的未探索位置，排除 `placed_locations`（P1-2）

**P3：placed_locations 排除目标发现**：`world.update()` 在发现目标类型时，若当前位置在 `placed_locations` 中则**不设置 `target_found`**。防止 pick_two_obj 第二周期中 agent 把自己放下的物品当作新目标再次拾取。

返回 `None` 时不注入（避免噪声）。去重条件：`hint 文本` 与 `_subgoal_idx` 同上一次注入完全一致时跳过，即**内容或子目标任意一个变化都会重新注入**——保证 agent 每次状态推进都能获得最新建议，同时避免同一状态连续重复注入。

---

### H6 触发：步数预算管理

`budget_check(remaining_steps, admissible)` 每轮执行：

- **软警告**（`remaining < warn_threshold` 且子目标为 FIND）：注入紧急探索提示，列出前 2 个优先未探索位置。`warn_threshold` 默认为 `h6_warn_threshold=7`，对 pick_two_obj 和 pick_cool_then_place 取 `max(7, h6_warn_threshold_multistep=12)`，因为这两类任务有额外步骤消耗。
- **硬强制**（`remaining < h6_force_threshold=4` 且子目标为 GOTO_DEST/PUT，且 agent 持有目标物品，且 put 动作在 admissible 中）：设置 `force_next_action = put_action`，下一轮 H2 执行。这是 harness 中**唯一的非探索类强制动作**，条件要求严格（世界模型确认 + admissible 确认）。

---

## 5. 结果沉淀（离线可分析）

每个 sample 的结果中携带：

- `result["harness_trace"]["h2"]`：每轮 Action Gate 记录（含 canonicalized / blocked / reason）
- `result["harness_trace"]["h3"]`：工具描述注入记录
- `result["harness_trace"]["h4"]`：每轮停滞检测记录（含 stall_score、intervention_level、audit_reason）
- `result["harness_trace"]["h5"]`：冷启动技能 + 每步指导记录（含 trigger 字段区分）
- `result["harness_trace"]["h6"]`：每轮预算检查记录（含 forced/hint 字段）

---

## 一次推理流程的触发顺序（简图）

```
1. init_task()                     → H0: 解析 TaskContext，初始化 WorldModel，扫描初始可取目标
2. build_h3_hint()                 → H3: 向工具描述注入任务类型 hint
3. cold_start_skill_hints()        → H5: 分类过滤 + BM25 排序，注入 top-k skill（由 h5_top_k 控制）
4. [每轮] pre_validate_action()    → H2: 动词安全 Action Gate；累计空轮次检测
5. [每轮] post_step_monitor()      → H4: 更新 WorldModel + SubgoalSM + 停滞检测（含 nothing-happens 检测）
6. [每轮] step_guidance()          → H5: 基于子目标 + 世界模型的精确 hint
7. [每轮] budget_check()           → H6: 步数预算警告 / 硬强制完成
```

---

## 已知限制与未来方向

| 问题 | 现状 | 可能方向 |
|------|------|---------|
| pick_cool 成功率偏低（~50%） | 大房间中柜子过多（15+），步数消耗在搜索 | H6 在 FIND 阶段更早提示优先位置 |
| H2 阈值误匹配 | agent 幻觉出 "from inventory" 等无效动词形式时，可能匹配错误目标 | 对特定幻觉模式（如包含 "from inventory" 的动作）增加阻断规则 |
| pick_and_place 中 agent 成功放置后继续执行 | ALFWorld 无"任务完成"信号给 agent，agent 不会主动停止 | 在 DONE 子目标后注入终止提示 |
