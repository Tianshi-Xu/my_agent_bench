# WebShop Harness 触发架构（按一次推理流程）

本文档描述当前仓库里 **已实现** 的 WebShop harness，在一次正常推理中会在什么阶段触发、触发哪些能力。

## 0. 开关层（是否启用）

任务配置入口在 `configs/tasks/webshop.yaml`（`webshop-env_train` 节）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `enabled` | bool | false | 总开关，false 时其余开关全部无效 |
| `h2` | bool | true | 前置校验（Action Gate） |
| `h3` | bool | true | 工具描述嵌入（购物策略 hint） |
| `h4` | bool | true | 购物监控（搜索死循环 / 价格超限 / 产品标题不符 / 产品页停滞） |
| `h5` | bool | true | 目标导向指导注入（属性核查表） |
| `h6` | bool | true | 步数预算管理（强制 Buy Now，v5 起强制前检查属性完整性） |

### 全量配置参数参考

`WebShopHarnessConfig` 的完整默认值（`src/server/harness/webshop.py`）：

| 参数 | 默认值 | 所属层 | 说明 |
|------|-------|-------|------|
| `h2_click_similarity_threshold` | 0.80 | H2 | 点击值模糊匹配阈值（比 ALFWorld 高，防止误匹配 ASIN） |
| `h2_repeat_click_block_after` | 2 | H2 | 连续 N 次相同点击后阻断 |
| `h4_search_loop_threshold` | 4 | H4 | back_to_search 累计次数触发搜索死循环干预 |
| `h4_duplicate_search_threshold` | 2 | H4 | 同一搜索词重复 N 次触发干预 |
| `h4_product_stall_turns` | 3 | H4 | 产品页连续 N 步无进展触发停滞干预 |
| `h5_hint_max_words` | 60 | H5 | 每步 hint 词数上限 |
| `h6_warn_threshold` | 5 | H6 | 剩余步数 < N 时注入紧急提示 |
| `h6_force_threshold` | 3 | H6 | 剩余步数 < N 且 buy now 在 clickables 时强制 |
| `price_tolerance` | 0.05 | H4/H5 | 价格容差 5%（略超预算仍接受） |

---

## 背景：为何要 harness

WebShop baseline（Qwen3-4B，200 episodes）表现：
- **平均奖励 0.517**，完整成功率仅 **12.5%**（reward=1.0）
- **76% 的 episode 有部分奖励**（0 < r < 1）：agent 确实完成了购买，但买了错误的属性（颜色/尺码/材质不对）
- 11.5% 零奖励：task limit 或根本没买

**最大问题不是"没有买"，而是"买错了"**。

根因分析：
1. 产品属性未完整选择就点 Buy Now（忘了选 size/color）
2. 产品不匹配需求（价格超限、规格不符）但 agent 仍然购买
3. 搜索死循环：不断 back-to-search + 新搜索，耗尽步数

---

## 层级设计总览

```
Episode 开始: H0（解析任务需求）
类初始化时:  H3（工具描述嵌入）
第 0 步:     H5（cold-start 搜索 hint）
每轮循环:
  agent 输出后  → H2（Action Gate：有效性校验 + 重复点击阻断 + buy-now属性预检 + IG OR逻辑）
  环境执行后    → H1（页面状态更新）
               → H4（购物监控：新ASIN品类验证→搜索循环→价格超限→页面停滞）
               → H5（目标导向 hint：属性核查表，IG OR组逻辑）
               → H6（预算管理：属性完整后才强制 Buy Now）
```

### 与 ALFWorld 的关键差异

| 方面 | ALFWorld | WebShop |
|-----|---------|---------|
| 子目标结构 | 固定子目标链（FIND→TAKE→...→PUT） | 无固定链，改用**购物阶段追踪器**（H1） |
| 主要失败原因 | 导航死循环、无效动作 | 属性选择遗漏、买错产品 |
| 核心 H5 输出 | 子目标对应的精确 admissible 动作 | 属性核查表（color/size 已选/未选） |
| H2 阈值 | 0.55（较宽松，防止探索阻断） | 0.80（较严格，防止误匹配 ASIN） |
| 新增层 | — | H1（页面状态追踪，WebShop 特有） |

### 泛化性设计原则

- 页面类型检测基于环境 API 信号（`has_search_bar`、observation 结构），在 train/test 完全一致。
- 属性核查表由 task instruction + 当前 page clickables 推断，**不使用训练集统计**。
- H6 强制 Buy Now 仅在 `"buy now"` 已在 clickables 时触发（admissible 守卫），不会产生非法动作。
- `_COLOR_TOKENS` / `_SIZE_TOKENS` 为常识词表，不依赖具体场景。
- 所有阈值为行为模式判断，不依赖具体商品名称。

---

## 1. Episode 开始前（初始化阶段）

### H0 触发：任务需求解析

**位置**：`WebShopHarnessRuntime.init_task(instruction)`，在 episode 的 `env.reset` 后、首轮 session.inject 前立即调用（一次性）。

从 task instruction 中提取结构化需求 `WebShopTaskRequirements`：

| 字段 | 提取策略 | 示例 |
|-----|---------|------|
| `item_keywords` | 去除价格/指令动词后的核心名词短语 | `"large cotton t-shirt"` |
| `color` | `_CORE_COLOR_TOKENS` 全词匹配；`_AMBIGUOUS_COLOR_TOKENS` 仅在 "X colored/color/shade" 上下文匹配 | `"pink"`, `"honey colored"→honey` |
| `color_alt` | OR 表达式的备选颜色（`"lavender or ivory"→color_alt=ivory`） | `"ivory"` |
| `size` | `_SIZE_TOKENS` 集合匹配，优先最长；支持词数字（`"size five"→5`）；含床/家具尺码（twin/full/queen/king） | `"xx-large"`, `"5"` |
| `material` | 预设材质列表匹配 | `"cotton"` |
| `price_max` | 正则匹配 `lower than/less than/under/< $X` | `30.0` |
| `quantity` | 匹配 `1 dozen / pack of N / Nx` 等 | `"1 dozen"` |
| `style_keywords` | 预设修饰词模式列表 | `["wireless", "waterproof"]` |
| `specs` | `_MEASUREMENT_RE` 匹配测量/规格值，支持连字符（`"13-ounce"→"13 ounce"`） | `["8 ounce", "32gb"]` |
| `task_keywords` | 任务描述中的有效词（4字符以上，过滤停用词），用于 H5 instruction-grounded 选项匹配 | `["huckleberry", "counter", "height"]` |

**颜色提取防误判**：`_AMBIGUOUS_COLOR_TOKENS`（honey/chocolate/natural/camel 等）仅在明确带有颜色语境时提取，避免"chocolate extract"→color=chocolate、"natural hair"→color=natural 此类误判。

**H0 不向对话注入任何内容**，只建立内部结构供后续层消费。

---

### H3 触发：工具描述嵌入

**位置**：`WebShop.__init__` 类初始化阶段（一次性），调用 `patch_webshop_tool_descriptions(tools)`。

向两个工具的 description 追加购物策略 hint：

| 工具 | 追加内容 |
|-----|---------|
| `search_action` | `"Search for the product using specific keywords matching all required attributes (item type, color, size, material, etc.). Include price constraint words if helpful."` |
| `click_action` | `"Click a product from results, select ALL required attributes (color, size, etc.) that match the task description, then click 'buy now'."` |

静态嵌入，无动态内容，对 train/test 完全一致。原始工具列表不被修改（deep copy）。

---

## 2. 第 0 步：H5 cold-start 搜索 hint

`step_guidance(step_num=0, ...)` 在首轮 observation 注入后调用：

- 页面为 HOME，step_num == 0 时生成初始搜索建议
- 建议格式：`"Hint: search for '{item_keywords} {color} {size}' to find a matching product."`
- 若无 color/size 要求则只用 item_keywords

---

## 3. 每轮：agent 输出后、环境执行前

### 无工具调用处理（task.py 层）

**位置**：`WebShop.sync_start_sample()` 循环内

当 agent 输出中没有工具调用时：
- 记录 `_no_tool_consecutive` 计数器
- 返回错误提示：`"No executable tool calls found! You should call a tool instead."`
- **若连续 ≥ 2 轮无工具调用**，追加当前页面对应的 H5 `step_guidance` hint
- **若连续 ≥ 3 轮无工具调用**且 `back to search` 在 clickables 中，强制执行 `click[back to search]`

---

### H2 触发：Action Gate（前置校验）

`pre_validate_action(tool_name, raw_value, has_search_bar, clickables)` 每轮执行：

1. **强制动作优先**：若 `force_next_action` 已设置（H5 强制或 H6 强制），且 agent 当前不是点击 "buy now"，则强制执行 force_next_action，忽略 agent 的动作。

2. **Search 合法性**：`search_action` 且 `has_search_bar=False` → 阻断，注入 `"Search is not available on this page. Click one of the available options."`

3. **Click 精确/模糊匹配**：
   - 精确匹配（大小写不敏感）→ 放行
   - 模糊匹配（SequenceMatcher ≥ 0.80）→ 放行并标记 `canonicalized=True`
   - 无匹配 → 阻断

4. **Buy Now 属性预检**（`_buy_now_precheck`）：agent 点击 buy now 时，检查以下条件，若有任意未满足则阻断：
   - color（含 OR 备选）可选但未选
   - size 可选但未选
   - spec（规格值）可选但未选
   - **instruction-grounded 选项**：通过 `task_keywords` 匹配到的 `other` 类选项可选但未选

5. **重复点击阻断**：连续 N 次相同 click（`h2_repeat_click_block_after=2`）→ 阻断；**永不阻断** `back to search / next > / < prev / buy now`（导航/购买关键路径）。

**返回字段**：`action / blocked / reason / canonicalized / block_message`

---

## 4. 每轮：环境执行后（拿到 observation 后）

### H1 触发：页面状态更新

`update_state(action, observation, has_search_bar, clickables)` 每步执行。

**页面类型检测**（`detect_page_type`）：

| 页面类型 | 检测信号 |
|---------|---------|
| `HOME` | `has_search_bar=True` |
| `SEARCH_RESULTS` | `"Total results:"` 在 observation 中 |
| `PRODUCT_DETAIL` | `"buy now"` 在 clickables 中，或含 Description+Features+Reviews |
| `UNKNOWN` | 以上均不满足 |

**追踪状态**（`WebShopPageState`）：

| 字段 | 说明 |
|-----|------|
| `page_type` | 当前页面类型 |
| `search_queries` | 本 episode 所有搜索词列表 |
| `back_to_search_count` | 点击 "back to search" 累计次数 |
| `current_asin` | 当前产品页 ASIN（从 `click[ASIN]` action 字符串提取，非 clickables） |
| `asins_visited` | 已访问产品列表 |
| `selected_attributes` | 当前产品页已选属性（`{"color": "pink", "size": "large"}`） |
| `attribute_options` | 当前产品页可选属性（从 clickables 解析，分 color/size/spec/other 四类） |
| `current_price` | 当前产品页价格（从 `"Price: $X.XX"` 解析） |
| `_stall_turns` | 同一产品页连续停留步数 |

**属性选择追踪（关键修复）**：
- `attribute_options` 分类使用复合匹配：`"sky blue"` / `"light brown"` / `"turquoise | ivory"` 等也能被分类为 color，`"4x-large"` 也能被分类为 size
- 使用 **pre-click 选项快照**进行归属判断：WebShop 在选中属性后往往把该选项从 clickables 移除，若先刷新再查找会导致漏判；现在先保存快照，再刷新，再用快照查找
- 产品页切换时（ASIN 变化）重置 `selected_attributes` 和 `attribute_options`

---

### H4 触发：购物监控

`post_step_monitor(action, observation, has_search_bar, clickables)` 每步执行。

检测按以下优先顺序执行，**首个触发的检测返回，不继续向下**：

**① 搜索死循环检测**（`search_loop`）：
- 条件：`back_to_search_count >= 4` 或同一搜索词重复 ≥ 2 次
- 干预（首次）：`"Harness: you have searched many times without success. Try a simpler query, e.g. '{simplified}', or pick any close match and buy it."` — simplified query 使用 `task_keywords[:5]`（**v6：比 item_keywords 更干净，过滤停用词和动词**）
- 干预（第 2+ 次循环重置，**v6 新**）：`"Harness: still searching after many attempts. Stop searching — pick the best matching product from the current results and click buy now."` — 强制 agent 放弃搜索
- `_search_loop_resets` 计数器跨循环累积，episode 重置时归零
- 触发后**同时重置** `back_to_search_count` 和 `search_queries`（保留最后一条）

**② 价格超限警告**（`price_over_budget`）：
- 条件：在 PRODUCT_DETAIL 页，`current_price > price_max × 1.05`
- 干预：`"Harness: this product costs $X but your budget is $Y. Go back to search and find a cheaper option."`

**③ 产品页停滞**（`product_stall`）：
- 条件：在 PRODUCT_DETAIL 页连续 ≥ `h4_product_stall_turns=3` 步
- 干预：直接注入 H5 属性核查表（见 H5 说明）
- 触发后重置 `_stall_turns`，避免连续重复触发

---

### H5 触发：每步目标导向指导

`step_guidance(step_num, max_steps, observation, has_search_bar, clickables)` 每步执行。

按页面类型生成 hint：

**HOME 页**（step_num == 0）：
```
Hint: search for '{item_keywords} {color} {size}' to find a matching product.
```

**SEARCH_RESULTS 页**：

| 情况 | hint 内容 |
|-----|----------|
| `back_to_search_count >= 2` | `"you've searched several times. Try different keywords. Requirements: {summary}."` |
| 有价格限制 | `"check prices before clicking (budget: $X). Look for: {summary}."` |
| 其他 | `"select a product matching: {summary}."` |

**PRODUCT_DETAIL 页**（核心）：

生成**属性核查表**（`_build_attribute_checklist`）：

```
Attribute checklist:
  - color=pink: [selected ✓]           # 已选
  - color=pink: [click 'pink']         # 可选但未选 → H2 会阻断 buy now
  - color=pink: [not available — page has: black, white]  # 本产品无此颜色
  - size=large: [selected ✓]
  - spec=13 ounce: [click '13 ounce (pack of 1)']   # 规格值匹配
  - feature: [click 'huckleberry']     # instruction-grounded 选项
  - price=$22.09 (budget: $30): [✓]
→ All checked. Click 'buy now' to complete the purchase.
→ Still need to select: color, huckleberry.
→ Price over budget. Go back to search for a cheaper product.
→ This product is missing required color options. Go back to search for a better match.
→ Verify this product matches the task description, then click 'buy now'.  # 无追踪属性时
```

**Instruction-grounded 选项匹配**（新功能）：
- `_instruction_grounded_matches(req, other_opts)`：用 `task_keywords` 词集与 `other` 类 clickable 做词汇重叠检测
- 匹配示例：`"huckleberry syrup"` 任务 → 页面选项 `"huckleberry"` 匹配；`"counter height barstools"` → 选项 `"counter height"` 匹配；`"vegan beard balm"` → 选项 `"vegan cedarwood beard balm"` 匹配
- 最多显示 2 个 IG 匹配项，H2 中最多阻断 1 个 IG 未选项

**verdict 规则**（`buy now` 在 clickables 时，优先级从高到低）：
- 价格超限 → "Price over budget. Go back to search"
- 有 `missing` 状态属性（页面上可点击但未选） → "Still need to select: X"（优先处理可点击的属性）
- 有 `unavailable` 状态属性（页面无此选项卡） → "No color=X selector (may be a single-variant product). If the title confirms it has X, click 'buy now'."（**v6 新逻辑：不再强制 go back，允许 agent 自主判断是否单品变体**）
- 全部 `ok` → "All checked. Click 'buy now'"
- 无追踪属性（`attr_statuses` 为空） → "Verify this product matches the task — look for 'kw1', 'kw2' in the title/description"（**v6：展示 task_keywords 辅助验证**）

**去重机制**：hint 内容 + 当前页面类型同上一次注入完全一致时跳过。
**词数上限**：超过 `h5_hint_max_words=60` 词时截断。

---

### H6 触发：步数预算管理

`budget_check(remaining_steps, clickables)` 每步执行：

| 条件 | 干预类型 | 内容 |
|-----|---------|------|
| `remaining < h6_force_threshold=3` 且 `"buy now"` 在 clickables | **硬强制** | 设置 `force_next_action = "click[buy now]"`，下一轮 H2 强制执行 |
| `remaining < h6_warn_threshold=5` 且在 SEARCH_RESULTS/HOME 页 | **软警告** | `"[N steps left] Urgently: click a product and buy it now."` |
| `remaining < h6_warn_threshold=5` 且在 PRODUCT_DETAIL 且 buy now 可用 | **软警告** | `"[N steps left] Buy now is available — click 'buy now'."` |

---

## 5. 关键数据结构

```python
WebShopTaskRequirements:
    item_keywords: str        # 核心商品名词
    color: Optional[str]      # 颜色要求（核心/歧义颜色词，上下文感知提取）
    color_alt: Optional[str]  # OR 备选颜色（"lavender or ivory" → color_alt="ivory"）
    size: Optional[str]       # 尺码要求（含 twin/full/queen/king，词数字 five→5）
    material: Optional[str]   # 材质要求
    price_max: Optional[float]  # 价格上限
    quantity: Optional[str]   # 数量要求
    style_keywords: List[str] # 预设修饰词（wireless/waterproof 等）
    specs: List[str]          # 测量/规格值（"8 ounce", "32gb", 归一化连字符）
    task_keywords: List[str]  # 任务描述有效词（供 IG 选项匹配使用）

WebShopPageState:
    page_type: str            # HOME / SEARCH_RESULTS / PRODUCT_DETAIL / UNKNOWN
    search_queries: List[str] # 本 episode 搜索历史
    back_to_search_count: int # 返回搜索页累计次数
    current_asin: Optional[str]   # 当前产品 ASIN（从 action 字符串提取）
    selected_attributes: Dict[str, str]    # 已选属性，使用 pre-click 快照追踪
    attribute_options: Dict[str, List[str]] # 可选属性（color/size/spec/other，支持复合选项）
    current_price: Optional[float] # 当前价格
```

---

## 6. 一次推理流程的触发顺序（简图）

```
1. __init__ 时: patch_webshop_tool_descriptions()  → H3: 工具描述嵌入
2. env.reset + init_task()                         → H0: 解析购物需求（含 task_keywords）
3. 第 0 步 step_guidance()                         → H5: 搜索 hint（cold-start）
4. [每轮] (无工具调用检测)                         → task.py: 连续 ≥2 轮无工具调用时附加 H5 hint
5. [每轮] pre_validate_action()                    → H2: 搜索合法性 + 点击模糊匹配 + 重复阻断 + buy-now 属性预检（含 IG）
6. [每轮] env.step()后 update_state()              → H1: 页面类型 + 属性选择（pre-click 快照 + 复合选项分类）
7. [每轮] post_step_monitor()                      → H4: 搜索死循环（双重重置）/ 价格超限 / 产品页停滞
8. [每轮] step_guidance()                          → H5: 属性核查表（含 IG 匹配 + 修正 verdict 规则）
9. [每轮] budget_check()                           → H6: 步数紧急警告 / 强制 Buy Now
```

---

## 7. 迭代历史与指标进展

| 版本 | 主要改动 | success@1 | avg reward |
|------|---------|-----------|------------|
| Baseline | 无 harness | 12.5% | 0.517 |
| v1 | H0–H6 基础实现 | 16.5% | 0.532 |
| v2 | Bug fix × 4（ASIN提取/regex/size-spec歧义/sep清洗） + 复合选项匹配 + half-size | 27.5% | 0.633 |
| v3 | 词数字(size five→5) / OR颜色 / 歧义颜色防误判 / spec归一化 / All-checked FP修复 / bed sizes | 28.0% | 0.626 |
| v4 | 复合颜色分类(sky blue/light brown) / H1 pre-click快照 / H4搜索循环双重重置 / IG选项匹配 | 28.5% | 0.612 |
| v5 | **IG OR组逻辑**（消除互斥死循环）/ **H0句号小数bug** / **多词尺码归一化**（extra large→x-large）/ **H4品类核查**（新ASIN第1步检测产品标题） / 颜色词扩充 / H6 force前属性预检 / 乘法spec展开 | 28.5% | 0.600 |
| v6（本次）| **A1 unavailable verdict 软化**（单品变体不再强制go-back）/ **verdict顺序修正**（missing优先于unavailable）/ **C1 verify hint展示task_keywords** / **D1 搜索建议用task_keywords**（过滤动词/停用词）/ **D2 循环逃逸升级**（第2次loop reset强制just-buy）/ **B1 H7误报修复**（扩充_H7_GENERIC_WORDS动词/形容词）/ **停用词扩充**（"interested/buying/wanting"等） | 待测 | 待测 |

---

## 8. 文件位置

| 文件 | 说明 |
|-----|------|
| `src/server/harness/webshop.py` | 完整 harness 实现（H0~H6 + 辅助函数） |
| `src/server/harness/__init__.py` | 导出 WebShop harness 类 |
| `src/server/tasks/webshop/task.py` | WebShop task，含所有 harness 钩子 |
| `configs/tasks/webshop.yaml` | 任务配置，含 `enabled/h2~h6` 开关 |

---

## 9. 容器映射与热更新

`webshop-env_train` 容器（`extra/docker-compose.yml`）已挂载以下卷：

```yaml
volumes:
  - ../configs:/app/configs
  - ../src/server/tasks/webshop:/app/src/server/tasks/webshop
  - ../src/server/harness:/app/src/server/harness
```

修改上述任一文件后，**重启 `webshop-env_train` 容器**即可生效，无需重新 build：

```bash
docker compose -f extra/docker-compose.yml restart webshop-env_train
```

---

## 10. 已知限制与未来方向

| 问题 | 现状 | 可能方向 |
|------|------|---------|
| IG 匹配精度 | 词汇重叠可能匹配到语义不相关的选项（如所有含 "height" 的选项） | 加入 TF-IDF 权重或 n-gram 匹配 |
| 搜索词优化有限 | H4 搜索死循环干预只提供简化建议词 | 基于需求子集动态生成多个候选查询 |
| 属性权重无感知 | WebShop 奖励按属性权重计算，harness 对所有属性一视同仁 | 引导 agent 优先完成高权重属性的选择 |
| 多页搜索结果未利用 | Agent 通常只看第 1 页结果 | H5 在 SEARCH_RESULTS 页提示浏览 next > |
