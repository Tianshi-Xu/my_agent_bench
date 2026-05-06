# ALFWorld / WebShop Baseline 轨迹配置说明

本文档用于指导模型或推理框架按当前仓库的 baseline 逻辑正确运行 ALFWorld 和 WebShop。这里的 baseline 指不启用 harness 的原始任务流程；如果配置文件中 `enabled: true`，会触发额外的 harness 逻辑，不属于本文档范围。

## 总体协议

两类任务都使用 OpenAI Chat Completions 风格的消息与工具调用协议：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."}
  ],
  "tools": [...]
}
```

`tools` 是请求体顶层字段，和 `messages` 并列，不是某一条 `system` 或 `user` message。OpenAI 服务端会在内部把工具 schema 序列化进模型上下文，通常等价于放在 system 级上下文附近、早于用户轨迹内容。代码侧不应把工具说明手写进普通消息，也不应把 `tools` 放入保存的轨迹消息列表。

每轮调用模型时都应传入当前完整 `messages` 和同一份 `tools`。模型输出应是 assistant message，可以包含 `tool_calls`。环境执行后，反馈必须作为 `role="tool"` 的 message 注入，并带回对应的 `tool_call_id`。

## 不处理 reasoning token

baseline 不对 reasoning token 做特殊处理：

- 不读取或保存 `completion_tokens_details.reasoning_tokens`。
- 不从 assistant content 中剥离 `<think>` 或隐藏推理文本。
- token 统计只累计 `prompt_tokens`、`completion_tokens`、`total_tokens`。

如果目标模型会输出显式推理文本，仍需要同时产生合法 `tool_calls`。任务代码只执行 tool call；纯文本回答会被视为无可执行动作。

## ALFWorld

### 工具定义

ALFWorld 只注册一个工具：

```json
{
  "type": "function",
  "function": {
    "name": "take_action",
    "description": "Take an action.",
    "parameters": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "description": "The action you would like to take"
        }
      },
      "required": ["action"],
      "additionalProperties": false
    }
  }
}
```

工具 schema 来源是 `configs/tasks/alfworld.yaml`。baseline 中不要修改工具 description，也不要追加 task-specific hint。

### System prompt

system prompt 由 `ALFWorld.get_task_instruction()` 生成，核心要求如下：

- agent 在 household 环境中完成任务。
- 必须使用提供的工具提交动作。
- 动作写入 `take_action` 的 `action` 参数。
- 每轮会给出可选动作列表。
- 动作必须来自 available actions。
- 环境反馈 `Nothing happened` 表示上一步无效，需要尝试其它动作。

### 初始 user prompt

episode reset 后，任务会从 ALFWorld observation 中去掉开头的 metadata 部分，然后构造初始用户消息：

```text
Here is your task. {initial_observation} AVAILABLE ACTIONS: {admissible_action_1}
{admissible_action_2}
...
```

`AVAILABLE ACTIONS:` 后面是当前环境给出的 admissible commands。模型的下一步必须调用：

```json
{
  "name": "take_action",
  "arguments": {"action": "one admissible action"}
}
```

### 环境反馈 message

每次执行动作后，环境反馈作为 tool message 返回：

```json
{
  "role": "tool",
  "tool_call_id": "<assistant_tool_call_id>",
  "content": "{observation} AVAILABLE ACTIONS: {new_admissible_action_1}\n{new_admissible_action_2}\n..."
}
```

这条消息必须使用上一步 assistant tool call 的 `tool_call_id`。不要把环境反馈改成 assistant message。

### 无 tool call 时

如果模型没有输出 tool call，baseline 会注入一条 user message：

```text
You MUST call the take_action tool — do NOT output plain text without a tool call.
```

并给本轮 reward 0，然后继续下一轮。

### 动作解析

baseline 只取第一条 tool call，并读取 arguments 中第一个值作为动作文本。之后会：

- `strip()`、转小写、只保留第一行。
- 如果动作精确出现在 admissible commands 中，直接执行。
- 否则用 BLEU 和 admissible commands 做近似匹配，超过很低阈值时映射到最相近 admissible action。
- 仍无法匹配时按原动作交给环境。

模型配置时最好强制输出 exactly one tool call，且 `action` 精确复制 `AVAILABLE ACTIONS` 中的一项。

### Few-shot prompt

`src/server/tasks/alfworld/prompts/*.json` 会被加载，但 baseline 主循环没有把这些 examples 注入会话。配置其它模型时不要默认添加这些 few-shot 示例，除非明确要偏离当前 baseline。

## WebShop

### 工具定义

WebShop 注册两个工具：

```json
{
  "type": "function",
  "function": {
    "name": "search_action",
    "description": "Use search functionality with specified keywords.",
    "parameters": {
      "type": "object",
      "properties": {
        "keywords": {
          "type": "string",
          "description": "The keywords to use in the search function."
        }
      },
      "required": ["keywords"],
      "additionalProperties": false
    }
  }
}
```

```json
{
  "type": "function",
  "function": {
    "name": "click_action",
    "description": "Click a button or link with a specified value.",
    "parameters": {
      "type": "object",
      "properties": {
        "value": {
          "type": "string",
          "description": "The value to click from the list of available actions."
        }
      },
      "required": ["value"],
      "additionalProperties": false
    }
  }
}
```

工具 schema 来源是 `configs/tasks/webshop.yaml`。

### System prompt

system prompt 是 `prompt_with_max_turn`，核心规则如下：

- 你是 web shopping agent，需要根据任务 instruction 找到并购买正确商品。
- 每轮必须调用 `search_action` 或 `click_action`，不能只输出文本。
- 搜索结果页要仔细匹配商品标题和任务关键条件。
- 商品页要选择所有必要属性，然后点击 `buy now`。
- `click_action.value` 必须精确等于 available clickables 中的一个值。

### 初始 user prompt

第 0 轮注入：

```text
The initial observation:
{observation}

Available Actions:
{available_actions}
```

`available_actions` 通常包含 `has_search_bar` 和 `clickables` 等环境结构。

### 后续环境反馈

如果上一轮执行了有效 action，反馈作为 tool message：

```json
{
  "role": "tool",
  "tool_call_id": "<assistant_tool_call_id>",
  "content": "Action: {action}\n\nObservation:\n{observation}\n\nAvailable Actions:\n{available_actions}"
}
```

其中 action 是环境内部格式：

- `search_action({"keywords": "..."})` 转成 `search[...]`
- `click_action({"value": "..."})` 转成 `click[...]`

如果上一轮没有 action，则下一轮会以 user message 形式给出：

```text
Observation:
{observation}

Available Actions:
{available_actions}
```

### 无 tool call 时

baseline 会把 observation 设置为：

```text
You must call a tool! NEVER respond with only text.
```

本轮不执行环境动作，reward 为 0，并继续下一轮。

### 动作解析

baseline 只取第一条 tool call：

- `search_action` 的第一个参数值转为 `search[value]`。
- `click_action` 的第一个参数值转为 `click[value]`。

baseline 不做严格的 clickables 预校验，非法点击会交给 WebShop 环境处理。因此模型侧应保证：

- 在 search bar 可用时用 `search_action`。
- 点击时 `value` 精确复制 `Available Actions` 中的 clickable 字符串，例如 `buy now`、颜色、尺码、商品 ASIN 或 `back to search`。

## 推荐模型侧配置

为了复现 baseline 轨迹，模型调用层应满足：

- 使用 Chat Completions 兼容接口。
- 每轮请求都发送完整 `messages` 和顶层 `tools`。
- 不把 tools schema 转写成普通 system/user prompt。
- 允许 assistant 返回标准 `tool_calls` 字段。
- 每轮最多执行第一条 tool call；最好通过模型配置或 prompt 约束只生成一条。
- 不使用 text-only agent 模式；如果只能输出纯文本，需要额外适配为 OpenAI tool call 结构。
- 不启用 harness：任务参数中 `enabled: false`，并关闭所有 h2/h3/h4/h5 干预。

## 最小轨迹模板

ALFWorld:

```text
system: ALFWorld task instruction
user: Here is your task. ... AVAILABLE ACTIONS: ...
assistant: tool_calls=[take_action({"action": "go to ..."})]
tool: observation + AVAILABLE ACTIONS
assistant: tool_calls=[take_action({"action": "..."})]
tool: observation + AVAILABLE ACTIONS
...
```

WebShop:

```text
system: WebShop shopping rules
user: The initial observation: ... Available Actions: ...
assistant: tool_calls=[search_action({"keywords": "..."})]
tool: Action: search[...] + Observation + Available Actions
assistant: tool_calls=[click_action({"value": "..."})]
tool: Action: click[...] + Observation + Available Actions
...
```

