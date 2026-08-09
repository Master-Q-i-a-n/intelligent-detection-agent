# 安防数据库

物理库：`safety_operations/data/security.db`，工具：`query_security_data`。

## 核心表

- `security_events`：一条最终安防事件。
  - 主键 `event_id`；来源 `source_system`；摄像头/区域 `camera_id`、`zone_id`。
  - 类型 `event_type`；发生时间 `occurred_at`；严重度 `severity`。
  - 最终复核 `final_decision`、`final_reason`、`recommended_action`。
  - 人工处置状态 `handling_status` 只能查询，不能由对话修改。
- `llm_reviews`：视频复核尝试，通过 `event_id` 关联；可查询 `decision`、`helmet_status`、`evidence_quality`、`visual_reason`、`explanation`、`reviewed_at`。
- `event_people`：事件中的人员、跟踪编号及安全帽/工装状态。
- `event_transitions`：规则状态随帧变化的审计记录。
- `alert_records`：通知发件箱及发送状态，通过 `event_id` 关联。
- `event_handling_actions`：人工确认、处理、关闭的操作审计，只读。
- `cameras`、`zones`：摄像头和区域字典。

## 限制

- 必须明确列名，禁止 `SELECT *`。
- 文件路径、视频源、完整提示词、模型原始输入输出和通知原始载荷均不可查询。
- 对话中不存在安防确认、处理或关闭工具。用户要求这些操作时，应指引其前往“安全作业”页面。
