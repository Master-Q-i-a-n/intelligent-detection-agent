# 表关联与粒度

## 用户主线

- 用户键统一按字符串处理：`user_id`。
- `telemetry.scada_observation.user_id` → `asset.user_meter.user_id`。
- `inspection.meter_check_record.user_id`、`inspection.meter_repair.user_id` → 用户。
- `vibration.daily_health.user_id`、`equipment.health_diagnosis.user_id` → 用户。
- 跨物理数据库不能在一次 SQL 中关联；分别查询后在回答或报告中按 `user_id` 和日期解释。

## 计量诊断

- `metering.diagnosis_run.run_id` → `metering.anomaly_interval.run_id`。
- 日期连接使用 `diagnosis_date`；异常区间使用时间戳。
- 同一用户同日可能有历史运行，查询最新结果时按 `created_at DESC` 排序或明确运行编号。

## 设备诊断

- 日级原始标签：`vibration.daily_health(user_id, data_date, meter_id)`。
- 模型结果：`equipment.health_diagnosis(user_id, diagnosis_date, meter_id)`。
- 趋势是时间段结果，不应当作某一个瞬时时刻的实测值。

## 安防

- `security_events.event_id` 是事件主键。
- `llm_reviews`、`event_people`、`event_transitions`、`alert_records`、`event_handling_actions` 通过 `event_id` 关联。
- 同一事件可能有多次 LLM 复核和多次人工操作；需要最新一次时必须按时间倒序并限制一条。
