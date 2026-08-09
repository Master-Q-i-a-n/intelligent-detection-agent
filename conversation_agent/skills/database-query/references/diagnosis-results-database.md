# 诊断结果数据库

物理库：`database/gas_ai_results.duckdb`，工具：`query_diagnosis_data`。

## 智能计量

- `metering.diagnosis_run`
  - 粒度：用户、诊断日期、诊断运行。
  - 主键：`run_id`；关联字段：`user_id`、`diagnosis_date`。
  - 实测/推导：`observed_volume`、`predicted_normal_volume`、`baseline_missing_volume`、`meter_bias_volume`、`makeup_volume`。
  - 结论：`quality_status`、`risk_score`、`risk_level`、`meter_spec_result`、`summary`。
  - `makeup_volume` 是算法估算补量，不是结算确认量。
- `metering.anomaly_interval`
  - 通过 `run_id` 关联诊断运行。
  - 区间：`start_time`、`end_time`；类型：`anomaly_type`；估算：`estimated_missing_volume`。
- `metering.work_order`
  - 旧计量算法自动建议工单，仅属于计量模块。

## 智能设备

- `equipment.health_diagnosis`
  - 粒度：用户、表具、诊断日期。
  - `predicted_stage`/`predicted_stage_name`：H0-H4 状态。
  - `predicted_health_index`：0-100 健康指数。
  - `confidence`、`risk_level`、`trend_label` 是算法结果。
- `equipment.health_trend`
  - 时间范围：`start_date`、`end_date`。
  - 趋势：`daily_slope`、`maximum_daily_drop`、`stage_sequence`。

## 对话工单

- `operations.work_order`：人工批准后由对话 Agent 创建，适用于计量、设备或安防。
- `operations.work_order_audit`：创建审计。
- 对话查询可以读取工单，但只有 `create_work_order` 工具可以写入。
