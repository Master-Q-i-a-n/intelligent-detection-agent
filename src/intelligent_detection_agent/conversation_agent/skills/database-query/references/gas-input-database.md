# 业务输入数据库

物理库：`database/gas_ai_input.duckdb`，工具：`query_business_data`。

## 用气与用户

- `telemetry.scada_observation`
  - 粒度：用户、管路、采集时刻。
  - 时间：`observed_at`，业务日期：`data_date`。
  - 企业：`user_id`、`entity_name`；管路：`pipeline_no`。
  - 压力/温度：`pressure`、`temperature`。
  - 工况瞬时/累计：`operational_instant`、`operational_cumulative`。
  - 标况瞬时/累计：`standard_instant`、`standard_cumulative`。
- `asset.user_meter`
  - 主键：`user_id`。
  - 企业/站点：`station_name`、`station_type`、`address`。
  - 表具：`meter_brand`、`meter_type`、`meter_model`、`quantity_min`、`quantity_max`。

## 检定与维修

- `inspection.meter_check_record`：一条检定记录，主键 `check_record_id`，通过 `user_id` 关联用户。
- `inspection.meter_check_point`：检定点，通过 `check_record_id` 关联检定记录；关键字段 `check_flow`、`indication_error`、`repeatability`。
- `inspection.meter_repair`：维修记录，通过 `user_id` 关联；故障和维修说明分别为 `fault_description`、`repair_description`。

## 设备振动

- `vibration.daily_health`：用户每天一个或多个设备窗口的日级标签与健康指标。
- `vibration.acceleration_window`：含三轴数组 `accel_x/y/z`，只有确实需要波形统计时才查询，避免返回完整数组。
- `equipment.vibration_sensor`：传感器档案，通过 `user_id`、`sensor_id` 关联。
- `vibration.health_label_dictionary`：H0-H4 阶段字典。
- `vibration.trajectory_dictionary`：健康趋势类型字典。

查询时间范围前使用 `MIN(data_date)`、`MAX(data_date)`；不要将数据库最大日期解释为“今天”。
