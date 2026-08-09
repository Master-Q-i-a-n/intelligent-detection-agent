# 指标口径

## 日标况用气量

`standard_instant` 的单位是 m³/h。原始采集可能接近一分钟，必须先按用户、管路和五分钟时间桶求平均，再积分：

`五分钟气量 = max(五分钟平均标况瞬时流量, 0) × 5 / 60`

日用气量等于全部管路、全部五分钟桶气量之和。不能直接对原始行执行 `SUM(standard_instant) * 5 / 60`。

DuckDB 推荐使用：

```sql
time_bucket(INTERVAL '5 minutes', observed_at)
```

每个管路先 `AVG(standard_instant)`，然后 `SUM(GREATEST(flow_5m, 0) * 5.0 / 60.0)`。

## 完整度与有效数据

- 每天理论上有 288 个五分钟桶。
- 管路完整度 = 五分钟桶中 `standard_instant` 非空的桶数 / 288。
- 当前计量算法口径：至少一条管路完整度达到 50%，当天用户数据才标记为有效（`quality_status=0`）；否则不可诊断（`quality_status=2`）。
- 排名查询默认展示有效用户，并同时说明被排除的无效用户数量，不能静默隐藏数据质量问题。

## 数值属性

- `standard_instant`、压力、温度、振动数组：实测采集值。
- `observed_volume`：由五分钟标况瞬时流量积分得到的推导值。
- `predicted_normal_volume`：历史基线预测值。
- `baseline_missing_volume`、`meter_bias_volume`、`makeup_volume`：算法估算值，不能表述为已确认补量或结算量。
- 健康指数、阶段、风险、置信度：设备模型输出，不是人工结论。
