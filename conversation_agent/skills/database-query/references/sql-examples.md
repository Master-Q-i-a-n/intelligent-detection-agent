# 只读 SQL 示例

示例用于理解口径，不可无条件照搬用户、日期和阈值。

## 某日超过阈值的有效用户

```sql
WITH five_minute AS (
  SELECT user_id, pipeline_no,
         time_bucket(INTERVAL '5 minutes', observed_at) AS bucket,
         AVG(CASE WHEN standard_instant >= 0 THEN standard_instant END) AS flow_5m
  FROM telemetry.scada_observation
  WHERE data_date = DATE '2025-01-12'
  GROUP BY user_id, pipeline_no, bucket
), pipeline_quality AS (
  SELECT user_id, pipeline_no,
         COUNT(flow_5m) / 288.0 AS completeness,
         SUM(COALESCE(GREATEST(flow_5m, 0), 0) * 5.0 / 60.0) AS volume_m3
  FROM five_minute
  GROUP BY user_id, pipeline_no
), daily AS (
  SELECT user_id, SUM(volume_m3) AS volume_m3,
         MAX(completeness) AS best_pipeline_completeness
  FROM pipeline_quality
  GROUP BY user_id
)
SELECT d.user_id, u.station_name AS company_name,
       ROUND(d.volume_m3, 2) AS volume_m3,
       ROUND(d.best_pipeline_completeness * 100, 2) AS best_completeness_pct
FROM daily d
LEFT JOIN asset.user_meter u ON u.user_id = d.user_id
WHERE d.best_pipeline_completeness >= 0.5 AND d.volume_m3 > 1000
ORDER BY d.volume_m3 DESC
```

## 数据覆盖范围

```sql
SELECT MIN(data_date) AS start_date, MAX(data_date) AS end_date
FROM telemetry.scada_observation
```

## 每个用户每日的最新计量诊断

DuckDB 0.10.3 禁止使用 `(user_id, diagnosis_date, created_at) IN (SELECT ...)` 形式的多列元组子查询。统一先聚合出最新时间，再通过完整键 `JOIN`：

```sql
WITH latest AS (
  SELECT user_id, diagnosis_date, MAX(created_at) AS max_created_at
  FROM metering.diagnosis_run
  GROUP BY user_id, diagnosis_date
)
SELECT d.user_id, d.user_name, d.diagnosis_date,
       d.quality_status, d.risk_level,
       d.meter_spec_result, d.summary, d.created_at
FROM metering.diagnosis_run AS d
JOIN latest AS l
  ON l.user_id = d.user_id
 AND l.diagnosis_date = d.diagnosis_date
 AND l.max_created_at = d.created_at
ORDER BY d.diagnosis_date DESC, d.user_id
LIMIT 20
```

如果业务键可能为空或同一最大时间存在重复记录，应先向用户说明数据边界，不得用任意一行冒充唯一最新记录。

## 最新确认安防事件

```sql
SELECT event_id, event_type, camera_id, occurred_at, severity,
       final_decision, final_reason, handling_status
FROM security_events
WHERE final_decision = 'CONFIRMED'
ORDER BY occurred_at DESC
LIMIT 20
```
