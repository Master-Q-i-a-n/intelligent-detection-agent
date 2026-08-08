-- 1. 查看三个输入逻辑库中的表/视图
SELECT table_schema, table_name, table_type
FROM information_schema.tables
WHERE table_schema IN ('asset', 'inspection', 'telemetry')
ORDER BY table_schema, table_name;

-- 2. 查询某用户的表具档案
SELECT * FROM asset.user_meter WHERE user_id = '1466892';

-- 3. 查询某用户的历史检定流量点与示值误差
SELECT r.user_id, r.company_name, r.meter_model, r.base_meter_no,
       r.check_time, p.point_no, p.check_flow, p.indication_error, p.repeatability
FROM inspection.meter_check_record r
JOIN inspection.meter_check_point p USING (check_record_id)
WHERE r.user_id = '1466892'
ORDER BY r.check_time, p.point_no;

-- 4. 查询某用户某天的SCADA观测数据
SELECT observed_at, pipeline_no, pressure, temperature,
       standard_instant, standard_cumulative
FROM telemetry.scada_observation
WHERE user_id = '1466892' AND data_date = DATE '2024-12-25'
ORDER BY observed_at, pipeline_no;

-- 5. 检查三类输入能同时关联的用户数
SELECT COUNT(DISTINCT o.user_id) AS linked_user_count
FROM telemetry.scada_observation o
JOIN asset.user_meter u ON u.user_id = o.user_id
JOIN inspection.meter_check_record c ON c.user_id = o.user_id
WHERE o.user_id IS NOT NULL;

