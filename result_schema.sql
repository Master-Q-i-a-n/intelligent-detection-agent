CREATE SCHEMA IF NOT EXISTS metering;
CREATE SCHEMA IF NOT EXISTS equipment;
CREATE SCHEMA IF NOT EXISTS operations;
CREATE SCHEMA IF NOT EXISTS inspection;

CREATE TABLE IF NOT EXISTS metering.diagnosis_run (
    run_id VARCHAR PRIMARY KEY,
    user_id VARCHAR,
    diagnosis_date DATE,
    user_name VARCHAR,
    status VARCHAR,
    quality_status INTEGER,
    model_gas_state INTEGER,
    observed_gas_state INTEGER,
    observed_volume DOUBLE,
    predicted_normal_volume DOUBLE,
    baseline_missing_volume DOUBLE,
    meter_bias_volume DOUBLE,
    makeup_volume DOUBLE,
    risk_score DOUBLE,
    risk_level VARCHAR,
    meter_spec_result VARCHAR,
    summary VARCHAR,
    details_json VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS metering.anomaly_interval (
    interval_id VARCHAR PRIMARY KEY,
    run_id VARCHAR,
    user_id VARCHAR,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    anomaly_type VARCHAR,
    pipeline_no INTEGER,
    observed_value DOUBLE,
    expected_value DOUBLE,
    estimated_missing_volume DOUBLE,
    severity VARCHAR,
    evidence_json VARCHAR
);

CREATE TABLE IF NOT EXISTS metering.work_order (
    work_order_id VARCHAR PRIMARY KEY,
    run_id VARCHAR,
    user_id VARCHAR,
    diagnosis_date DATE,
    order_type VARCHAR,
    priority VARCHAR,
    status VARCHAR,
    title VARCHAR,
    description VARCHAR,
    checklist_json VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS equipment.health_diagnosis (
    diagnosis_id VARCHAR PRIMARY KEY,
    user_id VARCHAR,
    company_name VARCHAR,
    meter_id VARCHAR,
    diagnosis_date DATE,
    predicted_stage VARCHAR,
    predicted_stage_name VARCHAR,
    predicted_health_index DOUBLE,
    confidence DOUBLE,
    coarse_status VARCHAR,
    risk_level VARCHAR,
    trend_label VARCHAR,
    model_version VARCHAR,
    details_json VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS equipment.health_trend (
    trend_id VARCHAR PRIMARY KEY,
    user_id VARCHAR,
    start_date DATE,
    end_date DATE,
    trend_label VARCHAR,
    health_index_start DOUBLE,
    health_index_end DOUBLE,
    daily_slope DOUBLE,
    maximum_daily_drop DOUBLE,
    stage_sequence VARCHAR,
    model_version VARCHAR,
    details_json VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 对话 Agent 的跨模块工单与审计，和诊断结果共用一个 DuckDB。
CREATE TABLE IF NOT EXISTS operations.work_order (
    work_order_id VARCHAR PRIMARY KEY,
    idempotency_key VARCHAR UNIQUE NOT NULL,
    source_module VARCHAR NOT NULL,
    user_id VARCHAR,
    source_reference_json VARCHAR NOT NULL,
    priority VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    description VARCHAR NOT NULL,
    checklist_json VARCHAR NOT NULL,
    created_by VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS operations.work_order_audit (
    audit_id VARCHAR PRIMARY KEY,
    work_order_id VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    operator VARCHAR NOT NULL,
    details_json VARCHAR NOT NULL,
    acted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 智能解读只缓存通过结构校验的 LLM 报告；算法输入变化后会产生新的指纹。
CREATE TABLE IF NOT EXISTS inspection.workflow_report (
    report_id VARCHAR PRIMARY KEY,
    input_fingerprint VARCHAR UNIQUE NOT NULL,
    module VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    diagnosis_date DATE NOT NULL,
    workflow_route VARCHAR NOT NULL,
    workflow_version VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    report_json VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
