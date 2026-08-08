CREATE SCHEMA IF NOT EXISTS asset;
CREATE SCHEMA IF NOT EXISTS inspection;
CREATE SCHEMA IF NOT EXISTS telemetry;
CREATE SCHEMA IF NOT EXISTS equipment;
CREATE SCHEMA IF NOT EXISTS vibration;

CREATE TABLE IF NOT EXISTS asset.user_meter (
    user_id VARCHAR PRIMARY KEY,
    station_type VARCHAR,
    meter_count INTEGER,
    station_name VARCHAR,
    address VARCHAR,
    meter_brand VARCHAR,
    meter_type VARCHAR,
    meter_model VARCHAR,
    range_text VARCHAR,
    quantity_min DOUBLE,
    quantity_max DOUBLE,
    prepaid_flag VARCHAR,
    scada_status VARCHAR,
    abnormal_meter_no VARCHAR,
    installation_status VARCHAR,
    source_file VARCHAR,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inspection.meter_check_record (
    check_record_id VARCHAR PRIMARY KEY,
    user_id VARCHAR,
    company_name VARCHAR,
    meter_brand VARCHAR,
    meter_type VARCHAR,
    meter_model VARCHAR,
    nominal_range VARCHAR,
    diameter VARCHAR,
    base_meter_no VARCHAR,
    corrector_no VARCHAR,
    backup_meter_no VARCHAR,
    offline_time TIMESTAMP,
    online_time TIMESTAMP,
    check_time TIMESTAMP,
    maintenance_status VARCHAR,
    parts_replacement VARCHAR,
    check_status VARCHAR,
    qualification_status VARCHAR,
    linear_data VARCHAR,
    source_row INTEGER,
    source_file VARCHAR,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inspection.meter_check_point (
    check_record_id VARCHAR,
    point_no INTEGER,
    check_flow DOUBLE,
    indication_error DOUBLE,
    repeatability DOUBLE,
    PRIMARY KEY (check_record_id, point_no)
);

CREATE TABLE IF NOT EXISTS inspection.meter_repair (
    repair_record_id VARCHAR PRIMARY KEY,
    user_id VARCHAR,
    company_name VARCHAR,
    meter_brand VARCHAR,
    meter_type VARCHAR,
    meter_model VARCHAR,
    diameter VARCHAR,
    meter_no VARCHAR,
    fault_description VARCHAR,
    repair_description VARCHAR,
    offline_time TIMESTAMP,
    online_time TIMESTAMP,
    source_row INTEGER,
    source_file VARCHAR,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telemetry.import_file_log (
    source_file VARCHAR PRIMARY KEY,
    observation_date DATE,
    entity_key VARCHAR,
    user_id VARCHAR,
    row_count BIGINT,
    pipeline_count INTEGER,
    status VARCHAR,
    error_message VARCHAR,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS equipment.vibration_sensor (
    sensor_id VARCHAR PRIMARY KEY,
    user_id VARCHAR,
    company_name VARCHAR,
    meter_id VARCHAR,
    sensor_position VARCHAR,
    axis_count INTEGER,
    sampling_rate_hz INTEGER,
    window_length INTEGER,
    source_type VARCHAR,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vibration.health_label_dictionary (
    stage_label VARCHAR PRIMARY KEY,
    stage_order INTEGER,
    stage_name VARCHAR,
    coarse_label INTEGER,
    health_index_min DOUBLE,
    health_index_max DOUBLE,
    description VARCHAR
);

CREATE TABLE IF NOT EXISTS vibration.trajectory_dictionary (
    trajectory_type VARCHAR PRIMARY KEY,
    trajectory_name VARCHAR,
    stage_sequence VARCHAR,
    description VARCHAR
);

CREATE TABLE IF NOT EXISTS vibration.build_manifest (
    build_id VARCHAR PRIMARY KEY,
    start_date DATE,
    end_date DATE,
    enterprise_count INTEGER,
    date_count INTEGER,
    window_count BIGINT,
    sample_rate_hz INTEGER,
    window_length INTEGER,
    axis_count INTEGER,
    synthetic_flag BOOLEAN,
    source_root VARCHAR,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
