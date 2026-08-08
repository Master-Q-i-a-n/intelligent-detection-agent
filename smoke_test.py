from __future__ import annotations

import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime_libs"))

import duckdb
from fastapi.encoders import jsonable_encoder

from smart_metering import INPUT_DB, RESULT_DB, SmartMeteringService


def main():
    with duckdb.connect(str(INPUT_DB), read_only=True) as con:
        assert con.execute("SELECT COUNT(*) FROM asset.user_meter").fetchone()[0] == 535
        assert con.execute("SELECT COUNT(*) FROM inspection.meter_check_point").fetchone()[0] == 9570
        assert con.execute("SELECT COUNT(*) FROM telemetry.scada_observation").fetchone()[0] == 20794808

    service = SmartMeteringService(use_deep_model=False)
    normal = service.get_saved_result("1609283", date(2025, 1, 12))
    anomaly = service.get_saved_result("2267475", date(2025, 1, 12))
    assert normal and normal["status"] == "completed"
    assert anomaly and anomaly["status"] == "completed"
    assert anomaly["model_gas_state"] == 1
    assert anomaly["observed_gas_state"] == 0
    assert anomaly["risk_level"] == "严重"
    assert jsonable_encoder(normal)
    assert jsonable_encoder(anomaly)

    with duckdb.connect(str(RESULT_DB), read_only=True) as con:
        assert con.execute("SELECT COUNT(*) FROM metering.diagnosis_run").fetchone()[0] >= 2
        assert con.execute("SELECT COUNT(*) FROM metering.anomaly_interval").fetchone()[0] >= 1
        assert con.execute("SELECT COUNT(*) FROM metering.work_order").fetchone()[0] >= 2
    print("SMART_METERING_SMOKE_TEST_OK")


if __name__ == "__main__":
    main()

