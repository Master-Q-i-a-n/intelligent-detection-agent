# 项目维护脚本

该目录只存放需要人工执行的离线维护入口，线上 API 启动仍使用项目根目录的 `start_api.ps1`。

- `build_database.py`：从原始计量文件重建主数据、检定和 SCADA 数据。
- `build_vibration_database.py`：重建设备振动数据。
- `metering_cli.py`：执行单企业计量诊断。
- `equipment_cli.py`：训练或执行设备健康诊断并导出 Agent 输入。
- `validate_*.py`：校验数据库、算法结果和 Agent 输入。

所有脚本都应从项目根目录运行，例如：

```powershell
uv run python .\scripts\validate_database.py
```
