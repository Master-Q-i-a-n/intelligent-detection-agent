import os
from pathlib import Path

from intelligent_detection_agent.safety_operations.env import load_project_env


def test_load_project_env_without_overriding_shell(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nARK_MODEL_ID=env-model\nARK_API_KEY='env-key'\nexport SAFETY_AGENT_TOKEN=env-token\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ARK_MODEL_ID", "shell-model")
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("SAFETY_AGENT_TOKEN", raising=False)

    load_project_env(env_file)

    assert os.environ["ARK_MODEL_ID"] == "shell-model"
    assert os.environ["ARK_API_KEY"] == "env-key"
    assert os.environ["SAFETY_AGENT_TOKEN"] == "env-token"
