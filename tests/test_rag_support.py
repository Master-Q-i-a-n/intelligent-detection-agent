from pathlib import Path

import pytest

from intelligent_detection_agent.conversation_agent.rag_support import rag_asset_url, resolve_rag_image


def test_resolve_rag_image_stays_inside_document_root(tmp_path: Path) -> None:
    image = tmp_path / "dataset" / "doc" / "温度传感器" / "images" / "fig_001" / "fig_001.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")

    resolved = resolve_rag_image(tmp_path, "温度传感器.pdf", "images/fig_001/fig_001.png")

    assert resolved == image.resolve()
    assert rag_asset_url("温度传感器.pdf", "images/fig_001/fig_001.png").startswith("/chat/rag-assets/")


@pytest.mark.parametrize(
    ("source", "image_path"),
    [
        ("../温度传感器.pdf", "images/fig.png"),
        ("温度传感器.pdf", "../secret.png"),
        ("温度传感器.pdf", "images/readme.txt"),
    ],
)
def test_resolve_rag_image_rejects_unsafe_paths(tmp_path: Path, source: str, image_path: str) -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_rag_image(tmp_path, source, image_path)
