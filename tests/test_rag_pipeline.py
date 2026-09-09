from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from qdrant_client import models

from intelligent_detection_agent.rag.pipeline import (
    EMBEDDING_DIMENSIONS,
    BailianClient,
    RagConfig,
    RagPipeline,
    batched,
    build_payload,
    load_records,
    stable_point_id,
)


def _config() -> RagConfig:
    return RagConfig(
        qdrant_url="http://127.0.0.1:6333",
        collection_name="test_collection",
        dashscope_api_key="test-key",
        dashscope_workspace_id="test-workspace",
    )


def _record(chunk_id: str = "chunk_00001") -> dict:
    return {
        "chunk_id": chunk_id,
        "source": "标准.pdf",
        "text": "正文",
        "embed_text": "用于向量化的正文",
        "page_numbers": [1],
        "metadata": {"headings": ["标题"], "images": []},
    }


def test_load_records_and_stable_id(tmp_path: Path) -> None:
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps([_record()], ensure_ascii=False),
        encoding="utf-8",
    )

    records = load_records(records_path)

    assert records[0]["chunk_id"] == "chunk_00001"
    assert stable_point_id(records[0]) == stable_point_id(records[0])


def test_build_payload_adds_relative_image_path(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "fig_003"
    image_dir.mkdir(parents=True)
    (image_dir / "fig_003.png").write_bytes(b"png")
    record = _record()
    record["metadata"]["images"] = [
        {"folder": "images/fig_003/", "page_no": 18}
    ]

    payload = build_payload(record, tmp_path)

    assert payload["images"][0]["folder"] == "images/fig_003/"
    assert (
        payload["images"][0]["image_path"]
        == "images/fig_003/fig_003.png"
    )
    assert not Path(payload["images"][0]["image_path"]).is_absolute()


def test_batched_uses_embedding_limit() -> None:
    batches = list(batched(list(range(41)), 20))
    assert [len(batch) for batch in batches] == [20, 20, 1]


def test_bailian_embedding_and_rerank_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("text-embedding"):
            count = len(body["input"]["texts"])
            return httpx.Response(
                200,
                json={
                    "output": {
                        "embeddings": [
                            {
                                "text_index": index,
                                "embedding": [float(index)]
                                * EMBEDDING_DIMENSIONS,
                            }
                            for index in range(count)
                        ]
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.5},
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = BailianClient("key", "workspace", client=http_client)

    vectors = client.embed(["a", "b"], text_type="document")
    reranked = client.rerank("q", ["a", "b"], top_n=2)

    assert len(vectors) == 2
    assert len(vectors[0]) == EMBEDDING_DIMENSIONS
    assert [item["index"] for item in reranked] == [1, 0]


class _FakeBailian:
    def embed(self, texts, *, text_type):
        assert text_type == "query"
        assert texts == ["原问题"]
        return [[0.1] * EMBEDDING_DIMENSIONS]

    def rerank(self, query, documents, *, top_n):
        assert query == "原问题"
        assert len(documents) == 2
        assert top_n == 2
        return [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.75},
        ]


class _FakeQdrant:
    def __init__(self) -> None:
        self.calls = []
        self.points = [
            SimpleNamespace(
                id="1",
                score=0.8,
                payload={
                    "chunk_id": "chunk_1",
                    "embed_text": "文档一",
                    "text": "文档一",
                    "images": [],
                },
            ),
            SimpleNamespace(
                id="2",
                score=0.7,
                payload={
                    "chunk_id": "chunk_2",
                    "embed_text": "文档二",
                    "text": "文档二",
                    "images": [],
                },
            ),
        ]

    def get_collections(self):
        return SimpleNamespace(collections=[])

    def collection_exists(self, name):
        return True

    def get_collection(self, name):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={
                        "dense": SimpleNamespace(size=EMBEDDING_DIMENSIONS)
                    },
                    sparse_vectors={"bm25": SimpleNamespace()},
                )
            )
        )

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("using") == "bm25":
            points = [self.points[1], self.points[0]]
        else:
            points = self.points
        return SimpleNamespace(points=points)


def test_search_uses_agent_query_for_dense_bm25_rrf_and_rerank() -> None:
    fake_qdrant = _FakeQdrant()
    pipeline = RagPipeline(
        _config(),
        qdrant=fake_qdrant,
        bailian=_FakeBailian(),
    )

    result = pipeline.search("原问题", hybrid_limit=2, top_n=2)

    assert result["retrieval_query"] == "原问题"
    assert "rewritten_query" not in result
    assert result["results"][0]["payload"]["chunk_id"] == "chunk_2"
    assert len(fake_qdrant.calls) == 3
    assert fake_qdrant.calls[0]["using"] == "dense"
    assert fake_qdrant.calls[1]["using"] == "bm25"
    assert fake_qdrant.calls[1]["query"].text == "原问题"
    assert isinstance(
        fake_qdrant.calls[2]["query"],
        models.FusionQuery,
    )
    bm25_document = fake_qdrant.calls[1]["query"]
    assert bm25_document.options == {"tokenizer": "multilingual"}


def test_duplicate_chunk_id_is_rejected(tmp_path: Path) -> None:
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps([_record(), _record()], ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chunk_id 重复"):
        load_records(records_path)
