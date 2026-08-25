"""标准文档的向量入库、混合检索和重排序实现。"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import httpx
from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "doc"
    / "用气体超声流量计测量天然气流量"
)
DEFAULT_RECORDS_PATH = DEFAULT_DOCUMENT_ROOT / "ingest" / "records.json"
DEFAULT_COLLECTION_NAME = "gbt18604_2023"
EMBEDDING_MODEL = "qwen3.7-text-embedding"
EMBEDDING_DIMENSIONS = 1024
RERANK_MODEL = "qwen3-rerank"
EMBEDDING_BATCH_SIZE = 20
BM25_OPTIONS = {"tokenizer": "multilingual"}
RERANK_INSTRUCTION = (
    "Given a technical standards query, retrieve passages that directly "
    "answer the query."
)
POINT_NAMESPACE = uuid.UUID("9b9469e7-8bd5-4ba6-b6e0-6da38b70cc32")


class RagConfigurationError(RuntimeError):
    """RAG 运行所需配置不完整。"""


@dataclass(frozen=True)
class RagConfig:
    """从现有项目环境变量读取的独立 RAG 配置。"""

    qdrant_url: str
    collection_name: str
    dashscope_api_key: str | None
    dashscope_workspace_id: str | None

    @classmethod
    def from_env(cls) -> "RagConfig":
        return cls(
            qdrant_url=os.getenv(
                "RAG_QDRANT_URL",
                "http://127.0.0.1:6333",
            ).rstrip("/"),
            collection_name=os.getenv(
                "RAG_COLLECTION_NAME",
                DEFAULT_COLLECTION_NAME,
            ),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            dashscope_workspace_id=os.getenv("DASHSCOPE_WORKSPACE_ID"),
        )

    def require_dashscope(self) -> None:
        missing = []
        if not self.dashscope_api_key:
            missing.append("DASHSCOPE_API_KEY")
        if not self.dashscope_workspace_id:
            missing.append("DASHSCOPE_WORKSPACE_ID")
        if missing:
            raise RagConfigurationError(
                "百炼配置缺失，请先设置：" + ", ".join(missing)
            )


def load_records(records_path: Path) -> list[dict[str, Any]]:
    """读取并校验 Docling 生成的 records.json。"""

    if not records_path.is_file():
        raise FileNotFoundError(f"records.json 不存在：{records_path}")
    data = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("records.json 必须是非空 JSON 数组。")

    required = {"chunk_id", "source", "text", "embed_text"}
    seen_chunk_ids: set[str] = set()
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"第 {index + 1} 条记录不是对象。")
        missing = sorted(required.difference(record))
        if missing:
            raise ValueError(
                f"第 {index + 1} 条记录缺少字段：{', '.join(missing)}"
            )
        chunk_id = str(record["chunk_id"])
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"chunk_id 重复：{chunk_id}")
        seen_chunk_ids.add(chunk_id)
        if not str(record["embed_text"]).strip():
            raise ValueError(f"{chunk_id} 的 embed_text 为空。")
    return data


def stable_point_id(record: dict[str, Any]) -> str:
    """使用来源和 chunk_id 生成可重复 upsert 的 UUID。"""

    key = f"{record['source']}::{record['chunk_id']}"
    return str(uuid.uuid5(POINT_NAMESPACE, key))


def _normalize_image_payload(
    image: dict[str, Any],
    document_root: Path,
) -> dict[str, Any]:
    """补齐图片相对路径，并阻止路径逃逸到文档目录之外。"""

    folder = str(image.get("folder", "")).replace("\\", "/").strip()
    folder_path = PurePosixPath(folder)
    if (
        not folder
        or folder_path.is_absolute()
        or ".." in folder_path.parts
    ):
        raise ValueError(f"非法图片目录：{folder!r}")

    folder_path = PurePosixPath(*folder_path.parts)
    image_name = f"{folder_path.name}.png"
    relative_path = folder_path / image_name
    resolved_path = document_root.joinpath(*relative_path.parts).resolve()
    root = document_root.resolve()
    if root not in resolved_path.parents:
        raise ValueError(f"图片路径超出文档目录：{relative_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"图片文件不存在：{resolved_path}")

    normalized = dict(image)
    normalized["folder"] = folder_path.as_posix().rstrip("/") + "/"
    normalized["image_path"] = relative_path.as_posix()
    return normalized


def build_payload(
    record: dict[str, Any],
    document_root: Path,
) -> dict[str, Any]:
    """构造 Qdrant Payload，不保存机器绝对路径或图片二进制。"""

    metadata = record.get("metadata") or {}
    images = [
        _normalize_image_payload(dict(image), document_root)
        for image in metadata.get("images", [])
    ]
    embed_text = str(record["embed_text"])
    return {
        "chunk_id": str(record["chunk_id"]),
        "source": str(record["source"]),
        "text": str(record["text"]),
        "embed_text": embed_text,
        "page_numbers": list(record.get("page_numbers") or []),
        "headings": list(metadata.get("headings") or []),
        "images": images,
        "content_hash": hashlib.sha256(
            embed_text.encode("utf-8")
        ).hexdigest(),
    }


def batched(
    items: Sequence[Any],
    size: int,
) -> Iterable[Sequence[Any]]:
    """按固定大小切分输入，百炼 Embedding 单批最多 20 条。"""

    if size <= 0:
        raise ValueError("批大小必须大于 0。")
    for start in range(0, len(items), size):
        yield items[start : start + size]


class BailianClient:
    """百炼 Embedding 与 Rerank HTTP 客户端。"""

    def __init__(
        self,
        api_key: str,
        workspace_id: str,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.client = client or httpx.Client(timeout=90.0)
        host = f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com"
        self.embedding_url = (
            host
            + "/api/v1/services/embeddings/"
            "text-embedding/text-embedding"
        )
        self.rerank_url = host + "/compatible-api/v1/reranks"

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response: httpx.Response | None = None
        for attempt in range(3):
            response = self.client.post(url, headers=headers, json=payload)
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt < 2:
                time.sleep(2**attempt)
        assert response is not None
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"百炼接口返回非 JSON 响应：HTTP {response.status_code}"
            ) from exc
        if response.is_error:
            message = body.get("message") or body.get("error") or body
            raise RuntimeError(
                f"百炼接口调用失败：HTTP {response.status_code}，{message}"
            )
        return body

    def embed(
        self,
        texts: Sequence[str],
        *,
        text_type: str,
    ) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > EMBEDDING_BATCH_SIZE:
            raise ValueError("qwen3.7-text-embedding 单批最多 20 条。")
        body = self._post(
            self.embedding_url,
            {
                "model": EMBEDDING_MODEL,
                "input": {"texts": list(texts)},
                "parameters": {
                    "dimension": EMBEDDING_DIMENSIONS,
                    "output_type": "dense",
                    "text_type": text_type,
                },
            },
        )
        embeddings = body.get("output", {}).get("embeddings", [])
        embeddings = sorted(
            embeddings,
            key=lambda item: int(item.get("text_index", 0)),
        )
        vectors = [item.get("embedding") for item in embeddings]
        if len(vectors) != len(texts):
            raise RuntimeError(
                "百炼返回的向量数量与输入数量不一致："
                f"{len(vectors)} != {len(texts)}"
            )
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSIONS:
                raise RuntimeError("百炼返回了错误维度的 Embedding。")
        return vectors

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        body = self._post(
            self.rerank_url,
            {
                "model": RERANK_MODEL,
                "query": query,
                "documents": list(documents),
                "top_n": min(top_n, len(documents)),
                "instruct": RERANK_INSTRUCTION,
            },
        )
        results = body.get("results")
        if not isinstance(results, list):
            raise RuntimeError("百炼 Rerank 响应缺少 results。")
        return results


class RagPipeline:
    """独立的入库、混合召回和重排序流水线。"""

    def __init__(
        self,
        config: RagConfig,
        *,
        qdrant: QdrantClient | None = None,
        bailian: BailianClient | None = None,
    ) -> None:
        self.config = config
        # 对 HTTP 自托管实例启用服务端 Document 推理；Dense 向量仍由百炼生成。
        self.qdrant = qdrant or QdrantClient(
            url=config.qdrant_url,
            timeout=60,
            cloud_inference=True,
        )
        self.bailian = bailian

    def _get_bailian(self) -> BailianClient:
        if self.bailian is None:
            self.config.require_dashscope()
            self.bailian = BailianClient(
                api_key=self.config.dashscope_api_key or "",
                workspace_id=self.config.dashscope_workspace_id or "",
            )
        return self.bailian

    def check_qdrant(self) -> None:
        try:
            self.qdrant.get_collections()
        except Exception as exc:
            raise RuntimeError(
                f"无法连接 Qdrant：{self.config.qdrant_url}。"
                "请先运行 scripts\\start_qdrant.ps1。"
            ) from exc

    def _collection_exists(self) -> bool:
        return self.qdrant.collection_exists(self.config.collection_name)

    def _create_collection(self) -> None:
        self.qdrant.create_collection(
            collection_name=self.config.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    def _validate_collection(self) -> None:
        info = self.qdrant.get_collection(self.config.collection_name)
        vectors = info.config.params.vectors
        sparse_vectors = info.config.params.sparse_vectors
        dense = vectors.get("dense") if isinstance(vectors, dict) else None
        if dense is None or dense.size != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                "现有 Collection 的 dense 向量配置不匹配，"
                "请使用 index --recreate 重建。"
            )
        if not isinstance(sparse_vectors, dict) or "bm25" not in sparse_vectors:
            raise RuntimeError(
                "现有 Collection 缺少 bm25 稀疏向量，"
                "请使用 index --recreate 重建。"
            )

    def prepare_collection(self, *, recreate: bool) -> None:
        self.check_qdrant()
        exists = self._collection_exists()
        if exists and recreate:
            self.qdrant.delete_collection(self.config.collection_name)
            exists = False
        if not exists:
            self._create_collection()
        else:
            self._validate_collection()

    def index_records(
        self,
        records_path: Path,
        *,
        recreate: bool = False,
    ) -> int:
        records_path = records_path.resolve()
        records = load_records(records_path)
        document_root = records_path.parent.parent
        # 先校验远端模型凭据，再执行可能删除 Collection 的 --recreate。
        bailian = self._get_bailian()
        self.prepare_collection(recreate=recreate)

        indexed = 0
        for batch in batched(records, EMBEDDING_BATCH_SIZE):
            texts = [str(record["embed_text"]) for record in batch]
            vectors = bailian.embed(texts, text_type="document")
            points = []
            for record, vector in zip(batch, vectors, strict=True):
                payload = build_payload(record, document_root)
                points.append(
                    models.PointStruct(
                        id=stable_point_id(record),
                        vector={
                            "dense": vector,
                            "bm25": models.Document(
                                text=str(record["embed_text"]),
                                model="qdrant/bm25",
                                options=BM25_OPTIONS,
                            ),
                        },
                        payload=payload,
                    )
                )
            self.qdrant.upsert(
                collection_name=self.config.collection_name,
                points=points,
                wait=True,
            )
            indexed += len(points)
            print(f"已入库 {indexed}/{len(records)}")
        return indexed

    def _query_dense(
        self,
        vector: list[float],
        limit: int,
    ) -> list[Any]:
        return self.qdrant.query_points(
            collection_name=self.config.collection_name,
            query=vector,
            using="dense",
            limit=limit,
            with_payload=True,
        ).points

    def _query_bm25(self, query: str, limit: int) -> list[Any]:
        return self.qdrant.query_points(
            collection_name=self.config.collection_name,
            query=models.Document(
                text=query,
                model="qdrant/bm25",
                options=BM25_OPTIONS,
            ),
            using="bm25",
            limit=limit,
            with_payload=True,
        ).points

    def _query_hybrid(
        self,
        query: str,
        vector: list[float],
        limit: int,
    ) -> list[Any]:
        return self.qdrant.query_points(
            collection_name=self.config.collection_name,
            prefetch=[
                models.Prefetch(
                    query=vector,
                    using="dense",
                    limit=limit,
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="qdrant/bm25",
                        options=BM25_OPTIONS,
                    ),
                    using="bm25",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points

    def search(
        self,
        original_query: str,
        *,
        hybrid_limit: int = 20,
        top_n: int = 5,
    ) -> dict[str, Any]:
        if not original_query.strip():
            raise ValueError("查询问题不能为空。")
        self.check_qdrant()
        if not self._collection_exists():
            raise RuntimeError(
                f"Collection 不存在：{self.config.collection_name}"
            )
        self._validate_collection()

        # 多轮上下文补全由 Agent Skill 完成；检索管线不再二次调用 LLM 改写。
        retrieval_query = original_query.strip()
        vector = self._get_bailian().embed(
            [retrieval_query],
            text_type="query",
        )[0]

        dense_points = self._query_dense(vector, hybrid_limit)
        bm25_points = self._query_bm25(retrieval_query, hybrid_limit)
        hybrid_points = self._query_hybrid(
            retrieval_query,
            vector,
            hybrid_limit,
        )

        dense_scores = {
            str(point.id): {"rank": rank, "score": float(point.score)}
            for rank, point in enumerate(dense_points, start=1)
        }
        bm25_scores = {
            str(point.id): {"rank": rank, "score": float(point.score)}
            for rank, point in enumerate(bm25_points, start=1)
        }
        candidates = []
        for rank, point in enumerate(hybrid_points, start=1):
            payload = dict(point.payload or {})
            candidates.append(
                {
                    "point_id": str(point.id),
                    "hybrid_rank": rank,
                    "rrf_score": float(point.score),
                    "dense": dense_scores.get(str(point.id)),
                    "bm25": bm25_scores.get(str(point.id)),
                    "payload": payload,
                }
            )

        documents = [
            str(item["payload"].get("embed_text", ""))
            for item in candidates
        ]
        reranked = self._get_bailian().rerank(
            retrieval_query,
            documents,
            top_n=top_n,
        )
        final_results = []
        for result in reranked:
            index = int(result["index"])
            if index < 0 or index >= len(candidates):
                raise RuntimeError("Rerank 返回了越界的候选索引。")
            item = dict(candidates[index])
            item["rerank_score"] = float(result["relevance_score"])
            final_results.append(item)

        return {
            "original_query": original_query,
            "retrieval_query": retrieval_query,
            "collection": self.config.collection_name,
            "hybrid_candidates": candidates,
            "results": final_results,
        }
