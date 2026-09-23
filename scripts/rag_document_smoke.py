"""显式运行小样本真实流水线，使用隔离目录与独立 Collection，不修改业务索引。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import replace

from pathlib import Path
from intelligent_detection_agent.paths import PROJECT_ROOT
from intelligent_detection_agent.rag.api import inspect_pdf
from intelligent_detection_agent.rag.documents import DocumentStore, new_document_id
from intelligent_detection_agent.rag.pipeline import RagConfig, RagPipeline
from intelligent_detection_agent.rag.worker import delete_document, parser_python, process_document
from intelligent_detection_agent.safety_operations.env import load_project_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--force-ocr', action='store_true')
    args = parser.parse_args()
    load_project_env()
    os.environ.setdefault('RAG_DOCLING_PYTHON', str(parser_python(PROJECT_ROOT)))
    run_id = uuid.uuid4().hex[:12]
    root = PROJECT_ROOT / 'output' / 'rag_smoke' / run_id
    store = DocumentStore(root)
    document_id = new_document_id()
    directory = root / 'dataset/doc' / document_id
    directory.mkdir(parents=True)
    shutil.copyfile(args.pdf, directory / 'original.pdf')
    start, end, total = inspect_pdf(directory / 'original.pdf', args.start, args.end)
    store.add(document_id=document_id, name=args.pdf.name, file_hash=hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
              start=start, end=end, total=total, force_ocr=args.force_ocr, directory=directory)
    config = replace(RagConfig.from_env(), collection_name='rag_smoke_' + run_id)
    pipeline = RagPipeline(config, root=root)
    started = time.monotonic()
    print('Smoke workspace:', root, flush=True)
    report = {'workspace': str(root), 'collection': config.collection_name, 'pages': [start, end]}
    try:
        pipeline.check_qdrant()
        store.update(document_id, status='running')
        process_document(store, document_id, pipeline)
        row = store.get(document_id)
        assert row['status'] == 'ready'
        records = json.loads((directory / 'ingest/records.json').read_text(encoding='utf-8'))
        assert all(start <= page <= end for record in records for page in record['page_numbers'])
        found = pipeline.search(args.query)
        assert found['results'], '指定问题未召回文档'
        assert all(item['payload']['document_id'] == document_id for item in found['results'])
        usage = [json.loads(path.read_text(encoding='utf-8')).get('usage', {}) for path in (directory / 'vision_cache').glob('*.json')]
        report.update(status='passed', seconds=round(time.monotonic() - started, 2),
                      images=row['image_count'], chunks=row['chunk_count'], vision_usage=usage,
                      top_result=found['results'][0]['payload']['embed_text'])
        store.update(document_id, status='deleting')
        assert not pipeline.search(args.query)['results']
        delete_document(store, document_id, pipeline)
        assert pipeline.qdrant.count(config.collection_name, exact=True).count == 0
        report['delete_verified'] = True
    except Exception as exc:
        report.update(status='failed', error=str(exc), seconds=round(time.monotonic() - started, 2))
        raise
    finally:
        # 只清理本次创建的独立测试 Collection，失败产物保留用于诊断。
        try:
            if pipeline.qdrant.collection_exists(config.collection_name):
                pipeline.qdrant.delete_collection(config.collection_name)
        finally:
            (root / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
