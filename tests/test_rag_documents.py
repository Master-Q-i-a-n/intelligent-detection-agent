from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from intelligent_detection_agent.rag.api import create_rag_router, inspect_pdf
from intelligent_detection_agent.rag.documents import DocumentStore, new_document_id
from intelligent_detection_agent.rag.pipeline import RagConfig, RagPipeline, stable_point_id
from intelligent_detection_agent.rag.vision import Cancelled, describe_images
from intelligent_detection_agent.rag.worker import delete_document, migrate_legacy, process_document
from intelligent_detection_agent.rag.docling_worker.ingest import ingest


def pdf_bytes(pages=3):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def client_for(root, authenticated=True):
    app = FastAPI()
    @app.middleware("http")
    async def auth(request, call_next):
        request.state.user = {"user_id": "test"} if authenticated else None
        return await call_next(request)
    app.include_router(create_rag_router(root, start_worker=False))
    return TestClient(app)


def test_upload_ranges_duplicate_and_delete_visibility(tmp_path):
    with client_for(tmp_path) as client:
        content = pdf_bytes()
        response = client.post('/rag/documents', files={'file': ('手册.pdf', content)}, data={'start_page': 2, 'end_page': 3})
        assert response.status_code == 202
        item = response.json()
        assert (item['start_page'], item['end_page'], item['total_pages']) == (2, 3, 3)
        same = client.post('/rag/documents', files={'file': ('重命名.pdf', content)}, data={'start_page': 2, 'end_page': 3})
        assert same.status_code == 200 and same.json()['id'] == item['id']
        other = client.post('/rag/documents', files={'file': ('手册.pdf', content)}, data={'start_page': 1, 'end_page': 1})
        assert other.status_code == 202 and other.json()['id'] != item['id']
        assert len(list((tmp_path / 'dataset/doc').iterdir())) == 2
        assert client.get(f"/rag/documents/{item['id']}/file").status_code == 200
        store = DocumentStore(tmp_path)
        store.update(item['id'], status='ready')
        assert item['id'] in store.ready_ids()
        assert client.delete(f"/rag/documents/{item['id']}").status_code == 202
        assert item['id'] not in store.ready_ids()
        assert client.get(f"/rag/documents/{item['id']}/file").status_code == 404
        assert client.post(f"/rag/documents/{item['id']}/retry").status_code == 409


@pytest.mark.parametrize('start,end', [(0, 1), (3, 2), (1, 4), (None, 2)])
def test_invalid_range_leaves_no_document(tmp_path, start, end):
    with client_for(tmp_path) as client:
        data = {key: value for key, value in [('start_page', start), ('end_page', end)] if value is not None}
        assert client.post('/rag/documents', files={'file': ('a.pdf', pdf_bytes())}, data=data).status_code == 400
        assert client.get('/rag/documents').json()['items'] == []
        assert not list((tmp_path / 'dataset/doc').iterdir())


def test_auth_and_non_pdf(tmp_path):
    with client_for(tmp_path, False) as client:
        assert client.get('/rag/documents').status_code == 401
    with client_for(tmp_path) as client:
        assert client.post('/rag/documents', files={'file': ('a.pdf', b'not pdf')}).status_code == 400


def figures(directory, count):
    (directory / 'chunks').mkdir(parents=True)
    (directory / 'chunks/chunks.jsonl').write_text(json.dumps({'id': 'chunk_1', 'text': '温度传感器安装方式'}) + '\n', encoding='utf-8')
    for index in range(count):
        name = f'fig_{index:03d}'
        folder = directory / 'images' / name
        folder.mkdir(parents=True)
        (folder / f'{name}.png').write_bytes(b'test image')
        (folder / 'metadata.json').write_text(json.dumps({'linked_chunk_ids': ['chunk_1']}), encoding='utf-8')


def test_vision_concurrency_and_resume(tmp_path):
    figures(tmp_path, 8)
    lock = threading.Lock()
    active = maximum = calls = 0
    failed = True
    def handler(request):
        nonlocal active, maximum, calls
        with lock:
            active += 1; calls += 1; maximum = max(maximum, active)
        time.sleep(0.02)
        name = json.loads(json.loads(request.content)['messages'][1]['content'][0]['text'])['name']
        with lock:
            active -= 1
        if failed and name == 'fig_003':
            return httpx.Response(200, json={'choices': [{'message': {'content': 'invalid'}}]})
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
            'name': name, 'description': '温度传感器安装示意图', 'image_type': 'diagram'})}}]})
    progress = []
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match='1 张'):
            describe_images(tmp_path, check_cancelled=lambda: None, progress=lambda *args: progress.append(args),
                            client=client, concurrency=4, sleep=lambda _: None)
        assert 1 < maximum <= 4
        assert len(list((tmp_path / 'vision_cache').glob('*.json'))) == 7
        first_calls = calls
        failed = False
        describe_images(tmp_path, check_cancelled=lambda: None, progress=lambda *args: None,
                        client=client, concurrency=4)
        assert calls - first_calls == 1
    assert len(json.loads((tmp_path / 'rag_image_descriptions.json').read_text(encoding='utf-8'))) == 8


def test_vision_defaults_to_50(tmp_path, monkeypatch):
    figures(tmp_path, 1)
    import intelligent_detection_agent.rag.vision as vision
    real_pool = vision.ThreadPoolExecutor
    seen = []
    def factory(**kwargs):
        seen.append(kwargs['max_workers'])
        return real_pool(**kwargs)
    monkeypatch.delenv('RAG_VISION_CONCURRENCY', raising=False)
    monkeypatch.setattr(vision, 'ThreadPoolExecutor', factory)
    handler = lambda request: httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
        'name': 'fig_000', 'description': '示意图', 'image_type': 'diagram'})}}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        describe_images(tmp_path, check_cancelled=lambda: None, progress=lambda *args: None, client=client)
    assert seen == [50]


def test_ingest_preserves_all_text_and_descriptions(tmp_path):
    figures(tmp_path, 1)
    text = '正文内容。' * 500
    (tmp_path / 'chunks/chunks.jsonl').write_text(json.dumps({'id': 'c1', 'source': 'a.pdf', 'text': text,
        'page_numbers': [9], 'images': ['images/fig_000/']}), encoding='utf-8')
    desc = '传感器位于管道顶部。'
    (tmp_path / 'rag_image_descriptions.json').write_text(json.dumps([{
        'name': 'fig_000', 'description': desc, 'image_type': 'diagram'}]), encoding='utf-8')
    result = ingest(tmp_path, model='deepseek-v4-flash-vision-exp')
    rows = json.loads(result['records_path'].read_text(encoding='utf-8'))
    assert ''.join(row['embed_text'] for row in rows) == text + '\n********\n本段落对应图片描述：' + desc
    assert all(row['text'] == text and row['page_numbers'] == [9] and row['num_tokens'] <= 512 for row in rows)
    assert len({row['chunk_id'] for row in rows}) == len(rows)


def test_delete_retry_and_running_state_guard(tmp_path):
    store = DocumentStore(tmp_path)
    document_id = new_document_id()
    directory = tmp_path / 'dataset/doc' / document_id
    directory.mkdir(parents=True)
    (directory / 'original.pdf').write_bytes(pdf_bytes())
    store.add(document_id=document_id, name='a.pdf', file_hash='hash', start=1, end=3, total=3, force_ocr=False, directory=directory)
    store.update(document_id, status='deleting')
    assert not store.update(document_id, expected=('running',), status='ready')
    pipeline = Mock()
    pipeline._collection_exists.return_value = True
    pipeline.qdrant.delete.side_effect = RuntimeError('offline')
    with pytest.raises(RuntimeError):
        delete_document(store, document_id, pipeline)
    assert directory.exists()
    pipeline.qdrant.delete.side_effect = None
    delete_document(store, document_id, pipeline)
    assert not directory.exists() and store.get(document_id)['status'] == 'deleted'
    assert pipeline.qdrant.delete.call_args.kwargs['wait'] is True


def test_resumes_index_from_last_batch_and_cancelled_never_publishes(tmp_path):
    store = DocumentStore(tmp_path)
    document_id = new_document_id()
    directory = tmp_path / 'dataset/doc' / document_id
    directory.mkdir(parents=True)
    store.add(document_id=document_id, name='a.pdf', file_hash='hash', start=1, end=3, total=3, force_ocr=False, directory=directory)
    store.update(document_id, status='running', stage='index', indexed_count=20)
    pipeline = Mock()
    def indexing(*args, **kwargs):
        assert kwargs['start_index'] == 20
        store.update(document_id, status='deleting')
        kwargs['progress'](30)
    pipeline.index_records.side_effect = indexing
    with pytest.raises(Cancelled):
        process_document(store, document_id, pipeline)
    assert store.get(document_id)['status'] == 'deleting'


def test_point_identity_and_visibility(tmp_path):
    a = {'source': 'a.pdf', 'chunk_id': 'c1', 'document_id': 'a'}
    assert stable_point_id(a) != stable_point_id({**a, 'document_id': 'b'})
    store = DocumentStore(tmp_path)
    pipeline = RagPipeline(RagConfig('http://localhost:6333', 'test', None, None), qdrant=Mock(), root=tmp_path)
    assert pipeline.search('test')['results'] == []
    pipeline.qdrant.query_points.assert_not_called()
    assert pipeline._visibility_kwargs()['query_filter'].must[0].key == 'document_id'


def test_legacy_migration_keeps_points_and_survives_restart(tmp_path):
    store = DocumentStore(tmp_path)
    pipeline = Mock()
    pipeline._collection_exists.return_value = True
    pipeline.qdrant.scroll.return_value = ([SimpleNamespace(id='old-point', payload={
        'source': '旧标准.pdf', 'page_numbers': [7, 8]})], None)
    # 模拟写 payload 成功后进程退出，恢复时 point 已有 document_id。
    original_update = store.update
    store.update = Mock(side_effect=RuntimeError('crash'))
    with pytest.raises(RuntimeError):
        migrate_legacy(store, pipeline)
    assert store.ready_ids() == []
    document_id = store.list()[0]['id']
    assert pipeline.qdrant.set_payload.call_args.kwargs['points'] == ['old-point']
    pipeline.qdrant.scroll.return_value = ([SimpleNamespace(id='old-point', payload={
        'source': '旧标准.pdf', 'document_id': document_id, 'page_numbers': [7, 8]})], None)
    store.update = original_update
    migrate_legacy(store, pipeline)
    assert store.ready_ids() == [document_id]
    assert store.get(document_id)['start_page'] == 7
    pipeline.qdrant.upsert.assert_not_called()


def test_managed_assets_reject_traversal_and_deleted_legacy(tmp_path):
    from intelligent_detection_agent.conversation_agent.rag_support import resolve_rag_image
    store = DocumentStore(tmp_path)
    directory = tmp_path / 'dataset/doc/旧标准'
    directory.mkdir(parents=True)
    (directory / 'image.png').write_bytes(b'image')
    document_id = new_document_id()
    store.add(document_id=document_id, name='旧标准.pdf', file_hash='hash', start=1, end=1, total=1,
              force_ocr=False, directory=directory)
    with store.connect() as db:
        db.execute('UPDATE documents SET legacy=1 WHERE id=?', (document_id,))
    outside = tmp_path / 'dataset/doc/outside.png'
    outside.write_bytes(b'outside')
    with pytest.raises(FileNotFoundError):
        store.asset(document_id, '../outside.png')
    store.update(document_id, status='deleting')
    with pytest.raises(FileNotFoundError, match='删除'):
        resolve_rag_image(tmp_path, '旧标准.pdf', 'image.png')


def test_no_figures_skips_paid_request(tmp_path):
    figures(tmp_path, 0)
    client = Mock()
    describe_images(tmp_path, check_cancelled=lambda: None, progress=lambda *args: None, client=client)
    client.post.assert_not_called()
    assert json.loads((tmp_path / 'rag_image_descriptions.json').read_text()) == []


@pytest.mark.parametrize('text,page,error', [('正常中文正文', 1, '页码'), ('G21G25G31' * 30, 7, '编码异常')])
def test_build_rejects_wrong_page_numbers_and_encoded_text_before_vision(tmp_path, monkeypatch, text, page, error):
    import intelligent_detection_agent.rag.worker as worker_module
    store = DocumentStore(tmp_path)
    document_id = new_document_id()
    directory = tmp_path / 'dataset/doc' / document_id
    directory.mkdir(parents=True)
    executable = tmp_path / 'python.exe'
    executable.write_bytes(b'fixture')
    monkeypatch.setattr(worker_module, 'parser_python', lambda _: executable)
    store.add(document_id=document_id, name='a.pdf', file_hash='hash', start=7, end=7, total=10,
              force_ocr=False, directory=directory)
    store.update(document_id, status='running')
    def parse(*args, **kwargs):
        (directory / 'chunks').mkdir()
        (directory / 'chunks/chunks.jsonl').write_text(json.dumps({'id': 'c1', 'text': text, 'page_numbers': [page]}), encoding='utf-8')
        return SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(worker_module.subprocess, 'Popen', parse)
    vision = Mock()
    monkeypatch.setattr(worker_module, 'describe_images', vision)
    with pytest.raises(ValueError, match=error):
        process_document(store, document_id, Mock())
    vision.assert_not_called()


def test_router_worker_lifecycle(tmp_path, monkeypatch):
    import intelligent_detection_agent.rag.api as api_module
    process = Mock()
    monkeypatch.setattr(api_module.subprocess, 'Popen', Mock(return_value=process))
    app = FastAPI()
    app.include_router(create_rag_router(tmp_path))
    with TestClient(app):
        assert api_module.subprocess.Popen.called
        command = api_module.subprocess.Popen.call_args.args[0]
        assert 'intelligent_detection_agent.rag.worker' in command
    process.wait.assert_called_once_with(timeout=5)
    assert not list((tmp_path / 'database').glob('*.stop'))
