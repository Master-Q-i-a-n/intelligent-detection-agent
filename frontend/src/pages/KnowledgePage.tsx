import { useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { requestJson } from '../api'

interface DocumentItem {
  id: string; name: string; status: string; stage: string
  start_page: number | null; end_page: number | null; total_pages: number | null
  image_count: number; image_done: number; chunk_count: number; indexed_count: number
  created_at: string; error: string; image_errors: Array<{ name: string; error: string }>
}
interface Listing {
  items: DocumentItem[]; worker_error: string; parser_available: boolean; upload_limit_mb: number
}
const stages = ['build', 'describe', 'ingest', 'index', 'ready']
const stageNames: Record<string, string> = { build: '解析页面', describe: '识别图片', ingest: '整理内容', index: '建立索引', ready: '可检索' }
const statuses: Record<string, string> = { queued: '排队中', running: '处理中', failed: '处理失败', ready: '可检索', deleting: '删除中', migrating: '登记中' }

export function KnowledgePage() {
  const [data, setData] = useState<Listing | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [range, setRange] = useState(false)
  const [start, setStart] = useState('1')
  const [end, setEnd] = useState('')
  const [ocr, setOcr] = useState(false)
  const [busy, setBusy] = useState(false)
  const [selected, setSelected] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<DocumentItem | null>(null)
  const [actionBusy, setActionBusy] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const refresh = useCallback(async (signal?: AbortSignal) => {
    const result = await requestJson<Listing>('/rag/documents', { signal })
    setData(result)
  }, [])
  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    // 上次轮询完成后再排下一次，慢请求不会累积。
    const poll = async () => {
      try { await refresh(controller.signal) } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '无法读取知识库')
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, 2000)
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [refresh])

  async function upload(event: FormEvent) {
    event.preventDefault()
    setError(''); setNotice('')
    if (!file) { setError('请选择 PDF 文件'); return }
    if (file.size > (data?.upload_limit_mb ?? 100) * 1024 * 1024) { setError('文件超过上传大小限制'); return }
    if (range && (!Number.isInteger(Number(start)) || !Number.isInteger(Number(end)) || Number(start) < 1 || Number(end) < Number(start))) {
      setError('请填写有效的起始页和结束页'); return
    }
    const form = new FormData()
    form.append('file', file); form.append('force_ocr', String(ocr))
    if (range) { form.append('start_page', start); form.append('end_page', end) }
    setBusy(true)
    try {
      const result = await requestJson<DocumentItem & { duplicate: boolean }>('/rag/documents', { method: 'POST', body: form, timeoutMs: 120000 })
      setSelected(result.id)
      setNotice(result.duplicate ? '此文件的相同页码范围已存在，已定位到原条目。' : '上传完成，后台将继续处理，离开页面不影响任务。')
      setFile(null)
      if (input.current) input.current.value = ''
      await refresh()
    } catch (reason) { setError(reason instanceof Error ? reason.message : '上传失败') }
    finally { setBusy(false) }
  }
  async function action(item: DocumentItem, remove: boolean) {
    setActionBusy(true); setError('')
    try {
      await requestJson(`/rag/documents/${item.id}${remove ? '' : '/retry'}`, { method: remove ? 'DELETE' : 'POST' })
      setDeleting(null)
      await refresh()
    } catch (reason) { setError(reason instanceof Error ? reason.message : '操作失败') }
    finally { setActionBusy(false) }
  }
  const detail = data?.items.find(item => item.id === selected)
  return <div className="knowledge-page">
    <div className="knowledge-intro"><span>技术资料 / 文档管理</span>
      <strong>{data?.items.filter(item => item.status === 'ready').length ?? 0}<small>份资料可检索</small></strong></div>
    {error && <div className="knowledge-alert" role="alert">{error}</div>}
    {notice && <div className="knowledge-notice" role="status">{notice}</div>}
    {data?.worker_error && <div className="knowledge-alert" role="alert">{data.worker_error}</div>}
    {data && !data.parser_available && <div className="knowledge-alert">解析服务尚未就绪，请在服务器配置 Docling 运行环境；已入库资料仍可使用。</div>}
    <div className="knowledge-grid">
      <form className="knowledge-upload" onSubmit={event => void upload(event)}>
        <h3>添加文档</h3>
        <label className="knowledge-file">选择 PDF 文件<input ref={input} type="file" accept=".pdf,application/pdf" disabled={busy} onChange={event => setFile(event.target.files?.[0] ?? null)} /><small>{file ? file.name : `单文件最多 ${data?.upload_limit_mb ?? 100} MiB`}</small></label>
        <fieldset disabled={busy}><legend>解析范围</legend>
          <label><input type="radio" name="pages" checked={!range} onChange={() => setRange(false)} />全文</label>
          <label><input type="radio" name="pages" checked={range} onChange={() => setRange(true)} />指定页码范围</label>
          {range && <div className="knowledge-range"><label>起始页<input type="number" min="1" step="1" required value={start} onChange={event => setStart(event.target.value)} /></label><span>—</span><label>结束页<input type="number" min={start || '1'} step="1" required value={end} onChange={event => setEnd(event.target.value)} /></label></div>}
          <small>按 PDF 实际页序，从第 1 页开始，包含起止页。</small>
        </fieldset>
        <label className="knowledge-check"><input type="checkbox" checked={ocr} disabled={busy} onChange={event => setOcr(event.target.checked)} />强制 OCR</label>
        <p className="knowledge-hint">扫描件或文本乱码时可开启。相同文件的不同页码范围将分别管理。</p>
        <button className="button button-primary" disabled={busy || !file}>{busy ? '正在上传…' : '上传并处理'}</button>
      </form>
      <section className="knowledge-library" aria-label="文档列表"><div className="knowledge-list-heading"><h3>知识库文档</h3><span>{data?.items.length ?? 0} 份</span></div>
        {!data ? <p>正在读取文档…</p> : data.items.length === 0 ? <div className="knowledge-empty">还没有文档。上传一份 PDF，开始建立知识库。</div> :
          <div className="knowledge-table-wrap"><table><thead><tr><th>文档 / 页码</th><th>状态</th><th>进度</th><th>操作</th></tr></thead><tbody>{data.items.map(item => <tr key={item.id} className={selected === item.id ? 'selected' : ''}>
            <td><button className="knowledge-name" onClick={() => setSelected(item.id)}>{item.name}</button><small>{item.start_page == null ? '历史页码未记录' : `第 ${item.start_page}–${item.end_page} 页`}{item.total_pages ? ` / 共 ${item.total_pages} 页` : ''}</small></td>
            <td><span className={`knowledge-status ${item.status}`}>{statuses[item.status] ?? item.status}</span></td>
            <td><span>{stageNames[item.stage] ?? item.stage}</span><small>图片 {item.image_done}/{item.image_count} · 分块 {item.indexed_count}/{item.chunk_count}</small></td>
            <td><div className="knowledge-actions"><button onClick={() => setSelected(item.id)}>详情</button>{item.status === 'failed' && <button disabled={actionBusy} onClick={() => void action(item, false)}>重试</button>}<button disabled={actionBusy || item.status === 'deleting'} onClick={() => setDeleting(item)}>删除</button></div></td>
          </tr>)}</tbody></table></div>}
        {detail && <section className="knowledge-detail" aria-label="文档详情"><h3>{detail.name}</h3><ol className="knowledge-stages">{stages.map((stage, index) => <li key={stage} className={index <= stages.indexOf(detail.stage) ? 'reached' : ''}>{stageNames[stage]}</li>)}</ol>
          <p>上传于 {new Date(detail.created_at).toLocaleString('zh-CN')}</p>
          {detail.error && <p className="knowledge-alert">{detail.error}</p>}
          {detail.image_errors.length > 0 && <ul>{detail.image_errors.map(item => <li key={item.name}>{item.name}：{item.error}</li>)}</ul>}
          {detail.status !== 'deleting' && <a href={`/rag/documents/${detail.id}/file#page=${detail.start_page ?? 1}`} target="_blank" rel="noreferrer">查看原 PDF ↗</a>}
        </section>}
      </section>
    </div>
    {deleting && <div className="knowledge-modal"><section role="dialog" aria-modal="true" aria-labelledby="delete-document-title"><h3 id="delete-document-title">删除这份文档？</h3><p>{deleting.name} · 第 {deleting.start_page ?? '?'}–{deleting.end_page ?? '?'} 页</p><p>该条目将立即退出检索，随后清理原文件、解析图片和索引。其他页码范围的条目不受影响。</p><div className="knowledge-actions"><button autoFocus disabled={actionBusy} onClick={() => setDeleting(null)}>取消</button><button disabled={actionBusy} onClick={() => void action(deleting, true)}>{actionBusy ? '正在提交…' : '确认删除'}</button></div></section></div>}
  </div>
}
