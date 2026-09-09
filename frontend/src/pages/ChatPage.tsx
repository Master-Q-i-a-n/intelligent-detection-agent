import { useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties, ClipboardEvent as ReactClipboardEvent, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '../api'
import type { ChatArtifact, ChatArtifactType, ChatAttachment, ChatInterrupt, ChatStreamEvent, ChatThreadSummary, ChatTodo, ChatTurnResponse } from '../types'

interface DisplayMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  generator?: string
  artifactIds: string[]
  attachments: ChatAttachment[]
}

interface PendingImage {
  key: string
  file: File
  previewUrl: string
}

const MAX_IMAGES_PER_MESSAGE = 4
const MAX_IMAGE_BYTES = 8 * 1024 * 1024
const MAX_IMAGE_TOTAL_BYTES = 24 * 1024 * 1024
const SUPPORTED_IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp'])

type EvidenceArtifactType = Exclude<ChatArtifactType, 'rag_retrieval'>

interface QueryPayload {
  type: 'query_result'
  query_id: string
  source: string
  sql: string
  columns: string[]
  rows: Array<Record<string, unknown>>
  row_count: number
  truncated: boolean
  elapsed_ms: number
}

interface ReportChart {
  id: string
  type: 'line' | 'bar' | 'pie' | 'scatter'
  title: string
  unit: string
  source_query_id: string
  x_field: string
  y_fields: string[]
  series_names?: string[]
}

interface ReportPayload {
  type: 'report'
  report_id: string
  title: string
  generated_at: string
  summary: string
  report_markdown: string
  datasets: QueryPayload[]
  charts: ReportChart[]
  recommendations: string[]
}

function newThreadId(): string {
  return `chat_${crypto.randomUUID().replaceAll('-', '')}`
}

function generatorLabel(generator: string): string {
  // 后端保留完整 provider 链路用于追踪，界面只显示用户关心的最终模型名。
  return generator.split(':').filter(Boolean).at(-1) || generator
}

function isNarrowViewport(): boolean {
  return typeof window !== 'undefined' && typeof window.matchMedia === 'function'
    ? window.matchMedia('(max-width: 1050px)').matches
    : false
}

function MarkdownImage({ src, alt }: { src?: string; alt?: string }) {
  // 只允许后端校验过的同源知识库图片，避免模型输出任意远程跟踪图片。
  if (!src?.startsWith('/chat/rag-assets/')) return <span className="chat-markdown-image-blocked">[图片地址不可用]</span>
  return <img src={src} alt={alt || '知识库资料图片'} loading="lazy" />
}

function asQueryPayload(artifact: ChatArtifact): QueryPayload | null {
  if (artifact.type !== 'query_result') return null
  return artifact.payload as unknown as QueryPayload
}

function asReportPayload(artifact: ChatArtifact): ReportPayload | null {
  if (artifact.type !== 'report') return null
  return artifact.payload as unknown as ReportPayload
}

function chartOption(chart: ReportChart, dataset: QueryPayload): EChartsOption {
  const rows = dataset.rows || []
  const xValues = rows.map((row) => String(row[chart.x_field] ?? '—'))
  const colors = ['#21d4d0', '#3f91ff', '#f2ad35', '#a67cff', '#ff5263']
  const common: EChartsOption = {
    animationDuration: 450,
    color: colors,
    tooltip: { trigger: chart.type === 'pie' ? 'item' : 'axis', backgroundColor: '#071924ee', borderColor: '#275064', textStyle: { color: '#e4f2f8' } },
  }
  if (chart.type === 'pie') {
    const yField = chart.y_fields[0]
    return {
      ...common,
      legend: { bottom: 0, textStyle: { color: '#87a1b1' } },
      series: [{
        type: 'pie', radius: ['42%', '70%'], center: ['50%', '43%'],
        data: rows.map((row) => ({ name: String(row[chart.x_field] ?? '未命名'), value: Number(row[yField] ?? 0) })),
        label: { color: '#c8dce7', formatter: '{b}: {c}' },
      }],
    } as EChartsOption
  }
  if (chart.type === 'scatter') {
    const xField = chart.x_field
    return {
      ...common,
      grid: { top: 24, right: 28, bottom: 48, left: 64 },
      xAxis: { type: 'value', name: xField, axisLabel: { color: '#87a1b1' }, splitLine: { lineStyle: { color: '#183444' } } },
      yAxis: { type: 'value', name: chart.unit, axisLabel: { color: '#87a1b1' }, splitLine: { lineStyle: { color: '#183444' } } },
      series: chart.y_fields.map((field, index) => ({
        name: chart.series_names?.[index] || field,
        type: 'scatter',
        data: rows.map((row) => [Number(row[xField]), Number(row[field])]),
      })),
    } as EChartsOption
  }
  const seriesType: 'line' | 'bar' = chart.type === 'bar' ? 'bar' : 'line'
  return {
    ...common,
    legend: { top: 0, textStyle: { color: '#87a1b1' } },
    grid: { top: 42, right: 28, bottom: 48, left: 64 },
    xAxis: {
      type: 'category', data: xValues, axisLabel: { color: '#87a1b1', hideOverlap: true },
      axisLine: { lineStyle: { color: '#254556' } }, axisTick: { show: false },
    },
    yAxis: {
      type: 'value', name: chart.unit, nameTextStyle: { color: '#87a1b1' }, axisLabel: { color: '#87a1b1' },
      splitLine: { lineStyle: { color: '#183444', type: 'dashed' } },
    },
    series: chart.y_fields.map((field, index) => ({
      name: chart.series_names?.[index] || field,
      type: seriesType,
      data: rows.map((row) => row[field] === null || row[field] === undefined ? null : Number(row[field])),
      showSymbol: chart.type !== 'line' || rows.length < 20,
      connectNulls: false,
      smooth: false,
      barMaxWidth: 36,
      lineStyle: { width: 2 },
    })),
  } as EChartsOption
}

function escapeHtml(value: string): string {
  return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;')
}

function ReportViewer({ report }: { report: ReportPayload }) {
  const chartRefs = useRef<Record<string, ReactECharts | null>>({})
  const reportBodyRef = useRef<HTMLDivElement | null>(null)
  const chartContainerRef = useRef<HTMLDivElement | null>(null)
  const datasets = useMemo(() => Object.fromEntries(report.datasets.map((item) => [item.query_id, item])), [report.datasets])

  useEffect(() => {
    const container = chartContainerRef.current
    if (!container || typeof ResizeObserver === 'undefined') return
    let frame = 0
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        Object.values(chartRefs.current).forEach((instance) => instance?.getEchartsInstance().resize())
      })
    })
    observer.observe(container)
    return () => { observer.disconnect(); cancelAnimationFrame(frame) }
  }, [])

  function downloadHtml() {
    const chartImages = report.charts.map((chart) => {
      const image = chartRefs.current[chart.id]?.getEchartsInstance().getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#ffffff' })
      return image ? `<section><h2>${escapeHtml(chart.title)}</h2><img src="${image}" alt="${escapeHtml(chart.title)}"></section>` : ''
    }).join('\n')
    const tables = report.datasets.map((dataset) => {
      const head = dataset.columns.map((column) => `<th>${escapeHtml(column)}</th>`).join('')
      const body = dataset.rows.map((row) => `<tr>${dataset.columns.map((column) => `<td>${escapeHtml(String(row[column] ?? '—'))}</td>`).join('')}</tr>`).join('')
      return `<section><h2>数据来源 ${escapeHtml(dataset.query_id)}</h2><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table><details><summary>SQL</summary><pre>${escapeHtml(dataset.sql)}</pre></details></section>`
    }).join('\n')
    const recommendations = report.recommendations.map((item) => `<li>${escapeHtml(item)}</li>`).join('')
    // reportBodyRef 来自 ReactMarkdown，未启用 rehypeRaw，因此下载内容也不会执行报告中的原始 HTML。
    const renderedMarkdown = reportBodyRef.current?.innerHTML || `<p>${escapeHtml(report.report_markdown)}</p>`
    const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>${escapeHtml(report.title)}</title><style>body{font:15px/1.7 "Microsoft YaHei",sans-serif;color:#173142;max-width:1100px;margin:36px auto;padding:0 24px}h1{border-bottom:3px solid #18b9bb;padding-bottom:12px}h2{margin-top:30px;color:#174d65}img{display:block;max-width:100%;margin:12px auto}pre{overflow:auto;white-space:pre-wrap;background:#f2f6f8;padding:14px}code{font-family:Consolas,monospace}table{width:100%;border-collapse:collapse;font-size:13px}th,td{border:1px solid #cbd9df;padding:7px;text-align:left}th{background:#eaf2f5}.meta{color:#66808d}</style></head><body><h1>${escapeHtml(report.title)}</h1><p class="meta">生成时间：${escapeHtml(report.generated_at)}</p><h2>摘要</h2><p>${escapeHtml(report.summary)}</p><div>${renderedMarkdown}</div>${chartImages}${tables}<h2>建议</h2><ul>${recommendations}</ul></body></html>`
    const url = URL.createObjectURL(new Blob([html], { type: 'text/html;charset=utf-8' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${report.title.replace(/[\\/:*?"<>|]/g, '_') || '智能检测报告'}.html`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return (
    <article className="chat-report-card">
      <header>
        <div><small>STRUCTURED REPORT</small><h3>{report.title}</h3><p>{report.generated_at}</p></div>
        <button className="button button-primary" type="button" onClick={downloadHtml}>下载含图报告</button>
      </header>
      <div className="chat-report-summary"><strong>摘要</strong><p>{report.summary}</p></div>
      <div className="chat-report-markdown" ref={reportBodyRef}><ReactMarkdown remarkPlugins={[remarkGfm]}>{report.report_markdown}</ReactMarkdown></div>
      <div className="chat-report-charts" ref={chartContainerRef}>
        {report.charts.map((chart) => {
          const dataset = datasets[chart.source_query_id]
          if (!dataset) return null
          return (
            <section key={chart.id}>
              <h4>{chart.title}<small>{chart.unit}</small></h4>
              <ReactECharts
                ref={(instance) => { chartRefs.current[chart.id] = instance }}
                option={chartOption(chart, dataset)}
                notMerge
                style={{ width: '100%', height: 320 }}
              />
              <p>数据来源：{chart.source_query_id}</p>
            </section>
          )
        })}
      </div>
      {report.recommendations.length > 0 && <div className="chat-report-recommendations"><strong>建议</strong><ul>{report.recommendations.map((item) => <li key={item}>{item}</li>)}</ul></div>}
    </article>
  )
}

function QueryResultCard({ query }: { query: QueryPayload }) {
  return (
    <details className="chat-query-card">
      <summary><span>{query.source.toUpperCase()}</span><strong>{query.query_id}</strong><small>{query.row_count} 行 · {query.elapsed_ms} ms</small></summary>
      <div className="chat-query-body">
        <pre>{query.sql}</pre>
        {query.rows.length ? (
          <div className="table-scroll"><table><thead><tr>{query.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>
            {query.rows.slice(0, 100).map((row, index) => <tr key={index}>{query.columns.map((column) => <td key={column} title={String(row[column] ?? '')}>{String(row[column] ?? '—')}</td>)}</tr>)}
          </tbody></table></div>
        ) : <p className="chat-no-rows">查询成功，但没有符合条件的数据。</p>}
        {query.truncated && <p className="chat-data-notice">结果已按安全上限截断。</p>}
      </div>
    </details>
  )
}

export function ChatPage() {
  const [threadId, setThreadId] = useState(newThreadId)
  const [messages, setMessages] = useState<DisplayMessage[]>([{
    id: 'welcome', role: 'assistant',
    content: '可以查询用气、智能计量、智能设备和安全作业数据，也可以生成带图报告或在确认后创建工单。',
    artifactIds: [], attachments: [],
  }])
  const [artifacts, setArtifacts] = useState<Record<string, ChatArtifact>>({})
  const [interruptState, setInterruptState] = useState<ChatInterrupt | null>(null)
  const [input, setInput] = useState('')
  const [pendingImages, setPendingImages] = useState<PendingImage[]>([])
  const [uploading, setUploading] = useState(false)
  const [resumeText, setResumeText] = useState('')
  const [editText, setEditText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [configured, setConfigured] = useState<boolean | null>(null)
  const [streamingText, setStreamingText] = useState('')
  const [todos, setTodos] = useState<ChatTodo[]>([])
  const [todoOpen, setTodoOpen] = useState(false)
  const [panelOpen, setPanelOpen] = useState(false)
  const [artifactView, setArtifactView] = useState<EvidenceArtifactType>('report')
  const [selectedArtifactIds, setSelectedArtifactIds] = useState<string[]>([])
  const [narrowPanel, setNarrowPanel] = useState(isNarrowViewport)
  const [splitPercent, setSplitPercent] = useState(44)
  const [threads, setThreads] = useState<ChatThreadSummary[]>([])
  const [historyOpen, setHistoryOpen] = useState(true)
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historyError, setHistoryError] = useState('')
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const pendingImagesRef = useRef<PendingImage[]>([])
  const messageEndRef = useRef<HTMLDivElement | null>(null)
  const workspaceRef = useRef<HTMLDivElement | null>(null)
  const panelRef = useRef<HTMLElement | null>(null)
  const panelCloseRef = useRef<HTMLButtonElement | null>(null)
  const artifactTriggerRef = useRef<HTMLButtonElement | null>(null)
  const draggingRef = useRef(false)
  const requestTokenRef = useRef(0)
  const deltaBufferRef = useRef('')
  const deltaTimerRef = useRef<number | null>(null)
  const receivedDoneRef = useRef(false)
  const todoWasShownRef = useRef(false)
  const seenArtifactIdsRef = useRef<Set<string>>(new Set())

  useEffect(() => {
    const controller = new AbortController()
    api.chatStatus(controller.signal).then((status) => setConfigured(status.configured)).catch(() => setConfigured(false))
    api.chatThreads(controller.signal)
      .then((result) => setThreads(result.items || []))
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setHistoryError(reason instanceof Error ? reason.message : '历史对话读取失败')
      })
      .finally(() => { if (!controller.signal.aborted) setHistoryLoading(false) })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    const media = window.matchMedia('(max-width: 1050px)')
    const update = () => setNarrowPanel(media.matches)
    update()
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])

  useEffect(() => () => {
    controllerRef.current?.abort()
    if (deltaTimerRef.current !== null) window.clearTimeout(deltaTimerRef.current)
    pendingImagesRef.current.forEach((image) => URL.revokeObjectURL(image.previewUrl))
  }, [])

  useEffect(() => {
    pendingImagesRef.current = pendingImages
  }, [pendingImages])

  useEffect(() => {
    if (typeof messageEndRef.current?.scrollIntoView === 'function') {
      messageEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }
  }, [busy, interruptState, messages, streamingText])

  useEffect(() => {
    if (!panelOpen || !narrowPanel) return
    const panel = panelRef.current
    if (!panel) return
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    window.requestAnimationFrame(() => panelCloseRef.current?.focus())

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        closeArtifactPanel()
        return
      }
      if (event.key !== 'Tab') return
      const focusable = Array.from(panel.querySelectorAll<HTMLElement>('button:not(:disabled), a[href], summary, [tabindex]:not([tabindex="-1"])'))
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault(); last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault(); first.focus()
      }
    }
    panel.addEventListener('keydown', handleKeyDown)
    return () => {
      panel.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = previousOverflow
    }
  }, [narrowPanel, panelOpen])

  function mergeArtifact(artifact: ChatArtifact) {
    setArtifacts((current) => ({ ...current, [artifact.id]: artifact }))
  }

  function clearStreamingText() {
    if (deltaTimerRef.current !== null) window.clearTimeout(deltaTimerRef.current)
    deltaTimerRef.current = null
    deltaBufferRef.current = ''
    setStreamingText('')
  }

  // 约 60ms 合并一次 token，避免高速流式事件引发无意义的逐字重绘。
  function queueAnswerDelta(delta: string) {
    deltaBufferRef.current += delta
    if (deltaTimerRef.current !== null) return
    deltaTimerRef.current = window.setTimeout(() => {
      const buffered = deltaBufferRef.current
      deltaBufferRef.current = ''
      deltaTimerRef.current = null
      if (buffered) setStreamingText((current) => current + buffered)
    }, 60)
  }

  function applyTodos(items: ChatTodo[]) {
    setTodos(items)
    if (items.length && !todoWasShownRef.current) {
      todoWasShownRef.current = true
      setTodoOpen(true)
    }
  }

  function applyResponse(response: ChatTurnResponse) {
    const artifactIds = response.artifacts.map((artifact) => artifact.id)
    const hasNewReport = response.artifacts.some((artifact) => artifact.type === 'report' && !seenArtifactIdsRef.current.has(artifact.id))
    response.artifacts.forEach((artifact) => seenArtifactIdsRef.current.add(artifact.id))
    setArtifacts((current) => {
      const next = { ...current }
      response.artifacts.forEach((artifact) => { next[artifact.id] = artifact })
      return next
    })
    if (response.message) {
      setMessages((current) => [...current, {
        id: crypto.randomUUID(), role: 'assistant', content: response.message,
        generator: response.generator, artifactIds, attachments: [],
      }])
    }
    if (hasNewReport) {
      // 新报告始终自动展开；SQL 和工单只通过回答下方按钮主动打开。
      artifactTriggerRef.current = null
      setSelectedArtifactIds(artifactIds)
      setArtifactView('report')
      setPanelOpen(true)
    }
    if (response.todos) applyTodos(response.todos)
    const nextInterrupt = response.status === 'interrupted' ? response.interrupt || null : null
    // 与确认卡片在同一次状态更新中回填参数，避免卡片先出现而内容仍为空的瞬间。
    if (nextInterrupt?.kind === 'work_order_approval') {
      const args = nextInterrupt.action?.arguments || nextInterrupt.action?.args || {}
      setEditText(JSON.stringify(args, null, 2))
    } else {
      setEditText('')
    }
    setResumeText('')
    setInterruptState(nextInterrupt)
  }

  async function refreshHistory() {
    try {
      const result = await api.chatThreads()
      setThreads(result.items || [])
      setHistoryError('')
    } catch (reason) {
      setHistoryError(reason instanceof Error ? reason.message : '历史对话刷新失败')
    }
  }

  async function openHistoryThread(nextThreadId: string) {
    if (nextThreadId === threadId) return
    controllerRef.current?.abort()
    requestTokenRef.current += 1
    clearStreamingText()
    setHistoryLoading(true)
    setBusy(false)
    setHistoryError('')
    try {
      const detail = await api.chatThread(nextThreadId)
      setThreadId(detail.thread.thread_id)
      setMessages(detail.messages.map((message) => ({
        id: message.id,
        role: message.role,
        content: message.content,
        generator: message.generator || undefined,
        artifactIds: message.artifact_ids || [],
        attachments: message.attachments || [],
      })))
      setArtifacts(Object.fromEntries(detail.artifacts.map((artifact) => [artifact.id, artifact])))
      setTodos(detail.todos || [])
      setTodoOpen(false)
      todoWasShownRef.current = (detail.todos || []).length > 0
      setInterruptState(detail.interrupt || null)
      if (detail.interrupt?.kind === 'work_order_approval') {
        const args = detail.interrupt.action?.arguments || detail.interrupt.action?.args || {}
        setEditText(JSON.stringify(args, null, 2))
      } else {
        setEditText('')
      }
      setResumeText('')
      setInput('')
      clearPendingImages()
      setError(detail.last_error || '')
      setPanelOpen(false)
      setSelectedArtifactIds([])
      seenArtifactIdsRef.current = new Set(detail.artifacts.map((artifact) => artifact.id))
      if (isNarrowViewport()) setHistoryOpen(false)
    } catch (reason) {
      setHistoryError(reason instanceof Error ? reason.message : '历史对话打开失败')
    } finally {
      setHistoryLoading(false)
    }
  }

  async function deleteHistoryThread(nextThreadId: string) {
    setHistoryError('')
    if (nextThreadId === threadId) {
      // 永久删除当前会话时先忽略迟到事件；后端还会主动终止对应 Agent run。
      controllerRef.current?.abort()
      requestTokenRef.current += 1
      clearStreamingText()
      setBusy(false)
    }
    try {
      await api.deleteChatThread(nextThreadId)
      setDeleteConfirmId(null)
      if (nextThreadId === threadId) resetConversation()
      await refreshHistory()
    } catch (reason) {
      setHistoryError(reason instanceof Error ? reason.message : '历史对话删除失败')
    }
  }

  function handleStreamEvent(streamEvent: ChatStreamEvent, requestToken: number) {
    if (requestTokenRef.current !== requestToken) return
    const data = streamEvent.data
    if (streamEvent.event === 'todo') {
      applyTodos(Array.isArray(data.items) ? data.items as ChatTodo[] : [])
      return
    }
    if (streamEvent.event === 'answer_delta') {
      queueAnswerDelta(String(data.delta || ''))
      return
    }
    if (streamEvent.event === 'answer_reset') {
      clearStreamingText()
      return
    }
    if (streamEvent.event === 'artifact') {
      mergeArtifact(data as unknown as ChatArtifact)
      return
    }
    if (streamEvent.event === 'interrupt') {
      setInterruptState(data as unknown as ChatInterrupt)
      return
    }
    if (streamEvent.event === 'done') {
      receivedDoneRef.current = true
      clearStreamingText()
      applyResponse(data as unknown as ChatTurnResponse)
      return
    }
    if (streamEvent.event === 'error') {
      throw new Error(String(data.message || 'Agent 流式执行失败'))
    }
  }

  async function runStream(start: (onEvent: (event: ChatStreamEvent) => void, signal: AbortSignal) => Promise<void>) {
    const requestToken = requestTokenRef.current + 1
    requestTokenRef.current = requestToken
    receivedDoneRef.current = false
    clearStreamingText()
    const controller = new AbortController()
    controllerRef.current = controller
    setBusy(true)
    let completed = false
    try {
      await start((event) => handleStreamEvent(event, requestToken), controller.signal)
      if (!receivedDoneRef.current && !controller.signal.aborted) throw new Error('流式响应提前结束，请重试')
      completed = receivedDoneRef.current
    } catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '对话请求失败')
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null
        setBusy(false)
      }
      if (!controller.signal.aborted) await refreshHistory()
    }
    return completed
  }

  async function sendMessage() {
    const message = input.trim()
    const images = pendingImages
    if ((!message && !images.length) || busy || uploading || interruptState) return
    setUploading(true)
    setError('')
    const uploaded: ChatAttachment[] = []
    try {
      // 顺序上传便于在中途失败时准确清理已经成功的临时附件。
      for (const image of images) {
        uploaded.push(await api.uploadChatAttachment(threadId, image.file))
      }
    } catch (reason) {
      await Promise.all(uploaded.map((attachment) => api.deleteChatAttachment(attachment.id).catch(() => undefined)))
      setError(reason instanceof Error ? reason.message : '图片上传失败')
      setUploading(false)
      return
    }
    setInput('')
    clearPendingImages()
    setMessages((current) => [...current, {
      id: crypto.randomUUID(), role: 'user', content: message, artifactIds: [], attachments: uploaded,
    }])
    setUploading(false)
    await runStream((onEvent, signal) => api.chatTurnStream(
      threadId, message, uploaded.map((attachment) => attachment.id), onEvent, signal,
    ))
  }

  function addPendingImages(files: File[]) {
    if (!files.length) return
    const current = pendingImagesRef.current
    if (files.some((file) => !SUPPORTED_IMAGE_TYPES.has(file.type))) {
      setError('仅支持 JPEG、PNG 和 WebP 图片。')
      return
    }
    if (files.some((file) => file.size > MAX_IMAGE_BYTES)) {
      setError('单张图片不能超过 8 MiB。')
      return
    }
    if (current.length + files.length > MAX_IMAGES_PER_MESSAGE) {
      setError(`每条消息最多上传 ${MAX_IMAGES_PER_MESSAGE} 张图片。`)
      return
    }
    const totalBytes = current.reduce((sum, image) => sum + image.file.size, 0)
      + files.reduce((sum, file) => sum + file.size, 0)
    if (totalBytes > MAX_IMAGE_TOTAL_BYTES) {
      setError('每条消息的图片总大小不能超过 24 MiB。')
      return
    }
    const next = [...current, ...files.map((file) => ({
      key: crypto.randomUUID(), file, previewUrl: URL.createObjectURL(file),
    }))]
    pendingImagesRef.current = next
    setPendingImages(next)
    setError('')
  }

  function handleImagePaste(event: ReactClipboardEvent<HTMLTextAreaElement>) {
    const itemFiles = Array.from(event.clipboardData.items)
      .filter((item) => item.kind === 'file' && item.type.startsWith('image/'))
      .map((item) => item.getAsFile())
      .filter((file): file is File => file !== null)
    const files = itemFiles.length
      ? itemFiles
      : Array.from(event.clipboardData.files).filter((file) => file.type.startsWith('image/'))
    // 保留浏览器默认粘贴，因此剪贴板同时含文字和图片时，文字仍会进入输入框。
    addPendingImages(files)
  }

  function removePendingImage(key: string) {
    const current = pendingImagesRef.current
    const removed = current.find((image) => image.key === key)
    if (removed) URL.revokeObjectURL(removed.previewUrl)
    const next = current.filter((image) => image.key !== key)
    pendingImagesRef.current = next
    setPendingImages(next)
  }

  function clearPendingImages() {
    pendingImagesRef.current.forEach((image) => URL.revokeObjectURL(image.previewUrl))
    pendingImagesRef.current = []
    setPendingImages([])
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  async function resume(decision: 'answer' | 'approve' | 'edit' | 'reject') {
    if (!interruptState || busy) return
    setError('')
    try {
      const activeInterrupt = interruptState
      let editedAction: Record<string, unknown> | undefined
      if (decision === 'edit') {
        const args = JSON.parse(editText) as Record<string, unknown>
        editedAction = { name: 'create_work_order', args }
      }
      if (decision === 'answer' && resumeText.trim()) {
        setMessages((current) => [...current, { id: crypto.randomUUID(), role: 'user', content: resumeText.trim(), artifactIds: [], attachments: [] }])
      }
      // 用户提交后立即收起确认卡；如果网络或后端失败，再恢复原卡片供重试。
      setInterruptState(null)
      const completed = await runStream((onEvent, signal) => api.chatResumeStream({
        thread_id: threadId,
        kind: activeInterrupt.kind,
        decision,
        message: resumeText,
        edited_action: editedAction,
      }, onEvent, signal))
      if (!completed) setInterruptState(activeInterrupt)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '恢复对话失败')
    }
  }

  function stopStream() {
    controllerRef.current?.abort()
    requestTokenRef.current += 1
    clearStreamingText()
    setBusy(false)
  }

  function resetConversation() {
    controllerRef.current?.abort()
    setThreadId(newThreadId())
    setMessages([{ id: 'welcome', role: 'assistant', content: '已开始新的对话。请告诉我需要查询的对象、时间或报告主题。', artifactIds: [], attachments: [] }])
    setArtifacts({})
    setInterruptState(null)
    setInput('')
    clearPendingImages()
    setError('')
    setBusy(false)
    setTodos([])
    setTodoOpen(false)
    setPanelOpen(false)
    setSelectedArtifactIds([])
    setArtifactView('report')
    clearStreamingText()
    todoWasShownRef.current = false
    seenArtifactIdsRef.current.clear()
    setDeleteConfirmId(null)
  }

  function openArtifactPanel(message: DisplayMessage, type: EvidenceArtifactType, trigger: HTMLButtonElement) {
    artifactTriggerRef.current = trigger
    setSelectedArtifactIds(message.artifactIds)
    setArtifactView(type)
    setPanelOpen(true)
  }

  function closeArtifactPanel() {
    setPanelOpen(false)
    const trigger = artifactTriggerRef.current
    artifactTriggerRef.current = null
    if (trigger) window.requestAnimationFrame(() => trigger.focus())
  }

  function clampSplit(clientX: number) {
    const rect = workspaceRef.current?.getBoundingClientRect()
    if (!rect || rect.width <= 0) return splitPercent
    const minLeft = 400
    const minRight = 480
    const separatorSpace = 28
    const left = Math.max(minLeft, Math.min(clientX - rect.left, rect.width - minRight - separatorSpace))
    return Math.max(25, Math.min(70, left / rect.width * 100))
  }

  function handleSplitterPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    draggingRef.current = true
    event.currentTarget.setPointerCapture(event.pointerId)
    setSplitPercent(clampSplit(event.clientX))
  }

  function handleSplitterPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    if (draggingRef.current) setSplitPercent(clampSplit(event.clientX))
  }

  function handleSplitterKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
    event.preventDefault()
    setSplitPercent((current) => Math.max(25, Math.min(70, current + (event.key === 'ArrowLeft' ? -2 : 2))))
  }

  const selectedArtifacts = selectedArtifactIds.map((id) => artifacts[id]).filter((artifact): artifact is ChatArtifact => !!artifact)
  const reports = selectedArtifacts.map(asReportPayload).filter((item): item is ReportPayload => item !== null)
  const queries = selectedArtifacts.map(asQueryPayload).filter((item): item is QueryPayload => item !== null)
  const workOrders = selectedArtifacts.filter((item) => item.type === 'work_order')
  const panelCounts: Record<EvidenceArtifactType, number> = {
    report: reports.length,
    query_result: queries.length,
    work_order: workOrders.length,
  }

  return (
    <div className={historyOpen ? 'chat-page-shell history-open' : 'chat-page-shell history-collapsed'}>
      <aside className={historyOpen ? 'chat-history-panel open' : 'chat-history-panel'} aria-label="历史对话">
        <header>
          {historyOpen && <div><small>CONVERSATIONS</small><strong>历史对话</strong></div>}
          <button type="button" aria-label={historyOpen ? '收起历史对话' : '展开历史对话'} onClick={() => setHistoryOpen((value) => !value)}>{historyOpen ? '‹' : '›'}</button>
        </header>
        {historyOpen && <>
          <button className="chat-history-new" type="button" onClick={resetConversation}>＋ 新建对话</button>
          <div className="chat-history-list">
            {historyLoading && <p className="chat-history-state">正在读取历史…</p>}
            {!historyLoading && !threads.length && <p className="chat-history-state">还没有历史对话</p>}
            {threads.map((thread) => <article className={thread.thread_id === threadId ? 'active' : ''} key={thread.thread_id}>
              <button className="chat-history-open" type="button" onClick={() => void openHistoryThread(thread.thread_id)}>
                <strong title={thread.title}>{thread.title}</strong>
                <small>{new Date(thread.updated_at).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })}</small>
                {thread.status !== 'completed' && <span className={`thread-status ${thread.status}`}>{thread.status === 'interrupted' ? '待确认' : '执行失败'}</span>}
              </button>
              {deleteConfirmId === thread.thread_id ? <div className="chat-history-confirm"><span>永久删除？</span><button type="button" onClick={() => void deleteHistoryThread(thread.thread_id)}>删除</button><button type="button" onClick={() => setDeleteConfirmId(null)}>取消</button></div> : <button className="chat-history-delete" type="button" aria-label={`删除对话 ${thread.title}`} onClick={() => setDeleteConfirmId(thread.thread_id)}>×</button>}
            </article>)}
          </div>
          {historyError && <p className="chat-history-error" role="alert">{historyError}</p>}
        </>}
      </aside>
      {historyOpen && <button className="chat-history-backdrop" type="button" aria-label="关闭历史对话" onClick={() => setHistoryOpen(false)} />}
      <div className={panelOpen ? 'chat-workspace with-evidence' : 'chat-workspace'} ref={workspaceRef} style={{ '--chat-left': `${splitPercent}%` } as CSSProperties}>
      <section className="chat-main-panel">
        <header className="chat-toolbar">
          <div className="chat-agent-state"><span className={configured ? 'chat-status-dot online' : 'chat-status-dot'} /><strong>{configured === null ? '正在检查 Agent' : configured ? '对话 Agent 已就绪' : '对话 Agent 未配置'}</strong></div>
          <div className="chat-toolbar-actions">
            <button className="button button-ghost chat-history-toolbar-button" type="button" onClick={() => setHistoryOpen((value) => !value)}>{historyOpen ? '收起历史' : '历史对话'}</button>
          </div>
        </header>
        <div className="chat-messages" aria-live="polite">
          {messages.map((message) => {
            const messageArtifacts = message.artifactIds.map((id) => artifacts[id]).filter((artifact): artifact is ChatArtifact => !!artifact)
            const counts: Record<EvidenceArtifactType, number> = {
              report: messageArtifacts.filter((artifact) => artifact.type === 'report').length,
              query_result: messageArtifacts.filter((artifact) => artifact.type === 'query_result').length,
              work_order: messageArtifacts.filter((artifact) => artifact.type === 'work_order').length,
            }
            return <article key={message.id} className={`chat-message ${message.role}`}>
              <span>{message.role === 'assistant' ? 'YH' : 'YOU'}</span>
              <div className="chat-message-bubble">
                <small>{message.role === 'assistant' ? '智能助手' : '当前用户'}</small>
                {!!message.attachments.length && <div className="chat-message-images">{message.attachments.map((attachment) => <img key={attachment.id} src={attachment.preview_url} alt={attachment.name} loading="lazy" />)}</div>}
                {message.role === 'assistant' ? <div className="chat-message-markdown"><ReactMarkdown remarkPlugins={[remarkGfm]} components={{ img: MarkdownImage }}>{message.content}</ReactMarkdown></div> : message.content ? <p>{message.content}</p> : null}
                {message.generator && <em>{generatorLabel(message.generator)}</em>}
                {message.role === 'assistant' && Object.values(counts).some((count) => count > 0) && <div className="chat-artifact-actions" aria-label="本轮结构化产物">
                  {counts.report > 0 && <button type="button" onClick={(event) => openArtifactPanel(message, 'report', event.currentTarget)}>查看报告（{counts.report}）</button>}
                  {counts.query_result > 0 && <button type="button" onClick={(event) => openArtifactPanel(message, 'query_result', event.currentTarget)}>查看 SQL 查询（{counts.query_result}）</button>}
                  {counts.work_order > 0 && <button type="button" onClick={(event) => openArtifactPanel(message, 'work_order', event.currentTarget)}>查看工单（{counts.work_order}）</button>}
                </div>}
              </div>
            </article>
          })}
          {busy && <article className="chat-message assistant pending"><span>YH</span><div><small>智能助手 · 流式生成</small><p>{streamingText || '正在规划并查询数据'}<span className="typing-dots" aria-label="正在生成"><i /><i /><i /></span></p></div></article>}
          {error && <div className="chat-error" role="alert"><strong>请求失败</strong><p>{error}</p></div>}

          {interruptState?.kind === 'clarification' && <section className="chat-hitl-card">
            <small>INFORMATION REQUIRED</small><h3>需要补充信息</h3><p>{interruptState.question}</p>
            {!!interruptState.missing_information?.length && <ul>{interruptState.missing_information.map((item) => <li key={item}>{item}</li>)}</ul>}
            {!!interruptState.suggestions?.length && <div className="chat-suggestions">{interruptState.suggestions.map((item) => <button type="button" key={item} onClick={() => setResumeText(item)}>{item}</button>)}</div>}
            <textarea aria-label="补充信息" value={resumeText} onChange={(event) => setResumeText(event.target.value)} placeholder="补充缺少的范围、阈值或对象…" />
            <button className="button button-primary" type="button" disabled={!resumeText.trim() || busy} onClick={() => void resume('answer')}>继续处理</button>
          </section>}

          {interruptState?.kind === 'work_order_approval' && <section className="chat-hitl-card work-order">
            <small>HUMAN APPROVAL</small><h3>工单等待确认</h3><p>{interruptState.action?.description || '只有批准后才会写入工单数据库。'}</p>
            <textarea className="work-order-json" aria-label="工单内容" value={editText} onChange={(event) => setEditText(event.target.value)} />
            <label><span>拒绝原因（可选）</span><input value={resumeText} onChange={(event) => setResumeText(event.target.value)} /></label>
            <div className="hitl-actions">
              <button className="button button-primary" type="button" disabled={busy} onClick={() => void resume('approve')}>批准并创建</button>
              <button className="button button-ghost" type="button" disabled={busy} onClick={() => void resume('edit')}>按修改内容创建</button>
              <button className="button button-danger" type="button" disabled={busy} onClick={() => void resume('reject')}>拒绝</button>
            </div>
          </section>}
          <div ref={messageEndRef} />
        </div>
        {todos.length > 0 && <section className="chat-todo-panel">
          <button type="button" aria-expanded={todoOpen} onClick={() => setTodoOpen((value) => !value)}><span>执行计划</span><small>{todos.filter((item) => item.status === 'completed').length}/{todos.length} 已完成</small><b>{todoOpen ? '收起' : '展开'}</b></button>
          {todoOpen && <ol>{todos.map((todo, index) => <li className={todo.status} key={`${index}-${todo.content}`}><span>{todo.status === 'completed' ? '✓' : todo.status === 'in_progress' ? '→' : '·'}</span><p>{todo.content}</p></li>)}</ol>}
        </section>}
        <footer className="chat-composer">
          {!!pendingImages.length && <div className="chat-image-preview-list" aria-label="待发送图片">{pendingImages.map((image) => <figure key={image.key}>
            <img src={image.previewUrl} alt={image.file.name} />
            <figcaption title={image.file.name}>{image.file.name}</figcaption>
            <button type="button" aria-label={`移除图片 ${image.file.name}`} onClick={() => removePendingImage(image.key)}>×</button>
          </figure>)}</div>}
          <div className="chat-composer-editor">
            <textarea
              aria-label="对话输入"
              value={input}
              disabled={busy || uploading || !!interruptState}
              placeholder={interruptState ? '请先处理上方确认卡片' : '输入问题，也可以选择或直接粘贴图片…'}
              onChange={(event) => setInput(event.target.value)}
              onPaste={handleImagePaste}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void sendMessage() }
              }}
            />
            <button className="chat-image-picker" type="button" disabled={busy || uploading || !!interruptState || pendingImages.length >= MAX_IMAGES_PER_MESSAGE} onClick={() => fileInputRef.current?.click()}>＋ 添加图片</button>
            <input ref={fileInputRef} className="chat-image-file-input" type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={(event) => { addPendingImages(Array.from(event.target.files || [])); event.target.value = '' }} />
          </div>
          {busy ? <button className="button button-danger" type="button" onClick={stopStream}>停止等待</button> : <button className="button button-primary" type="button" disabled={uploading || (!input.trim() && !pendingImages.length) || !!interruptState} onClick={() => void sendMessage()}>{uploading ? '上传中…' : '发送'}</button>}
          <small>Enter 发送 · Shift+Enter 换行 · 支持 Ctrl+V 粘贴图片</small>
        </footer>
      </section>

      {panelOpen && <div
        className="chat-splitter"
        role="separator"
        aria-label="调整对话区和报告区宽度"
        aria-orientation="vertical"
        aria-valuemin={25}
        aria-valuemax={70}
        aria-valuenow={Math.round(splitPercent)}
        tabIndex={0}
        onPointerDown={handleSplitterPointerDown}
        onPointerMove={handleSplitterPointerMove}
        onPointerUp={(event) => { draggingRef.current = false; event.currentTarget.releasePointerCapture(event.pointerId) }}
        onPointerCancel={() => { draggingRef.current = false }}
        onKeyDown={handleSplitterKeyDown}
      ><span /></div>}

      {panelOpen && <aside
        className="chat-evidence-panel"
        ref={panelRef}
        role={narrowPanel ? 'dialog' : 'region'}
        aria-modal={narrowPanel || undefined}
        aria-label="数据与报告"
      >
        <header className="chat-evidence-header">
          <div><small>TRACEABLE OUTPUT</small><h2>数据与报告</h2><p>仅展示当前回答对应的结构化产物。</p></div>
          <button className="button button-ghost" type="button" ref={panelCloseRef} onClick={closeArtifactPanel}>隐藏</button>
        </header>
        <div className="chat-evidence-tabs" role="tablist" aria-label="产物类型">
          <button role="tab" type="button" aria-selected={artifactView === 'report'} disabled={!panelCounts.report} onClick={() => setArtifactView('report')}>报告 <span>{panelCounts.report}</span></button>
          <button role="tab" type="button" aria-selected={artifactView === 'query_result'} disabled={!panelCounts.query_result} onClick={() => setArtifactView('query_result')}>SQL 查询 <span>{panelCounts.query_result}</span></button>
          <button role="tab" type="button" aria-selected={artifactView === 'work_order'} disabled={!panelCounts.work_order} onClick={() => setArtifactView('work_order')}>工单 <span>{panelCounts.work_order}</span></button>
        </div>
        <div className="chat-evidence-content" role="tabpanel">
          {artifactView === 'work_order' && workOrders.map((artifact) => {
            const payload = artifact.payload
            const checklist = Array.isArray(payload.checklist) ? payload.checklist : []
            const sourceReference = payload.source_reference && typeof payload.source_reference === 'object'
              ? payload.source_reference as Record<string, unknown>
              : null
            return <article className="chat-work-order-result" key={artifact.id}>
              <header>
                <div><small>WORK ORDER</small><strong>{String(payload.work_order_id || artifact.id)}</strong></div>
                <span>{String(payload.status || 'OPEN')}</span>
              </header>
              <h3>{String(payload.title || '工单已创建')}</h3>
              <dl>
                {payload.priority != null && <><dt>优先级</dt><dd>{String(payload.priority)}</dd></>}
                {payload.source_module != null && <><dt>来源模块</dt><dd>{String(payload.source_module)}</dd></>}
                {payload.user_id != null && <><dt>用户编号</dt><dd>{String(payload.user_id)}</dd></>}
                {payload.created_at != null && <><dt>创建时间</dt><dd>{String(payload.created_at)}</dd></>}
              </dl>
              {payload.description != null && <section><h4>工单说明</h4><p>{String(payload.description)}</p></section>}
              {checklist.length > 0 && <section><h4>检查清单</h4><ol>{checklist.map((item, index) => <li key={`${index}-${String(item)}`}>{String(item)}</li>)}</ol></section>}
              {sourceReference && <section><h4>关联证据</h4><pre>{JSON.stringify(sourceReference, null, 2)}</pre></section>}
            </article>
          })}
          {artifactView === 'report' && reports.map((report) => <ReportViewer report={report} key={report.report_id} />)}
          {artifactView === 'query_result' && queries.map((query) => <QueryResultCard query={query} key={query.query_id} />)}
        </div>
      </aside>}
      </div>
    </div>
  )
}
