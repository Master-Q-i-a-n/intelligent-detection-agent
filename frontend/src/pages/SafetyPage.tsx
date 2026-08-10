import { useEffect, useState } from 'react'
import { api } from '../api'
import { DataTable, EmptyState, ErrorState, KpiGrid, LoadingState, Panel } from '../components/Common'
import type { SecurityEvent, SecurityOverview } from '../types'

const eventNames: Record<string, string> = {
  PPE_INSPECTION: 'PPE 佩戴异常',
  NO_HELMET: '未佩戴安全帽',
  OVER_COUNT: '作业区超员',
  DWELL: '人员异常滞留',
}

const statusNames: Record<string, string> = {
  NEW: '待确认', ACKNOWLEDGED: '已确认', PROCESSING: '处理中', CLOSED: '已关闭',
}

const ppeStatusNames: Record<string, string> = {
  WORN: '已佩戴',
  NOT_WORN: '未佩戴',
  REMOVED_DURING_WORK: '作业中摘除',
  UNCERTAIN: '无法确认',
  NEEDS_REVIEW: '待复核',
}

const visibilityNames: Record<string, string> = {
  CLEAR: '清晰可见', PARTIAL: '部分可见', NOT_VISIBLE: '不可见',
}

const qualityNames: Record<string, string> = {
  GOOD: '良好', LIMITED: '有限', POOR: '较差',
}

const evidenceNames: Record<string, string> = {
  PPE_ANOMALY_IMAGE: '异常时刻截图',
  REVIEW_VIDEO: '复核视频',
  OVERVIEW_IMAGE: '现场全景',
  PERSON_IMAGE: '人员证据',
}

function formatEvidenceTime(value?: number | null) {
  return value == null ? '—' : `${Number(value).toFixed(2)} 秒`
}

export function SafetyPage({
  refreshToken,
  liveSequence = 0,
  onBusyChange,
  onChanged,
  username,
}: {
  refreshToken: number
  liveSequence?: number
  onBusyChange: (busy: boolean) => void
  onChanged: () => void
  username: string
}) {
  const [overview, setOverview] = useState<SecurityOverview | null>(null)
  const [events, setEvents] = useState<SecurityEvent[]>([])
  const [selected, setSelected] = useState<SecurityEvent | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [decision, setDecision] = useState('')
  const [handlingStatus, setHandlingStatus] = useState('')
  const [eventType, setEventType] = useState('')
  const [operator, setOperator] = useState(username)
  const [comment, setComment] = useState('')
  const [acting, setActing] = useState(false)

  useEffect(() => {
    // 登录用户变化时同步默认操作人，用户仍可在事件处置前手动修改。
    setOperator(username)
  }, [username])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError('')
    onBusyChange(true)
    Promise.all([
      api.securityOverview(controller.signal),
      api.securityEvents({ decision, handling_status: handlingStatus, event_type: eventType }, controller.signal),
    ])
      .then(([nextOverview, eventResult]) => {
        setOverview(nextOverview)
        setEvents(eventResult.items)
        if (selected) {
          const remains = eventResult.items.some((item) => item.event_id === selected.event_id)
          if (!remains) {
            setSelected(null)
          } else {
            // 刷新列表时同步详情，避免处置状态或复核说明停留在旧版本。
            api.securityEvent(selected.event_id, controller.signal).then(setSelected).catch(() => undefined)
          }
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '未知错误')
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false)
          onBusyChange(false)
        }
      })
    return () => {
      controller.abort()
      onBusyChange(false)
    }
  }, [decision, eventType, handlingStatus, refreshToken, liveSequence, onBusyChange])

  async function openEvent(eventId: string) {
    setError('')
    try {
      setSelected(await api.securityEvent(eventId))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '事件详情读取失败')
    }
  }

  async function performAction(action: 'ACKNOWLEDGE' | 'START_PROCESSING' | 'CLOSE') {
    if (!selected) return
    setActing(true)
    setError('')
    try {
      await api.securityAction(selected.event_id, action, operator.trim() || username, comment)
      setSelected(await api.securityEvent(selected.event_id))
      const [nextOverview, nextEvents] = await Promise.all([
        api.securityOverview(),
        api.securityEvents({ decision, handling_status: handlingStatus, event_type: eventType }),
      ])
      setOverview(nextOverview)
      setEvents(nextEvents.items)
      setComment('')
      onChanged()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '处置状态更新失败')
    } finally {
      setActing(false)
    }
  }

  if (loading && !overview) return <LoadingState label="正在接收安全作业事件" />
  if (error && !overview) return <ErrorState message={error} />
  if (!overview) return <EmptyState title="安防数据链路未就绪" detail="检查安全作业 SQLite 是否已完成迁移。" />

  return (
    <div className="page-stack safety-page">
      {error && <div className="inline-error" role="alert">{error}</div>}
      <div className="section-heading safety-heading">
        <div><small>VISION · REVIEW · RESPONSE</small><h2>安全作业事件闭环</h2></div>
        <span className="safety-live"><i />视频规则与多模态复核已接入</span>
      </div>
      <KpiGrid items={[
        { label: '已确认安防事件', value: overview.confirmed, note: '多模态复核确认', tone: 'red' },
        { label: '待人工复核', value: overview.review_required, note: '证据不足，不触发正式告警', tone: 'amber' },
        { label: '待确认处置', value: overview.new_count, note: '尚未由登录用户确认', tone: 'cyan' },
        { label: '处理中', value: overview.processing, note: `未关闭高风险 ${overview.high_risk}`, tone: 'blue' },
      ]} />

      <Panel title="事件作业台" eyebrow="规则证据、模型复核与人工处置保持同一事件编号" className="table-panel safety-table-panel">
        <div className="table-tools safety-filters">
          <select value={decision} onChange={(event) => setDecision(event.target.value)} aria-label="按复核结论筛选">
            <option value="">全部结论</option><option value="CONFIRMED">已确认问题</option><option value="UNCERTAIN">待人工复核</option>
          </select>
          <select value={handlingStatus} onChange={(event) => setHandlingStatus(event.target.value)} aria-label="按处置状态筛选">
            <option value="">全部处置状态</option><option value="NEW">待确认</option><option value="ACKNOWLEDGED">已确认</option><option value="PROCESSING">处理中</option><option value="CLOSED">已关闭</option>
          </select>
          <select value={eventType} onChange={(event) => setEventType(event.target.value)} aria-label="按事件类型筛选">
            <option value="">全部事件类型</option><option value="PPE_INSPECTION">PPE 佩戴异常</option><option value="NO_HELMET">未佩戴安全帽</option><option value="OVER_COUNT">作业区超员</option><option value="DWELL">人员异常滞留</option>
          </select>
        </div>
        <DataTable minWidth={880}>
          <thead><tr><th>结论</th><th>事件</th><th>摄像头 / 区域</th><th>视频位置</th><th>复核说明</th><th>处置状态</th><th>操作</th></tr></thead>
          <tbody>{events.length ? events.map((event) => (
            <tr key={event.event_id} className={selected?.event_id === event.event_id ? 'selected-row' : ''}>
              <td><span className={`decision-badge ${event.final_decision.toLowerCase()}`}>{event.final_decision === 'CONFIRMED' ? '确认问题' : '需要复核'}</span></td>
              <td><b>{eventNames[event.event_type] || event.event_type}</b><small className="cell-sub">Track {event.primary_track_id ?? '—'}</small></td>
              <td>{event.camera_id}<small className="cell-sub">{event.zone_id || '未配置区域'}</small></td>
              <td>{event.occurred_at || `视频第 ${Number(event.activated_video_seconds || 0).toFixed(1)} 秒`}</td>
              <td className="evidence-cell" title={event.latest_review_explanation || event.final_reason || ''}>{event.latest_review_explanation || event.final_reason || '—'}</td>
              <td><span className={`handling-status status-${event.handling_status.toLowerCase()}`}>{statusNames[event.handling_status] || event.handling_status}</span></td>
              <td><button type="button" className="text-button" onClick={() => openEvent(event.event_id)}>打开事件 →</button></td>
            </tr>
          )) : <tr><td colSpan={7}><div className="table-empty">当前筛选条件下没有已送达 Agent 的安防事件</div></td></tr>}</tbody>
        </DataTable>
      </Panel>

      {selected && <SafetyEventDetail event={selected} operator={operator} comment={comment} acting={acting} onOperatorChange={setOperator} onCommentChange={setComment} onAction={performAction} />}
    </div>
  )
}

function SafetyEventDetail({
  event, operator, comment, acting, onOperatorChange, onCommentChange, onAction,
}: {
  event: SecurityEvent
  operator: string
  comment: string
  acting: boolean
  onOperatorChange: (value: string) => void
  onCommentChange: (value: string) => void
  onAction: (action: 'ACKNOWLEDGE' | 'START_PROCESSING' | 'CLOSE') => void
}) {
  const images = (event.evidence || []).filter((item) => item.mime_type?.startsWith('image/'))
  const videos = (event.evidence || []).filter((item) => item.mime_type?.startsWith('video/'))
  // 联合 PPE 事件只展示一张异常锚点主图；旧事件仍保留原有多图证据展示。
  const displayImages = event.event_type === 'PPE_INSPECTION'
    ? images.filter((item) => item.evidence_type === 'PPE_ANOMALY_IMAGE').slice(0, 1)
    : images
  const rulePeople = event.yolo_rule_metrics?.people || []
  const visualPeople = event.latest_review_ppe_results || []
  const evidenceUrl = (id: number) => `/security/events/${encodeURIComponent(event.event_id)}/evidence/${id}`
  return (
    <Panel title="事件证据与处置" eyebrow={`${event.event_id} · ${event.source_system}`} className="safety-detail">
      <div className="incident-ribbon">
        <div><small>FINAL DECISION</small><strong>{event.final_decision === 'CONFIRMED' ? '复核确认存在安全问题' : '证据不足，等待人工复核'}</strong></div>
        <span>{eventNames[event.event_type] || event.event_type}</span>
      </div>
      <div className="safety-detail-grid">
        <div className={`evidence-gallery ${event.event_type === 'PPE_INSPECTION' ? 'ppe-evidence-gallery' : ''}`}>
          {displayImages.map((item) => <figure key={item.evidence_id}><img src={evidenceUrl(item.evidence_id)} alt={evidenceNames[item.evidence_type] || item.evidence_type} /><figcaption>{evidenceNames[item.evidence_type] || item.evidence_type}</figcaption></figure>)}
          {videos.map((item) => <SafetyVideoEvidence key={item.evidence_id} src={evidenceUrl(item.evidence_id)} label={evidenceNames[item.evidence_type] || item.evidence_type} />)}
          {!displayImages.length && !videos.length && <div className="panel-empty">该事件没有可用图片或视频证据</div>}
        </div>
        <div className="incident-analysis">
          {!!event.people?.length && <div className="ppe-people"><small>人员 PPE 结论</small><div className="ppe-status-grid">
            {event.people.map((person) => <article key={person.track_id}>
              <strong>Track {person.track_id}</strong>
              <span>安全帽 <b data-status={person.helmet_status || ''}>{ppeStatusNames[person.helmet_status || ''] || person.helmet_status || '—'}</b></span>
              <span>手套 <b data-status={person.gloves_status || ''}>{ppeStatusNames[person.gloves_status || ''] || person.gloves_status || '—'}</b></span>
              <span>护目镜 <b data-status={person.goggles_status || ''}>{ppeStatusNames[person.goggles_status || ''] || person.goggles_status || '—'}</b></span>
            </article>)}
          </div></div>}
          <div className="source-analysis yolo-analysis"><small>YOLO 规则检测结果</small>
            {event.event_type === 'PPE_INSPECTION' && rulePeople.length ? <div className="source-result-list">
              {rulePeople.map((person) => <article key={person.track_id}>
                <strong>Track {person.track_id}</strong>
                <p>安全帽：<b data-status={person.helmet_rule_status || ''}>{ppeStatusNames[person.helmet_rule_status || ''] || person.helmet_rule_status || '—'}</b>；最后佩戴 {formatEvidenceTime(person.helmet_last_worn_at)}，首次明确未佩戴 {formatEvidenceTime(person.first_no_helmet_at)}。</p>
                <p>手套：正向 {person.gloves_positive_frames ?? 0} 帧 / {Number(person.gloves_effective_seconds || 0).toFixed(2)} 秒，反向提示 {person.no_gloves_positive_frames ?? 0} 帧；{person.gloves_rule_status === 'WORN' ? '达到阈值，规则确认已佩戴' : '未达到正向确认阈值，交由视觉复核'}。</p>
                <p>护目镜：正向 {person.goggles_positive_frames ?? 0} 帧 / {Number(person.goggles_effective_seconds || 0).toFixed(2)} 秒，反向提示 {person.no_goggle_positive_frames ?? 0} 帧；{person.goggles_rule_status === 'WORN' ? '达到阈值，规则确认已佩戴' : '未达到正向确认阈值，交由视觉复核'}。</p>
              </article>)}
            </div> : <p>{event.final_reason || '未提供规则检测说明'}</p>}
          </div>
          <div className="source-analysis doubao-analysis"><small>豆包视觉复核</small>
            {event.event_type === 'PPE_INSPECTION' && visualPeople.length ? <div className="source-result-list">
              {visualPeople.map((person) => <article key={person.track_id}>
                <strong>Track {person.track_id}</strong>
                <p>{person.visual_reason || '未返回该人员的视觉原因。'}</p>
                <span>可见性：{visibilityNames[person.visibility || ''] || person.visibility || '—'} · 证据质量：{qualityNames[person.evidence_quality || ''] || person.evidence_quality || '—'}</span>
              </article>)}
            </div> : <p>{event.latest_review_explanation || '未返回视觉复核说明'}</p>}
          </div>
          <div><small>建议处置</small><p>{event.recommended_action || '结合现场情况核验并记录结果。'}</p></div>
          <div className="handling-form">
            <label><span>操作人</span><input value={operator} disabled={acting || event.handling_status === 'CLOSED'} onChange={(e) => onOperatorChange(e.target.value)} /></label>
            <label><span>处置备注</span><textarea rows={3} value={comment} disabled={acting || event.handling_status === 'CLOSED'} onChange={(e) => onCommentChange(e.target.value)} placeholder="记录现场核验、纠正措施或关闭原因" /></label>
            <div className="handling-actions">
              {event.handling_status === 'NEW' && <button className="button button-primary" disabled={acting} onClick={() => onAction('ACKNOWLEDGE')}>确认收到</button>}
              {event.handling_status === 'ACKNOWLEDGED' && <button className="button button-primary" disabled={acting} onClick={() => onAction('START_PROCESSING')}>开始处理</button>}
              {event.handling_status !== 'CLOSED' && <button className="button button-ghost" disabled={acting} onClick={() => onAction('CLOSE')}>关闭事件</button>}
              {event.handling_status === 'CLOSED' && <span className="closed-label">事件已闭环</span>}
            </div>
          </div>
        </div>
      </div>
      {!!event.actions?.length && <div className="action-audit"><strong>处置审计</strong>{event.actions.map((action) => <div key={action.action_id}><span>{action.acted_at}</span><b>{action.operator}</b><p>{action.previous_status} → {action.new_status}{action.comment ? ` · ${action.comment}` : ''}</p></div>)}</div>}
    </Panel>
  )
}

function SafetyVideoEvidence({ src, label }: { src: string; label: string }) {
  const [unsupported, setUnsupported] = useState(false)

  return (
    <figure className="review-video">
      {unsupported ? (
        <div className="video-fallback" role="status">
          <strong>浏览器无法播放该证据编码</strong>
          <span>原始视频仍已保留，可下载后用本地播放器查看。</span>
          <a className="button button-ghost" href={src} download>下载证据视频</a>
        </div>
      ) : (
        <video src={src} controls autoPlay loop muted playsInline preload="metadata" onError={() => setUnsupported(true)} />
      )}
      <figcaption>{label}</figcaption>
    </figure>
  )
}
