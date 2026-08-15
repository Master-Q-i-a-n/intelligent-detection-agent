import type { ReactNode } from 'react'
import type { AgentReport } from '../types'

export function Panel({
  title,
  eyebrow,
  aside,
  className = '',
  children,
}: {
  title: string
  eyebrow?: string
  aside?: ReactNode
  className?: string
  children: ReactNode
}) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel-head">
        <div>
          <h3>{title}</h3>
          {eyebrow && <p>{eyebrow}</p>}
        </div>
        {aside && <div className="panel-aside">{aside}</div>}
      </header>
      {children}
    </section>
  )
}

export function KpiGrid({
  items,
}: {
  items: Array<{ label: string; value: string | number; note?: string; tone?: 'cyan' | 'blue' | 'amber' | 'red' | 'green' }>
}) {
  return (
    <div className="kpi-grid">
      {items.map((item) => (
        <article className={`kpi-card tone-${item.tone ?? 'cyan'}`} key={item.label}>
          <span>{item.label}</span>
          <strong title={String(item.value)}>{item.value}</strong>
          <small>{item.note || '\u00a0'}</small>
          <i aria-hidden="true" />
        </article>
      ))}
    </div>
  )
}

export function LoadingState({ label = '正在读取检测数据' }: { label?: string }) {
  return (
    <div className="state-card" role="status">
      <span className="scanner" aria-hidden="true" />
      <strong>{label}</strong>
      <small>数据链路处理中，请稍候</small>
    </div>
  )
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state-card state-error" role="alert">
      <span className="state-code">ERR</span>
      <strong>数据读取失败</strong>
      <small>{message}</small>
      {onRetry && (
        <button className="button button-ghost" type="button" onClick={onRetry}>
          重新读取
        </button>
      )}
    </div>
  )
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="state-card" role="status">
      <span className="state-code">Ø</span>
      <strong>{title}</strong>
      <small>{detail}</small>
    </div>
  )
}

export function PercentBars({
  items,
}: {
  items: Array<{ label: string; value: number; display?: string; tone?: string }>
}) {
  return (
    <div className="percent-bars">
      {items.map((item) => (
        <div className="percent-row" key={item.label}>
          <span title={item.label}>{item.label}</span>
          <div className="percent-track">
            <i className={item.tone ?? ''} style={{ width: `${Math.max(0, Math.min(100, item.value))}%` }} />
          </div>
          <output>{item.display ?? item.value.toFixed(1)}</output>
        </div>
      ))}
    </div>
  )
}

function reportList(report: AgentReport, keys: string[]): string[] {
  for (const key of keys) {
    const value = report[key]
    if (Array.isArray(value)) {
      return value.map((item) => (typeof item === 'string' ? item : JSON.stringify(item)))
    }
  }
  return []
}

export function AgentReportView({ report }: { report: AgentReport }) {
  const evidence = reportList(report, ['evidence_chain', 'evidence'])
  const checks = reportList(report, ['checklist', 'field_checklist', 'recommendations'])
  const evidenceItems = Array.isArray(report.evidence_items) ? report.evidence_items : []
  const causes = Array.isArray(report.possible_causes) ? report.possible_causes : []
  const confidence = typeof report.analysis_confidence === 'number'
    ? Math.max(0, Math.min(1, report.analysis_confidence))
    : null
  const routeLabels: Record<string, string> = {
    data_quality: '数据质量路径', normal: '常规监测路径', single_anomaly: '单项异常路径', high_risk: '高风险路径',
    uncertainty: '低置信复核路径', recovery: '恢复观察路径', critical: '严重退化路径', degradation: '性能衰减路径', stable: '稳定运行路径',
  }
  return (
    <div className="agent-report">
      <div className="report-meta">
        <span>{report.workflow_route ? routeLabels[report.workflow_route] || report.workflow_route : '结构化解读'}</span>
        {report.cache_hit && <em>已复用相同诊断</em>}
      </div>
      <h4>{report.title || '智能检查结论'}</h4>
      <p>{report.inspection_conclusion || report.conclusion || report.summary || '报告已返回，请结合结构化字段查看。'}</p>
      {(report.risk_level || report.decision || confidence !== null) && (
        <div className="report-decision-strip">
          <div><small>算法风险</small><strong>{report.risk_level || '待判定'}{typeof report.risk_score === 'number' ? ` · ${report.risk_score} 分` : ''}</strong></div>
          <div><small>处置级别</small><strong>{report.decision || '结合现场复核'}</strong></div>
          <div className="confidence-cell"><small>分析置信度</small><strong>{confidence === null ? '—' : `${Math.round(confidence * 100)}%`}</strong>{confidence !== null && <i><span style={{ width: `${confidence * 100}%` }} /></i>}</div>
        </div>
      )}
      {causes.length > 0 && (
        <div className="report-section cause-section">
          <strong>原因排序</strong>
          <div className="cause-list">{causes.map((cause) => (
            <article className="cause-card" key={`${cause.rank}-${cause.cause}`}>
              <span className="cause-rank">{String(cause.rank).padStart(2, '0')}</span>
              <div><h5>{cause.cause}</h5><small>可信度 {Math.round(cause.confidence * 100)}%</small>
                {!!cause.supporting_evidence_ids.length && <p>支持证据 {cause.supporting_evidence_ids.join(' · ')}</p>}
                {!!cause.counter_evidence_ids.length && <p className="counter-ref">反向证据 {cause.counter_evidence_ids.join(' · ')}</p>}
              </div>
            </article>
          ))}</div>
        </div>
      )}
      {evidenceItems.length > 0 ? (
        <div className="report-section evidence-ledger">
          <strong>证据台账</strong>
          <div>{evidenceItems.map((item) => (
            <article key={item.evidence_id}>
              <div><b>{item.category}</b><p>{item.statement}</p></div>
            </article>
          ))}</div>
        </div>
      ) : evidence.length > 0 && (
        <div className="report-section">
          <strong>证据链</strong>
          <ol>{evidence.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ol>
        </div>
      )}
      {((report.counter_evidence?.length || 0) > 0 || (report.missing_evidence?.length || 0) > 0) && (
        <div className="report-evidence-balance">
          <section><strong>反证与边界</strong>{report.counter_evidence?.length ? <ul>{report.counter_evidence.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul> : <p>当前没有形成明确反证。</p>}</section>
          <section><strong>仍缺少的证据</strong>{report.missing_evidence?.length ? <ul>{report.missing_evidence.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul> : <p>现有结构化证据已覆盖当前分析路径。</p>}</section>
        </div>
      )}
      {checks.length > 0 && (
        <div className="report-section">
          <strong>现场核查顺序</strong>
          <ol className="check-sequence">{checks.map((item, index) => <li key={`${index}-${item}`}><span>{index + 1}</span>{item}</li>)}</ol>
        </div>
      )}
      {(report.work_order_advice || report.work_order_suggestion) && (
        <div className="report-section"><strong>工单建议</strong><p>{report.work_order_advice || report.work_order_suggestion}</p></div>
      )}
      {!evidence.length && !checks.length && (
        <details>
          <summary>查看完整结构化结果</summary>
          <pre>{JSON.stringify(report, null, 2)}</pre>
        </details>
      )}
      {report.llm_notice && <p className="report-notice">{report.llm_notice}</p>}
    </div>
  )
}

export function DataTable({ children, minWidth = 860 }: { children: ReactNode; minWidth?: number }) {
  return (
    <div className="table-scroll">
      <table style={{ minWidth }}>{children}</table>
    </div>
  )
}

export function formatNumber(value: unknown, digits = 1): string {
  const number = Number(value)
  if (!Number.isFinite(number)) return '—'
  return number.toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

export function clampPercent(value: unknown): number {
  const number = Number(value)
  return Number.isFinite(number) ? Math.max(0, Math.min(100, number)) : 0
}
