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
  return (
    <div className="agent-report">
      <div className="report-meta">
        <span>生成器</span>
        <b>{report.generator || '未标明'}</b>
      </div>
      <h4>{report.title || '智能检查结论'}</h4>
      <p>{report.inspection_conclusion || report.conclusion || report.summary || '报告已返回，请结合结构化字段查看。'}</p>
      {evidence.length > 0 && (
        <div className="report-section">
          <strong>证据链</strong>
          <ol>{evidence.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ol>
        </div>
      )}
      {checks.length > 0 && (
        <div className="report-section">
          <strong>现场检查</strong>
          <ul>{checks.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul>
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
