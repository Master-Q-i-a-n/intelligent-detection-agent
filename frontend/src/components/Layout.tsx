import type { PageKey, UserSummary } from '../types'

const navItems: Array<{ key: PageKey; index: string; label: string; english: string }> = [
  { key: 'overview', index: '00', label: '每日巡检总览', english: 'Daily Inspection' },
  { key: 'metering', index: '01', label: '智能计量详情', english: 'Metering Evidence' },
  { key: 'equipment', index: '02', label: '智能设备详情', english: 'Equipment Evidence' },
  { key: 'safety', index: '03', label: '安全作业', english: 'Safety Operations' },
]

export function Sidebar({
  page,
  onPageChange,
  autoAgentEnabled,
  onAutoAgentChange,
  safetyBadge = 0,
}: {
  page: PageKey
  onPageChange: (page: PageKey) => void
  autoAgentEnabled: boolean
  onAutoAgentChange: (enabled: boolean) => void
  safetyBadge?: number
}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">YH</span>
        <div>
          <strong>曜衡智控</strong>
          <small>GAS AI INSPECTION</small>
        </div>
      </div>
      <p className="nav-label">智能检测中心</p>
      <nav aria-label="业务模块">
        {navItems.map((item) => (
          <button
            type="button"
            key={item.key}
            className={page === item.key ? 'nav-item active' : 'nav-item'}
            aria-current={page === item.key ? 'page' : undefined}
            onClick={() => onPageChange(item.key)}
          >
            <span>{item.index}</span>
            <div>
              <b>{item.label}</b>
              <small>{item.english}</small>
            </div>
            {item.key === 'safety' && safetyBadge > 0 && <em className="nav-badge">{safetyBadge > 99 ? '99+' : safetyBadge}</em>}
          </button>
        ))}
      </nav>

      <section className={autoAgentEnabled ? 'agent-switch-card enabled' : 'agent-switch-card'} aria-label="Agent 自动调用设置">
        <div className="circuit-title">
          <span className="circuit-light" aria-hidden="true" />
          <div>
            <small>AI CONTROL LOOP</small>
            <strong>Agent 自动解读</strong>
          </div>
        </div>
        <button
          className="switch"
          type="button"
          role="switch"
          aria-checked={autoAgentEnabled}
          aria-label="Agent 自动解读"
          onClick={() => onAutoAgentChange(!autoAgentEnabled)}
        >
          <span />
        </button>
        <p>自动解读：<b>{autoAgentEnabled ? '已开启' : '已关闭'}</b></p>
        <small>{page === 'safety' ? '安防复核独立运行，本开关不影响安全告警' : autoAgentEnabled ? '企业、日期或模块变化时调用一次 LLM' : '当前不主动调用 LLM，手动生成仍可使用'}</small>
      </section>

      <div className="system-status">
        <span className="status-dot" />
        <div>
          <b>系统运行正常</b>
          <small>算法服务 / 数据库 / Agent</small>
        </div>
      </div>
    </aside>
  )
}

export function Topbar({
  page,
  users,
  userId,
  date,
  dates,
  busy,
  onUserChange,
  onDateChange,
  onRefresh,
}: {
  page: PageKey
  users: UserSummary[]
  userId: string
  date: string
  dates: string[]
  busy: boolean
  onUserChange: (userId: string) => void
  onDateChange: (date: string) => void
  onRefresh: () => void
}) {
  const titles = {
    overview: ['每日全量自诊断', 'DAILY AUTONOMOUS INSPECTION'],
    metering: ['智能计量证据中心', 'METERING DIAGNOSTIC EVIDENCE'],
    equipment: ['智能设备健康中心', 'EQUIPMENT HEALTH EVIDENCE'],
    safety: ['安全作业事件中心', 'SAFETY OPERATIONS CENTER'],
  }
  return (
    <header className="topbar">
      <div className="page-title">
        <small>{titles[page][1]}</small>
        <h1>{titles[page][0]}</h1>
      </div>
      <div className="filters">
        {page !== 'overview' && page !== 'safety' && (
          <label>
            <span>检测企业</span>
            <select value={userId} onChange={(event) => onUserChange(event.target.value)} disabled={!users.length}>
              {users.map((user) => (
                <option value={user.user_id} key={user.user_id}>
                  {user.company_name} · {user.user_id}
                </option>
              ))}
            </select>
          </label>
        )}
        {page !== 'safety' && <label>
          <span>检测日期</span>
          <select value={date} onChange={(event) => onDateChange(event.target.value)} disabled={!dates.length}>
            {!dates.length && <option value="">无可用日期</option>}
            {dates.map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </label>}
        <button className="button button-primary" type="button" disabled={busy || (page !== 'safety' && !date)} onClick={onRefresh}>
          {busy ? '数据处理中…' : page === 'overview' ? '执行每日全量诊断' : page === 'safety' ? '刷新安防事件' : '刷新诊断证据'}
        </button>
      </div>
    </header>
  )
}
