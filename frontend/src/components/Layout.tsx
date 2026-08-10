import { useEffect, useRef, useState } from 'react'
import type { AuthUser, PageKey, UserSummary } from '../types'

const navItems: Array<{ key: PageKey; label: string; english: string }> = [
  { key: 'overview', label: '每日巡检总览', english: 'Daily Inspection' },
  { key: 'metering', label: '智能计量详情', english: 'Metering Evidence' },
  { key: 'equipment', label: '智能设备详情', english: 'Equipment Evidence' },
  { key: 'safety', label: '安全作业', english: 'Safety Operations' },
  { key: 'chat', label: '智能问答', english: 'AI Assistant' },
]

function NavIcon({ page }: { page: PageKey }) {
  // 图标均使用 currentColor，激活态自动继承侧栏的仪表青色。
  if (page === 'overview') return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z" /></svg>
  if (page === 'metering') return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 19a8 8 0 1 1 14 0M12 12l4-3M8 19h8" /><circle cx="12" cy="12" r="1.3" /></svg>
  if (page === 'equipment') return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9.6 3.5h4.8l.7 2.2 2 .8 2.1-1 2.4 4.1-1.7 1.4v2.1l1.7 1.4-2.4 4.1-2.1-1-2 .8-.7 2.2H9.6l-.7-2.2-2-.8-2.1 1-2.4-4.1 1.7-1.4V11L2.4 9.6l2.4-4.1 2.1 1 2-.8z" /><circle cx="12" cy="12" r="3" /></svg>
  if (page === 'safety') return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l8 3v5c0 5.1-3.2 8.4-8 10-4.8-1.6-8-4.9-8-10V6zM8.5 12l2.2 2.2 4.8-5" /></svg>
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h16v11H9l-5 4zM8 9h8M8 12h5" /></svg>
}

export function Sidebar({
  page,
  onPageChange,
  autoAgentEnabled,
  onAutoAgentChange,
  safetyBadge = 0,
  currentUser,
  onLogout,
}: {
  page: PageKey
  onPageChange: (page: PageKey) => void
  autoAgentEnabled: boolean
  onAutoAgentChange: (enabled: boolean) => void
  safetyBadge?: number
  currentUser: AuthUser | null
  onLogout: () => void
}) {
  const [accountMenuOpen, setAccountMenuOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const accountAreaRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!accountMenuOpen && !settingsOpen) return
    const closeOnOutside = (event: PointerEvent) => {
      if (accountMenuOpen && !accountAreaRef.current?.contains(event.target as Node)) setAccountMenuOpen(false)
    }
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setAccountMenuOpen(false)
        setSettingsOpen(false)
      }
    }
    document.addEventListener('pointerdown', closeOnOutside)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('pointerdown', closeOnOutside)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [accountMenuOpen, settingsOpen])

  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">YH</span>
        <div>
          <strong>曜衡智控</strong>
          <small>GAS AI INSPECTION</small>
        </div>
      </div>
      <nav aria-label="业务模块">
        {navItems.map((item) => (
          <button
            type="button"
            disabled={!currentUser}
            key={item.key}
            className={page === item.key ? 'nav-item active' : 'nav-item'}
            aria-current={page === item.key ? 'page' : undefined}
            onClick={() => onPageChange(item.key)}
          >
            <span className="nav-icon"><NavIcon page={item.key} /></span>
            <div>
              <b>{item.label}</b>
              <small>{item.english}</small>
            </div>
            {item.key === 'safety' && safetyBadge > 0 && <em className="nav-badge">{safetyBadge > 99 ? '99+' : safetyBadge}</em>}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="system-status">
          <span className="status-dot" />
          <div>
            <b>系统运行正常</b>
            <small>算法服务 / 数据库 / Agent</small>
          </div>
        </div>

        <div className="account-area" ref={accountAreaRef}>
          {currentUser && accountMenuOpen && <div className="account-menu" role="menu" aria-label="账户菜单">
            <button type="button" role="menuitem" onClick={() => { setAccountMenuOpen(false); setSettingsOpen(true) }}>
              <span aria-hidden="true">⚙</span><b>系统设置</b>
            </button>
            <button className="logout" type="button" role="menuitem" onClick={() => { setAccountMenuOpen(false); onLogout() }}>
              <span aria-hidden="true">↪</span><b>退出登录</b>
            </button>
          </div>}
          <section className={currentUser ? 'account-card signed-in' : 'account-card'} aria-label="当前账号">
            <span>{currentUser ? currentUser.username.slice(0, 2).toUpperCase() : 'ID'}</span>
            <div><small>{currentUser ? 'SIGNED IN' : 'ACCOUNT REQUIRED'}</small><strong>{currentUser?.username || '尚未登录'}</strong></div>
            {currentUser ? <button className="account-more" type="button" aria-label="打开账户菜单" aria-haspopup="menu" aria-expanded={accountMenuOpen} onClick={() => setAccountMenuOpen((value) => !value)}>•••</button> : <b>登录 / 注册</b>}
          </section>
        </div>
      </div>

      {settingsOpen && <div className="settings-backdrop" role="presentation" onPointerDown={() => setSettingsOpen(false)}>
        <section className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settings-title" onPointerDown={(event) => event.stopPropagation()}>
          <header><div><small>PLATFORM CONTROL</small><h2 id="settings-title">系统设置</h2></div><button type="button" aria-label="关闭系统设置" onClick={() => setSettingsOpen(false)}>×</button></header>
          <div className="settings-content">
            <div className="settings-row">
              <div><strong>Agent 自动解读</strong><p>开启后，企业、日期或模块变化时自动调用一次 LLM。</p></div>
              <button className="settings-switch" type="button" role="switch" aria-checked={autoAgentEnabled} aria-label="Agent 自动解读" onClick={() => onAutoAgentChange(!autoAgentEnabled)}><span /></button>
            </div>
            <p className="settings-note"><span className={autoAgentEnabled ? 'enabled' : ''} />当前状态：{autoAgentEnabled ? '已开启' : '已关闭'}。刷新页面后默认恢复关闭。</p>
          </div>
        </section>
      </div>}
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
    chat: ['燃气业务智能问答', 'GAS BUSINESS AI ASSISTANT'],
  }
  return (
    <header className="topbar">
      <div className="page-title">
        <small>{titles[page][1]}</small>
        <h1>{titles[page][0]}</h1>
      </div>
      <div className="filters">
        {page !== 'overview' && page !== 'safety' && page !== 'chat' && (
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
        {page !== 'safety' && page !== 'chat' && <label>
          <span>检测日期</span>
          <select value={date} onChange={(event) => onDateChange(event.target.value)} disabled={!dates.length}>
            {!dates.length && <option value="">无可用日期</option>}
            {dates.map((item) => <option value={item} key={item}>{item}</option>)}
          </select>
        </label>}
        {page !== 'chat' && (
          <button className="button button-primary" type="button" disabled={busy || (page !== 'safety' && !date)} onClick={onRefresh}>
            {busy ? '数据处理中…' : page === 'overview' ? '执行每日全量诊断' : page === 'safety' ? '刷新安防事件' : '刷新诊断证据'}
          </button>
        )}
      </div>
    </header>
  )
}
