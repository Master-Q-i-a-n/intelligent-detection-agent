import { useEffect, useId, useMemo, useRef, useState } from 'react'
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

interface SearchableOption {
  value: string
  label: string
  secondary?: string
}

function SearchableCombobox({
  options,
  value,
  optionName,
  placeholder,
  onChange,
}: {
  options: SearchableOption[]
  value: string
  optionName: string
  placeholder: string
  onChange: (value: string) => void
}) {
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const listboxId = useId()
  const selectedOption = useMemo(() => options.find((option) => option.value === value), [options, value])
  const selectedLabel = selectedOption
    ? [selectedOption.label, selectedOption.secondary].filter(Boolean).join(' · ')
    : ''
  const [query, setQuery] = useState(selectedLabel)
  const [open, setOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(-1)
  const normalizedQuery = query.trim().toLocaleLowerCase()
  const filteredOptions = useMemo(() => options.filter((option) => {
    if (!normalizedQuery) return true
    return `${option.label} ${option.secondary || ''}`.toLocaleLowerCase().includes(normalizedQuery)
  }), [normalizedQuery, options])

  useEffect(() => {
    setQuery(selectedLabel)
  }, [selectedLabel])

  useEffect(() => {
    if (!open) return
    const closeOnOutside = (event: PointerEvent) => {
      if (!wrapperRef.current?.contains(event.target as Node)) {
        setOpen(false)
        setActiveIndex(-1)
        setQuery(selectedLabel)
      }
    }
    document.addEventListener('pointerdown', closeOnOutside)
    return () => document.removeEventListener('pointerdown', closeOnOutside)
  }, [open, selectedLabel])

  function selectOption(option: SearchableOption) {
    setQuery([option.label, option.secondary].filter(Boolean).join(' · '))
    setOpen(false)
    setActiveIndex(-1)
    if (option.value !== value) onChange(option.value)
  }

  function openAllOptions() {
    setQuery('')
    setOpen(true)
    const selectedIndex = options.findIndex((option) => option.value === value)
    setActiveIndex(selectedIndex >= 0 ? selectedIndex : (options.length ? 0 : -1))
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      if (!open) {
        openAllOptions()
        return
      }
      if (!filteredOptions.length) return
      const offset = event.key === 'ArrowDown' ? 1 : -1
      setActiveIndex((current) => {
        const start = current >= 0 ? current : (offset > 0 ? -1 : 0)
        return (start + offset + filteredOptions.length) % filteredOptions.length
      })
      return
    }
    if (event.key === 'Enter' && open && activeIndex >= 0 && filteredOptions[activeIndex]) {
      event.preventDefault()
      selectOption(filteredOptions[activeIndex])
      return
    }
    if (event.key === 'Escape' && open) {
      event.preventDefault()
      setOpen(false)
      setActiveIndex(-1)
      setQuery(selectedLabel)
    }
  }

  return (
    <div
      className="filter-combobox"
      ref={wrapperRef}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setOpen(false)
          setActiveIndex(-1)
          setQuery(selectedLabel)
        }
      }}
    >
      <input
        ref={inputRef}
        type="text"
        role="combobox"
        aria-autocomplete="list"
        aria-controls={listboxId}
        aria-expanded={open}
        aria-activedescendant={open && activeIndex >= 0 ? `${listboxId}-option-${activeIndex}` : undefined}
        value={query}
        placeholder={options.length ? placeholder : `无可用${optionName}`}
        disabled={!options.length}
        autoComplete="off"
        onFocus={(event) => {
          setOpen(true)
          setActiveIndex(filteredOptions.length ? 0 : -1)
          // 聚焦后选中当前展示值，用户可直接输入关键词替换。
          event.currentTarget.select()
        }}
        onChange={(event) => {
          setQuery(event.target.value)
          setOpen(true)
          setActiveIndex(options.length ? 0 : -1)
        }}
        onKeyDown={handleKeyDown}
      />
      <button
        className="filter-combobox-toggle"
        type="button"
        aria-label={open ? `关闭${optionName}列表` : `展开${optionName}列表`}
        aria-expanded={open}
        disabled={!options.length}
        onClick={() => {
          if (open) {
            setOpen(false)
            setActiveIndex(-1)
            setQuery(selectedLabel)
          } else {
            inputRef.current?.focus()
            // 先取得输入焦点，再展开完整列表，避免焦点事件覆盖当前高亮项。
            openAllOptions()
          }
        }}
      >
        <span aria-hidden="true" />
      </button>
      {open && (
        <div className="filter-options" id={listboxId} role="listbox" aria-label={`${optionName}候选列表`}>
          {filteredOptions.length ? filteredOptions.map((option, index) => (
            <button
              id={`${listboxId}-option-${index}`}
              type="button"
              role="option"
              tabIndex={-1}
              aria-selected={option.value === value}
              className={index === activeIndex ? 'active' : ''}
              key={option.value}
              onMouseEnter={() => setActiveIndex(index)}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => selectOption(option)}
            >
              <strong>{option.label}</strong>
              {option.secondary && <small>{option.secondary}</small>}
            </button>
          )) : <div className="filter-options-empty" role="status">没有匹配的{optionName}</div>}
        </div>
      )}
    </div>
  )
}

export function EnterpriseCombobox({ users, userId, onUserChange }: {
  users: UserSummary[]
  userId: string
  onUserChange: (userId: string) => void
}) {
  return <SearchableCombobox
    options={users.map((user) => ({ value: user.user_id, label: user.company_name, secondary: user.user_id }))}
    value={userId}
    optionName="企业"
    placeholder="输入企业名称或编号"
    onChange={onUserChange}
  />
}

export function DateCombobox({ dates, date, onDateChange }: {
  dates: string[]
  date: string
  onDateChange: (date: string) => void
}) {
  return <SearchableCombobox
    options={dates.map((item) => ({ value: item, label: item }))}
    value={date}
    optionName="日期"
    placeholder="输入检测日期"
    onChange={onDateChange}
  />
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
            <EnterpriseCombobox users={users} userId={userId} onUserChange={onUserChange} />
          </label>
        )}
        {page !== 'safety' && page !== 'chat' && <label>
          <span>检测日期</span>
          <DateCombobox dates={dates} date={date} onDateChange={onDateChange} />
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
