import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import { Sidebar, Topbar } from './components/Layout'
import type { AgentRecord, AutoAgentRequest } from './components/AgentControls'
import { EmptyState, LoadingState } from './components/Common'
import { EquipmentPage } from './pages/EquipmentPage'
import { MeteringPage } from './pages/MeteringPage'
import { OverviewPage } from './pages/OverviewPage'
import { SafetyPage } from './pages/SafetyPage'
import { ChatPage } from './pages/ChatPage'
import { AuthPage } from './pages/AuthPage'
import type { AuthUser, PageKey, SecurityEvent, SecurityOverview, UserSummary } from './types'

const AUTO_AGENT_STORAGE_KEY = 'yaoheng:auto-agent-enabled'

function datesBetween(range: [string, string] | [] | undefined): string[] {
  if (!range || range.length !== 2 || !range[0] || !range[1]) return []
  const start = new Date(`${range[0]}T00:00:00Z`)
  const end = new Date(`${range[1]}T00:00:00Z`)
  if (!Number.isFinite(start.getTime()) || !Number.isFinite(end.getTime()) || start > end) return []
  const dates: string[] = []
  for (const cursor = new Date(start); cursor <= end; cursor.setUTCDate(cursor.getUTCDate() + 1)) {
    dates.push(cursor.toISOString().slice(0, 10))
  }
  return dates
}

export default function App() {
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null)
  const [authChecking, setAuthChecking] = useState(true)
  const [page, setPage] = useState<PageKey>('overview')
  const [users, setUsers] = useState<UserSummary[]>([])
  const [globalRange, setGlobalRange] = useState<[string, string] | []>([])
  const [userId, setUserId] = useState('')
  const [date, setDate] = useState('')
  const [initializing, setInitializing] = useState(true)
  const [initialError, setInitialError] = useState('')
  const [refreshToken, setRefreshToken] = useState(0)
  const [busy, setBusy] = useState(false)
  const [autoAgentEnabled, setAutoAgentEnabled] = useState(() => {
    // 首次使用默认关闭；用户选择写入本地设置，刷新页面时恢复。
    try {
      return window.localStorage.getItem(AUTO_AGENT_STORAGE_KEY) === 'true'
    } catch {
      return false
    }
  })
  const [agentRecords, setAgentRecords] = useState<Record<string, AgentRecord>>({})
  const [securityOverviewState, setSecurityOverviewState] = useState<SecurityOverview | null>(null)
  const [securityToast, setSecurityToast] = useState<SecurityEvent | null>(null)
  const securitySequence = useRef<number | null>(null)
  const claimedAgentKeys = useRef(new Set<string>())
  const agentControllers = useRef(new Map<string, AbortController>())

  useEffect(() => {
    const controller = new AbortController()
    api.currentUser(controller.signal)
      .then(setCurrentUser)
      .catch(() => setCurrentUser(null))
      .finally(() => { if (!controller.signal.aborted) setAuthChecking(false) })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!currentUser) return
    const controller = new AbortController()
    setInitializing(true)
    api.users(controller.signal)
      .then((result) => {
        const nextUsers = result.items || []
        const firstUser = nextUsers[0]
        const initialDates = datesBetween(firstUser?.date_range || result.date_range)
        setUsers(nextUsers)
        setGlobalRange(result.date_range)
        setUserId(firstUser?.user_id || '')
        setDate(initialDates.at(-1) || '')
        if (!initialDates.length) setInitialError('接口未返回有效检测日期，未发起任何无日期请求。')
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setInitialError(reason instanceof Error ? reason.message : '初始化失败')
      })
      .finally(() => {
        if (!controller.signal.aborted) setInitializing(false)
      })
    return () => controller.abort()
  }, [currentUser])

  const selectedUser = useMemo(() => users.find((user) => user.user_id === userId), [userId, users])
  const dates = useMemo(() => datesBetween(selectedUser?.date_range || globalRange), [globalRange, selectedUser])
  const activeAgentKey = page === 'metering' || page === 'equipment' ? `${page}:${userId}:${date}` : ''

  const changeUser = useCallback((nextUserId: string) => {
    setUserId(nextUserId)
    const nextUser = users.find((user) => user.user_id === nextUserId)
    const nextDates = datesBetween(nextUser?.date_range || globalRange)
    setDate((current) => nextDates.includes(current) ? current : (nextDates.at(-1) || ''))
  }, [globalRange, users])

  const changeAutoAgent = useCallback((enabled: boolean) => {
    setAutoAgentEnabled(enabled)
    try {
      window.localStorage.setItem(AUTO_AGENT_STORAGE_KEY, String(enabled))
    } catch {
      // 浏览器禁用本地存储时仍保留本次页面会话内的设置。
    }
    if (!enabled) {
      // 取消浏览器等待并依赖请求控制器忽略迟到响应；服务端正在执行的 LLM 无法保证撤回。
      agentControllers.current.forEach((controller) => controller.abort())
      agentControllers.current.clear()
      setAgentRecords((current) => Object.fromEntries(Object.entries(current).map(([key, record]) => [
        key,
        record.status === 'loading' ? { status: 'idle' as const } : record,
      ])))
    }
  }, [])

  const requestAutoAgent = useCallback((request: AutoAgentRequest) => {
    if (claimedAgentKeys.current.has(request.key)) return
    claimedAgentKeys.current.add(request.key)
    const controller = new AbortController()
    agentControllers.current.set(request.key, controller)
    setAgentRecords((current) => ({ ...current, [request.key]: { status: 'loading' } }))
    api.inspectAgent(request.payload, controller.signal)
      .then((report) => {
        if (!controller.signal.aborted && agentControllers.current.get(request.key) === controller) {
          setAgentRecords((current) => ({ ...current, [request.key]: { status: 'success', report } }))
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted && agentControllers.current.get(request.key) === controller) {
          setAgentRecords((current) => ({ ...current, [request.key]: { status: 'error', error: reason instanceof Error ? reason.message : '未知错误' } }))
        }
      })
      .finally(() => {
        if (agentControllers.current.get(request.key) === controller) agentControllers.current.delete(request.key)
      })
  }, [])

  useEffect(() => {
    // 离开详情、切换模块/企业/日期时，取消旧键的前端等待并标记为已忽略。
    agentControllers.current.forEach((controller, key) => {
      if (key === activeAgentKey) return
      controller.abort()
      agentControllers.current.delete(key)
      setAgentRecords((current) => ({
        ...current,
        [key]: current[key]?.status === 'loading'
          ? { status: 'error', error: '页面上下文已变化，本次响应已忽略。' }
          : current[key],
      }))
    })
  }, [activeAgentKey])

  useEffect(() => () => {
    agentControllers.current.forEach((controller) => controller.abort())
  }, [])

  const refreshSecurityOverview = useCallback(async () => {
    try {
      setSecurityOverviewState(await api.securityOverview())
    } catch {
      // 安防模块离线不阻断计量和设备页面，页面内会展示明确错误状态。
    }
  }, [])

  useEffect(() => {
    if (!currentUser) {
      securitySequence.current = null
      setSecurityOverviewState(null)
      return
    }
    let active = true
    async function pollSecurity() {
      try {
        const overview = await api.securityOverview()
        if (!active) return
        setSecurityOverviewState(overview)
        if (securitySequence.current === null) {
          securitySequence.current = overview.latest_sequence
          return
        }
        const result = await api.securityEvents({ after_sequence: securitySequence.current, limit: 20 })
        if (!active) return
        if (result.items.length) {
          securitySequence.current = Math.max(...result.items.map((item) => item.notification_sequence), securitySequence.current)
          const confirmed = result.items.find((item) => item.final_decision === 'CONFIRMED')
          if (confirmed) setSecurityToast(confirmed)
        }
      } catch {
        // 轮询失败会在下一周期自动恢复，不重复弹出全局错误。
      }
    }
    void pollSecurity()
    const timer = window.setInterval(pollSecurity, 5_000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [currentUser])

  const openIssue = useCallback((nextPage: PageKey, nextUserId: string) => {
    changeUser(nextUserId)
    setPage(nextPage)
  }, [changeUser])

  async function logout() {
    agentControllers.current.forEach((controller) => controller.abort())
    try {
      await api.logout()
    } finally {
      setCurrentUser(null)
      setAgentRecords({})
      setUsers([])
      setPage('overview')
      setSecurityToast(null)
    }
  }

  if (authChecking) return <div className="boot-screen"><LoadingState label="正在验证登录状态" /></div>

  return (
    <div className="app-shell">
      <Sidebar page={page} onPageChange={setPage} autoAgentEnabled={autoAgentEnabled} onAutoAgentChange={changeAutoAgent} safetyBadge={securityOverviewState?.new_count || 0} currentUser={currentUser} onLogout={() => void logout()} />
      {!currentUser ? <AuthPage onAuthenticated={setCurrentUser} /> : initializing ? <div className="boot-screen"><LoadingState label="正在建立数据链路" /></div> : <main className="main-shell">
        <Topbar
          page={page}
          users={users}
          userId={userId}
          date={date}
          dates={dates}
          busy={busy}
          onUserChange={changeUser}
          onDateChange={setDate}
          onRefresh={() => setRefreshToken((value) => value + 1)}
        />
        <div className="content-shell">
          {initialError && !date && page !== 'safety' && page !== 'chat' ? (
            <EmptyState title="无法进入检测流程" detail={initialError} />
          ) : (
            <>
              {page === 'overview' && <OverviewPage active date={date} refreshToken={refreshToken} onBusyChange={setBusy} onOpenIssue={openIssue} />}
              {page === 'metering' && <MeteringPage active userId={userId} date={date} refreshToken={refreshToken} autoAgentEnabled={autoAgentEnabled} autoRecord={agentRecords[activeAgentKey]} onAutoRequest={requestAutoAgent} onBusyChange={setBusy} />}
              {page === 'equipment' && <EquipmentPage active userId={userId} date={date} refreshToken={refreshToken} autoAgentEnabled={autoAgentEnabled} autoRecord={agentRecords[activeAgentKey]} onAutoRequest={requestAutoAgent} onBusyChange={setBusy} />}
              {page === 'safety' && <SafetyPage refreshToken={refreshToken} liveSequence={securityOverviewState?.latest_sequence || 0} onBusyChange={setBusy} onChanged={refreshSecurityOverview} username={currentUser.username} />}
              {page === 'chat' && <ChatPage />}
            </>
          )}
        </div>
      </main>}
      {securityToast && (
        <button className="security-toast" type="button" onClick={() => { setPage('safety'); setSecurityToast(null) }} aria-label="打开新安防告警">
          <span>SAFETY ALERT</span>
          <strong>发现新的安全作业问题</strong>
          <small>{securityToast.latest_review_explanation || securityToast.final_reason || securityToast.event_type}</small>
          <i>打开事件 →</i>
        </button>
      )}
    </div>
  )
}

export { datesBetween }
