import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import { Sidebar, Topbar } from './components/Layout'
import type { AgentRecord, AutoAgentRequest } from './components/AgentControls'
import { EmptyState, LoadingState } from './components/Common'
import { AuthPage } from './pages/AuthPage'
import type { AuthUser, PageKey, SecurityEvent, SecurityOverview, UserListResponse } from './types'

// 页面代码仅在进入对应模块时下载，登录和导航不再等待图表、问答等依赖。
const pageLoaders = {
  overview: () => import('./pages/OverviewPage').then((module) => ({ default: module.OverviewPage })),
  metering: () => import('./pages/MeteringPage').then((module) => ({ default: module.MeteringPage })),
  equipment: () => import('./pages/EquipmentPage').then((module) => ({ default: module.EquipmentPage })),
  safety: () => import('./pages/SafetyPage').then((module) => ({ default: module.SafetyPage })),
  chat: () => import('./pages/ChatPage').then((module) => ({ default: module.ChatPage })),
  knowledge: () => import('./pages/KnowledgePage').then((module) => ({ default: module.KnowledgePage })),
}
const OverviewPage = lazy(pageLoaders.overview)
const MeteringPage = lazy(pageLoaders.metering)
const EquipmentPage = lazy(pageLoaders.equipment)
const SafetyPage = lazy(pageLoaders.safety)
const ChatPage = lazy(pageLoaders.chat)
const KnowledgePage = lazy(pageLoaders.knowledge)

type CatalogState = { data?: UserListResponse; error?: string }

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
  const [catalogs, setCatalogs] = useState<Partial<Record<'metering' | 'equipment', CatalogState>>>({})
  const [storedUserId, setUserId] = useState('')
  const [storedDate, setDate] = useState('')
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
    // 当前页代码与企业索引同时加载，避免先等数据再下载页面的串行等待。
    void pageLoaders[page]().catch(() => undefined)
  }, [currentUser, page])

  useEffect(() => {
    if (!currentUser) return
    const controller = new AbortController()
    setCatalogs({})
    // 两份索引独立返回，慢请求或单模块故障不阻塞其他页面。
    for (const module of ['metering', 'equipment'] as const) {
      api.users(module, controller.signal)
        .then((data) => {
          if (!controller.signal.aborted) setCatalogs((current) => ({ ...current, [module]: { data } }))
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted) setCatalogs((current) => ({
            ...current, [module]: { error: reason instanceof Error ? reason.message : '初始化失败' },
          }))
        })
    }
    return () => controller.abort()
  }, [currentUser])

  const meteringUsers = useMemo(() => catalogs.metering?.data?.items || [], [catalogs.metering])
  const equipmentUsers = useMemo(() => catalogs.equipment?.data?.items || [], [catalogs.equipment])
  // 总览优先使用计量日期；计量没有日期或加载失败时，才等待设备索引兜底。
  const overviewCatalog = catalogs.metering?.data?.date_range.length ? catalogs.metering : catalogs.equipment
  const globalRange = overviewCatalog?.data?.date_range || []
  const selectedCatalog = page === 'overview' ? (catalogs.metering ? overviewCatalog : undefined)
    : catalogs[page === 'equipment' ? 'equipment' : 'metering']
  const needsCatalog = page === 'overview' || page === 'metering' || page === 'equipment'
  const initializing = needsCatalog && !selectedCatalog
  const initialError = selectedCatalog?.error || '接口未返回有效检测日期，未发起任何无日期请求。'
  const users = page === 'equipment' ? equipmentUsers : meteringUsers
  const selectedUser = useMemo(() => users.find((user) => user.user_id === storedUserId) || users[0], [storedUserId, users])
  const userId = selectedUser?.user_id || ''
  const dates = useMemo(
    () => datesBetween(page === 'metering' || page === 'equipment'
      ? (selectedUser?.date_range || selectedCatalog?.data?.date_range) : selectedCatalog?.data?.date_range),
    [page, selectedCatalog, selectedUser],
  )
  const date = dates.includes(storedDate) ? storedDate : (dates.at(-1) || '')
  const activeAgentKey = page === 'metering' || page === 'equipment' ? `${page}:${userId}:${date}` : ''

  const changeUser = useCallback((nextUserId: string) => {
    setUserId(nextUserId)
    const nextUser = users.find((user) => user.user_id === nextUserId)
    const nextDates = datesBetween(nextUser?.date_range || globalRange)
    setDate((current) => nextDates.includes(current) ? current : (nextDates.at(-1) || ''))
  }, [globalRange, users])

  const changePage = useCallback((nextPage: PageKey) => {
    const nextUsers = nextPage === 'equipment' ? equipmentUsers : meteringUsers
    // 企业名单仍在初始化时只切换页面，不用空列表覆盖稍后返回的默认企业。
    if ((nextPage === 'metering' || nextPage === 'equipment') && nextUsers.length > 0 && !nextUsers.some((user) => user.user_id === userId)) {
      const firstUser = nextUsers[0]
      const nextDates = datesBetween(firstUser?.date_range || globalRange)
      setUserId(firstUser?.user_id || '')
      setDate(nextDates.at(-1) || '')
    }
    setPage(nextPage)
  }, [equipmentUsers, globalRange, meteringUsers, userId])

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
    const controller = new AbortController()
    let timer: number | undefined
    async function pollSecurity() {
      try {
        const overview = await api.securityOverview(controller.signal)
        if (!active) return
        setSecurityOverviewState(overview)
        if (securitySequence.current === null) {
          securitySequence.current = overview.latest_sequence
          return
        }
        const result = await api.securityEvents({ after_sequence: securitySequence.current, limit: 20 }, controller.signal)
        if (!active) return
        if (result.items.length) {
          securitySequence.current = Math.max(...result.items.map((item) => item.notification_sequence), securitySequence.current)
          const confirmed = result.items.find((item) => item.final_decision === 'CONFIRMED')
          if (confirmed) setSecurityToast(confirmed)
        }
      } catch {
        // 轮询失败会在下一周期自动恢复，不重复弹出全局错误。
      } finally {
        // 请求完成后再计时，避免服务慢时每五秒叠加一批请求。
        if (active) timer = window.setTimeout(pollSecurity, 5_000)
      }
    }
    void pollSecurity()
    return () => {
      active = false
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [currentUser])

  const openIssue = useCallback((nextPage: PageKey, nextUserId: string) => {
    const nextUsers = nextPage === 'equipment' ? equipmentUsers : meteringUsers
    const nextUser = nextUsers.find((user) => user.user_id === nextUserId)
    const nextDates = datesBetween(nextUser?.date_range || globalRange)
    setUserId(nextUserId)
    setDate((current) => nextDates.includes(current) ? current : (nextDates.at(-1) || ''))
    setPage(nextPage)
  }, [equipmentUsers, globalRange, meteringUsers])

  async function logout() {
    agentControllers.current.forEach((controller) => controller.abort())
    try {
      await api.logout()
    } finally {
      setCurrentUser(null)
      setAgentRecords({})
      setCatalogs({})
      setUserId('')
      setDate('')
      setPage('overview')
      setSecurityToast(null)
    }
  }

  if (authChecking) return <div className="boot-screen"><LoadingState label="正在验证登录状态" /></div>

  return (
    <div className="app-shell">
      <Sidebar page={page} onPageChange={changePage} autoAgentEnabled={autoAgentEnabled} onAutoAgentChange={changeAutoAgent} safetyBadge={securityOverviewState?.new_count || 0} currentUser={currentUser} onLogout={() => void logout()} />
      {!currentUser ? <AuthPage onAuthenticated={setCurrentUser} /> : <main className="main-shell">
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
          {initializing ? <LoadingState label="正在加载检测企业与日期" /> : !date && needsCatalog ? (
            <EmptyState title="无法进入检测流程" detail={initialError} />
          ) : (
            <Suspense fallback={<LoadingState label="正在加载页面" />}>
              {page === 'overview' && <OverviewPage active date={date} refreshToken={refreshToken} onBusyChange={setBusy} onOpenIssue={openIssue} />}
              {page === 'metering' && <MeteringPage active userId={userId} date={date} refreshToken={refreshToken} autoAgentEnabled={autoAgentEnabled} autoRecord={agentRecords[activeAgentKey]} onAutoRequest={requestAutoAgent} onBusyChange={setBusy} />}
              {page === 'equipment' && <EquipmentPage active userId={userId} date={date} refreshToken={refreshToken} autoAgentEnabled={autoAgentEnabled} autoRecord={agentRecords[activeAgentKey]} onAutoRequest={requestAutoAgent} onBusyChange={setBusy} />}
              {page === 'safety' && <SafetyPage refreshToken={refreshToken} liveSequence={securityOverviewState?.latest_sequence || 0} onBusyChange={setBusy} onChanged={refreshSecurityOverview} username={currentUser.username} />}
              {page === 'chat' && <ChatPage />}
              {page === 'knowledge' && <KnowledgePage />}
            </Suspense>
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
