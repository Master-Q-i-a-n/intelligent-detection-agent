import type {
  AgentInspectionPayload,
  AgentReport,
  DailyOverview,
  EquipmentDashboard,
  EquipmentWaveform,
  MeteringDiagnosis,
  MeteringHistoryItem,
  MeteringSignals,
  SecurityEvent,
  SecurityOverview,
  ChatResumePayload,
  ChatStatus,
  ChatStreamEvent,
  ChatTurnResponse,
  UserListResponse,
  AuthUser,
  ChatThreadDetail,
  ChatThreadSummary,
  ChatAttachment,
} from './types'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

interface StreamOptions extends RequestInit {
  onEvent: (event: ChatStreamEvent) => void
}

/** 解析 FastAPI 返回的 POST + SSE；注释心跳不会进入业务事件。 */
export async function requestSse(path: string, options: StreamOptions): Promise<void> {
  const { onEvent, ...requestOptions } = options
  const response = await fetch(path, {
    ...requestOptions,
    headers: {
      Accept: 'text/event-stream',
      'Content-Type': 'application/json',
      ...requestOptions.headers,
    },
  })
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null
    throw new ApiError(payload?.detail || `请求失败（HTTP ${response.status}）`, response.status)
  }
  if (!response.body) throw new ApiError('浏览器未收到流式响应', 502)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const dispatchBlock = (block: string) => {
    let eventName = 'message'
    const dataLines: string[] = []
    block.split('\n').forEach((line) => {
      if (line.startsWith(':')) return
      if (line.startsWith('event:')) eventName = line.slice(6).trim()
      if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
    })
    if (!dataLines.length || eventName === 'message') return
    const data = JSON.parse(dataLines.join('\n')) as Record<string, unknown>
    onEvent({ event: eventName as ChatStreamEvent['event'], data })
  }

  try {
    while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value, { stream: !done }).replaceAll('\r\n', '\n')
      let boundary = buffer.indexOf('\n\n')
      while (boundary >= 0) {
        dispatchBlock(buffer.slice(0, boundary))
        buffer = buffer.slice(boundary + 2)
        boundary = buffer.indexOf('\n\n')
      }
      if (done) break
    }
    if (buffer.trim()) dispatchBlock(buffer)
  } catch (error) {
    // 回调解析失败时主动取消响应体，使服务端尽快感知客户端已离开。
    await reader.cancel().catch(() => undefined)
    throw error
  } finally {
    reader.releaseLock()
  }
}

interface RequestOptions extends RequestInit {
  timeoutMs?: number
}

export async function requestJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), options.timeoutMs ?? 30_000)
  const externalSignal = options.signal
  const abortFromOutside = () => controller.abort()
  externalSignal?.addEventListener('abort', abortFromOutside, { once: true })

  try {
    const response = await fetch(path, {
      ...options,
      signal: controller.signal,
      headers: {
        Accept: 'application/json',
        ...(options.body && !(options.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}),
        ...options.headers,
      },
    })
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null
    if (!response.ok) {
      throw new ApiError(payload?.detail || `请求失败（HTTP ${response.status}）`, response.status)
    }
    return payload as T
  } catch (error) {
    if (controller.signal.aborted && !externalSignal?.aborted) {
      throw new ApiError('请求超时，请稍后重试', 408)
    }
    throw error
  } finally {
    window.clearTimeout(timeout)
    externalSignal?.removeEventListener('abort', abortFromOutside)
  }
}

export const api = {
  currentUser: (signal?: AbortSignal) => requestJson<AuthUser>('/auth/me', { signal }),
  login: (username: string, password: string, signal?: AbortSignal) =>
    requestJson<AuthUser>('/auth/login', {
      method: 'POST', body: JSON.stringify({ username, password }), signal,
    }),
  register: (username: string, password: string, signal?: AbortSignal) =>
    requestJson<AuthUser>('/auth/register', {
      method: 'POST', body: JSON.stringify({ username, password }), signal,
    }),
  logout: (signal?: AbortSignal) => requestJson<void>('/auth/logout', { method: 'POST', signal }),
  users: (module?: 'metering' | 'equipment', signal?: AbortSignal) => {
    const query = module ? `?module=${encodeURIComponent(module)}` : ''
    return requestJson<UserListResponse>(`/api/users${query}`, { signal })
  },
  overview: (date: string, signal?: AbortSignal) =>
    requestJson<DailyOverview>(`/daily/overview/${encodeURIComponent(date)}`, { signal }),
  meteringHistory: (userId: string, date: string, signal?: AbortSignal) =>
    requestJson<{ user_id: string; items: MeteringHistoryItem[] }>(
      `/daily/metering-history/${encodeURIComponent(userId)}/${encodeURIComponent(date)}?days=7`,
      { signal },
    ),
  meteringDiagnosis: (userId: string, date: string, signal?: AbortSignal) =>
    requestJson<MeteringDiagnosis>('/metering/diagnose', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        diagnosis_date: date,
        save: true,
        deep_model: false,
        create_work_order: false,
      }),
      signal,
      timeoutMs: 90_000,
    }),
  meteringSignals: (userId: string, date: string, signal?: AbortSignal) =>
    requestJson<MeteringSignals>(
      `/metering/signals/${encodeURIComponent(userId)}/${encodeURIComponent(date)}`,
      { signal, timeoutMs: 60_000 },
    ),
  equipmentDashboard: (userId: string, date: string, signal?: AbortSignal) =>
    requestJson<EquipmentDashboard>(
      `/equipment/dashboard/${encodeURIComponent(userId)}/${encodeURIComponent(date)}`,
      { signal },
    ),
  equipmentWaveform: (userId: string, date: string, signal?: AbortSignal) =>
    requestJson<EquipmentWaveform>(
      `/equipment/waveform/${encodeURIComponent(userId)}/${encodeURIComponent(date)}?points=180`,
      { signal },
    ),
  inspectAgent: (payload: AgentInspectionPayload, signal?: AbortSignal) =>
    requestJson<AgentReport>('/agent/inspect', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal,
      timeoutMs: 120_000,
    }),
  securityOverview: (signal?: AbortSignal) => requestJson<SecurityOverview>('/security/overview', { signal }),
  securityEvents: (params: Record<string, string | number | undefined> = {}, signal?: AbortSignal) => {
    const query = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== '') query.set(key, String(value))
    })
    const suffix = query.size ? `?${query.toString()}` : ''
    return requestJson<{ items: SecurityEvent[] }>(`/security/events${suffix}`, { signal })
  },
  securityEvent: (eventId: string, signal?: AbortSignal) =>
    requestJson<SecurityEvent>(`/security/events/${encodeURIComponent(eventId)}`, { signal }),
  securityAction: (eventId: string, action: string, operator: string, comment = '') =>
    requestJson<{ event_id: string; previous_status: string; handling_status: string }>(
      `/security/events/${encodeURIComponent(eventId)}/actions`,
      { method: 'POST', body: JSON.stringify({ action, operator, comment }) },
    ),
  chatStatus: (signal?: AbortSignal) => requestJson<ChatStatus>('/chat/status', { signal }),
  chatThreads: (signal?: AbortSignal) => requestJson<{ items: ChatThreadSummary[] }>('/chat/threads', { signal }),
  chatThread: (threadId: string, signal?: AbortSignal) =>
    requestJson<ChatThreadDetail>(`/chat/threads/${encodeURIComponent(threadId)}`, { signal }),
  deleteChatThread: (threadId: string, signal?: AbortSignal) =>
    requestJson<void>(`/chat/threads/${encodeURIComponent(threadId)}`, { method: 'DELETE', signal }),
  uploadChatAttachment: (threadId: string, file: File, signal?: AbortSignal) => {
    const body = new FormData()
    body.append('thread_id', threadId)
    body.append('file', file, file.name)
    return requestJson<ChatAttachment>('/chat/attachments', {
      method: 'POST', body, signal, timeoutMs: 60_000,
    })
  },
  deleteChatAttachment: (attachmentId: string, signal?: AbortSignal) =>
    requestJson<void>(`/chat/attachments/${encodeURIComponent(attachmentId)}`, { method: 'DELETE', signal }),
  chatTurn: (threadId: string, message: string, attachmentIds: string[] = [], signal?: AbortSignal) =>
    requestJson<ChatTurnResponse>('/chat/turns', {
      method: 'POST',
      body: JSON.stringify({ thread_id: threadId, message, attachment_ids: attachmentIds }),
      signal,
      timeoutMs: 180_000,
    }),
  chatResume: (payload: ChatResumePayload, signal?: AbortSignal) =>
    requestJson<ChatTurnResponse>('/chat/resume', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal,
      timeoutMs: 180_000,
    }),
  chatTurnStream: (threadId: string, message: string, attachmentIds: string[], onEvent: (event: ChatStreamEvent) => void, signal?: AbortSignal) =>
    requestSse('/chat/turns/stream', {
      method: 'POST', body: JSON.stringify({ thread_id: threadId, message, attachment_ids: attachmentIds }), onEvent, signal,
    }),
  chatResumeStream: (payload: ChatResumePayload, onEvent: (event: ChatStreamEvent) => void, signal?: AbortSignal) =>
    requestSse('/chat/resume/stream', {
      method: 'POST', body: JSON.stringify(payload), onEvent, signal,
    }),
}
