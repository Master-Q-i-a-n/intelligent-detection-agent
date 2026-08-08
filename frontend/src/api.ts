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
  UserListResponse,
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
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
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
  users: (signal?: AbortSignal) => requestJson<UserListResponse>('/api/users', { signal }),
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
      body: JSON.stringify({ user_id: userId, diagnosis_date: date, save: false, deep_model: false }),
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
}
