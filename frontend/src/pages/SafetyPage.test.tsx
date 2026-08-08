import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api'
import type { SecurityEvent } from '../types'
import { SafetyPage } from './SafetyPage'

vi.mock('../api', () => ({
  api: {
    securityOverview: vi.fn(), securityEvents: vi.fn(), securityEvent: vi.fn(), securityAction: vi.fn(),
  },
}))

const event: SecurityEvent = {
  notification_sequence: 1, notification_kind: 'CONFIRMED_ALERT', sent_at: '2026-08-08T10:00:00Z',
  event_id: 'event-1', source_system: 'yolo_track', event_type: 'NO_HELMET', camera_id: 'camera_01',
  zone_id: 'work_area', primary_track_id: 7, lifecycle_status: 'ACTIVE', activated_video_seconds: 5,
  occurred_at: null, severity: 'MEDIUM', final_decision: 'CONFIRMED', final_reason: '确认未佩戴安全帽',
  recommended_action: '立即纠正', handling_status: 'NEW', latest_review_helmet_status: 'NOT_WORN',
  latest_review_explanation: '目标人员头部清晰可见，确认未佩戴安全帽。', latest_reviewed_at: '2026-08-08T10:00:00Z',
  evidence_count: 0, evidence: [], actions: [],
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api.securityOverview).mockResolvedValue({ total: 1, confirmed: 1, review_required: 0, new_count: 1, processing: 0, high_risk: 0, latest_sequence: 1 })
  vi.mocked(api.securityEvents).mockResolvedValue({ items: [event] })
  vi.mocked(api.securityEvent).mockResolvedValue(event)
  vi.mocked(api.securityAction).mockResolvedValue({ event_id: 'event-1', previous_status: 'NEW', handling_status: 'ACKNOWLEDGED' })
})

describe('安全作业页面', () => {
  it('展示确认事件并写出处置动作', async () => {
    render(<SafetyPage refreshToken={0} onBusyChange={vi.fn()} onChanged={vi.fn()} />)
    expect(await screen.findByText('目标人员头部清晰可见，确认未佩戴安全帽。')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '打开事件 →' }))
    expect(await screen.findByText('复核确认存在安全问题')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '确认收到' }))
    await waitFor(() => expect(api.securityAction).toHaveBeenCalledWith('event-1', 'ACKNOWLEDGE', '平台操作员', ''))
  })

  it('收到新的通知序号后自动刷新事件列表', async () => {
    const props = { refreshToken: 0, onBusyChange: vi.fn(), onChanged: vi.fn() }
    const { rerender } = render(<SafetyPage {...props} liveSequence={1} />)
    await waitFor(() => expect(api.securityEvents).toHaveBeenCalledTimes(1))

    rerender(<SafetyPage {...props} liveSequence={2} />)

    await waitFor(() => expect(api.securityEvents).toHaveBeenCalledTimes(2))
  })

  it('自动循环播放证据视频并在事件关闭后锁定输入', async () => {
    const closedEvent: SecurityEvent = {
      ...event,
      handling_status: 'CLOSED',
      evidence: [{ evidence_id: 10, evidence_type: 'REVIEW_VIDEO', mime_type: 'video/mp4' }],
    }
    vi.mocked(api.securityEvent).mockResolvedValue(closedEvent)
    const { container } = render(<SafetyPage refreshToken={0} onBusyChange={vi.fn()} onChanged={vi.fn()} />)
    fireEvent.click(await screen.findByRole('button', { name: '打开事件 →' }))

    const video = await waitFor(() => {
      const element = container.querySelector('video')
      expect(element).not.toBeNull()
      return element as HTMLVideoElement
    })
    expect(video.autoplay).toBe(true)
    expect(video.loop).toBe(true)
    expect(video.muted).toBe(true)
    expect(screen.getByRole('textbox', { name: '操作人' })).toBeDisabled()
    expect(screen.getByRole('textbox', { name: '处置备注' })).toBeDisabled()
  })
})
