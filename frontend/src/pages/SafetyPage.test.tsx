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
    render(<SafetyPage refreshToken={0} onBusyChange={vi.fn()} onChanged={vi.fn()} username="admin" />)
    expect(await screen.findByText('目标人员头部清晰可见，确认未佩戴安全帽。')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '打开事件 →' }))
    expect(await screen.findByText('复核确认存在安全问题')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '确认收到' }))
    await waitFor(() => expect(api.securityAction).toHaveBeenCalledWith('event-1', 'ACKNOWLEDGE', 'admin', ''))
  })

  it('收到新的通知序号后自动刷新事件列表', async () => {
    const props = { refreshToken: 0, onBusyChange: vi.fn(), onChanged: vi.fn(), username: 'admin' }
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
    const { container } = render(<SafetyPage refreshToken={0} onBusyChange={vi.fn()} onChanged={vi.fn()} username="admin" />)
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

  it('PPE 详情按截图、视频顺序展示，并区分规则与豆包结论', async () => {
    const ppeEvent: SecurityEvent = {
      ...event,
      event_id: 'PPE_hash',
      event_type: 'PPE_INSPECTION',
      final_reason: '联合复核确认存在 PPE 佩戴异常。',
      evidence_count: 3,
      evidence: [
        { evidence_id: 11, evidence_type: 'OVERVIEW_IMAGE', mime_type: 'image/jpeg' },
        { evidence_id: 12, evidence_type: 'PPE_ANOMALY_IMAGE', mime_type: 'image/jpeg' },
        { evidence_id: 13, evidence_type: 'REVIEW_VIDEO', mime_type: 'video/mp4' },
      ],
      people: [{ track_id: 4, helmet_status: 'REMOVED_DURING_WORK', gloves_status: 'WORN', goggles_status: 'UNCERTAIN' }],
      yolo_rule_metrics: {
        anchor_track_id: 4,
        anchor_time_seconds: 1.6,
        people: [{
          track_id: 4,
          helmet_rule_status: 'REMOVED_DURING_WORK',
          helmet_last_worn_at: 1.2,
          first_no_helmet_at: 1.6,
          gloves_rule_status: 'WORN',
          goggles_rule_status: 'NEEDS_REVIEW',
          gloves_positive_frames: 8,
          goggles_positive_frames: 0,
          no_gloves_positive_frames: 1,
          no_goggle_positive_frames: 5,
          gloves_effective_seconds: 0.8,
          goggles_effective_seconds: 0,
        }],
      },
      latest_review_ppe_results: [{
        track_id: 4,
        helmet_status: 'REMOVED_DURING_WORK', gloves_status: 'WORN', goggles_status: 'UNCERTAIN',
        visibility: 'PARTIAL', evidence_quality: 'LIMITED',
        visual_reason: '人员先佩戴安全帽，随后摘下并拿在手中。', evidence_timestamps: [1.1, 1.7],
      }],
    }
    vi.mocked(api.securityEvent).mockResolvedValue(ppeEvent)
    const { container } = render(<SafetyPage refreshToken={0} onBusyChange={vi.fn()} onChanged={vi.fn()} username="admin" />)
    fireEvent.click(await screen.findByRole('button', { name: '打开事件 →' }))

    expect(await screen.findByText('YOLO 规则检测结果')).toBeInTheDocument()
    expect(screen.getByText('豆包视觉复核')).toBeInTheDocument()
    expect(screen.queryByText('多模态复核说明')).not.toBeInTheDocument()
    expect(screen.getByText(/正向 8 帧 \/ 0.80 秒/)).toBeInTheDocument()
    expect(screen.getByText('人员先佩戴安全帽，随后摘下并拿在手中。')).toBeInTheDocument()
    expect(screen.getByText(/可见性：部分可见 · 证据质量：有限/)).toBeInTheDocument()
    expect(screen.queryByText(/时间戳：/)).not.toBeInTheDocument()

    const gallery = container.querySelector('.evidence-gallery')
    const image = gallery?.querySelector('img')
    const video = gallery?.querySelector('video')
    expect(image?.getAttribute('alt')).toBe('异常时刻截图')
    expect(gallery?.querySelectorAll('img')).toHaveLength(1)
    expect(image && video && (image.compareDocumentPosition(video) & Node.DOCUMENT_POSITION_FOLLOWING)).toBeTruthy()
  })
})
