import { afterEach, describe, expect, it, vi } from 'vitest'
import { requestSse } from './api'
import type { ChatStreamEvent } from './types'

afterEach(() => vi.restoreAllMocks())

describe('SSE 客户端', () => {
  it('合并跨分片事件并忽略心跳注释', async () => {
    const chunks = [
      'event: meta\r\ndata: {"run_id":"r1"}\r\n\r\n: keep',
      '-alive\n\nevent: answer_delta\ndata: {"delta":"你"}\n\n',
      'event: answer_reset\ndata: {"message_id":"m1"}\n\nevent: done\ndata: {"status":"completed"}\n\n',
    ]
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        chunks.forEach((chunk) => controller.enqueue(new TextEncoder().encode(chunk)))
        controller.close()
      },
    })
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(stream, { status: 200 }))
    const events: ChatStreamEvent[] = []
    await requestSse('/chat/turns/stream', { method: 'POST', body: '{}', onEvent: (event) => events.push(event) })
    expect(events.map((event) => event.event)).toEqual(['meta', 'answer_delta', 'answer_reset', 'done'])
    expect(events[1].data.delta).toBe('你')
  })

  it('在流式接口返回 HTTP 错误时给出统一异常', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ detail: '未配置' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
    }))
    await expect(requestSse('/chat/turns/stream', { onEvent: () => undefined })).rejects.toMatchObject({ status: 503, message: '未配置', name: 'ApiError' })
  })
})
