import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api'
import { ChatPage } from './ChatPage'
import type { ChatStreamEvent, ChatTurnResponse } from '../types'

vi.mock('../api', () => ({
  api: {
    chatStatus: vi.fn(),
    chatTurn: vi.fn(),
    chatResume: vi.fn(),
    chatTurnStream: vi.fn(),
    chatResumeStream: vi.fn(),
    chatThreads: vi.fn(),
    chatThread: vi.fn(),
    deleteChatThread: vi.fn(),
  },
}))

function streamResponse(response: ChatTurnResponse) {
  return async (_threadId: string, _message: string, onEvent: (event: ChatStreamEvent) => void) => {
    onEvent({ event: 'meta', data: { run_id: 'run_test', thread_id: 'thread_test' } })
    response.artifacts.forEach((artifact) => onEvent({ event: 'artifact', data: artifact as unknown as Record<string, unknown> }))
    onEvent({ event: 'done', data: response as unknown as Record<string, unknown> })
  }
}

function resumeStreamResponse(response: ChatTurnResponse) {
  return async (_payload: unknown, onEvent: (event: ChatStreamEvent) => void) => {
    onEvent({ event: 'done', data: response as unknown as Record<string, unknown> })
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api.chatStatus).mockResolvedValue({ configured: true, provider: 'deepseek', model: 'deepseek-chat', memory: 'sqlite-user-thread', tracing_enabled: false })
  vi.mocked(api.chatThreads).mockResolvedValue({ items: [] })
})

describe('智能问答页面', () => {
  it('可以恢复并永久删除当前用户的历史对话', async () => {
    vi.mocked(api.chatThreads).mockResolvedValue({ items: [{
      thread_id: 'chat_history', title: '一月用气分析', created_at: '2025-01-01T08:00:00Z',
      updated_at: '2025-01-02T09:30:00Z', status: 'completed',
    }] })
    vi.mocked(api.chatThread).mockResolvedValue({
      thread: { thread_id: 'chat_history', title: '一月用气分析', created_at: '2025-01-01T08:00:00Z', updated_at: '2025-01-02T09:30:00Z' },
      messages: [
        { id: 'm1', role: 'user', content: '分析一月用气', artifact_ids: [], created_at: '2025-01-01T08:00:00Z' },
        { id: 'm2', role: 'assistant', content: '历史分析已完成。', generator: 'test', artifact_ids: ['q-history'], created_at: '2025-01-01T08:00:05Z' },
      ],
      artifacts: [{ type: 'query_result', id: 'q-history', payload: { type: 'query_result', query_id: 'q-history', source: 'business', sql: 'SELECT 1', columns: [], rows: [], row_count: 0, truncated: false, elapsed_ms: 1 } }],
      todos: [], interrupt: null, last_error: null,
    })
    vi.mocked(api.deleteChatThread).mockResolvedValue(undefined)
    render(<ChatPage />)
    const historyTitle = await screen.findByText('一月用气分析')
    fireEvent.click(historyTitle.closest('button') as HTMLButtonElement)
    expect(await screen.findByText('历史分析已完成。')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '查看 SQL 查询（1）' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '删除对话 一月用气分析' }))
    fireEvent.click(screen.getByRole('button', { name: '删除' }))
    await waitFor(() => expect(api.deleteChatThread).toHaveBeenCalledWith('chat_history'))
    expect(await screen.findByText('已开始新的对话。请告诉我需要查询的对象、时间或报告主题。')).toBeInTheDocument()
  })

  it('SQL 查询不会自动打开面板，可由当前回答按钮打开并隐藏', async () => {
    const response: ChatTurnResponse = {
      status: 'completed', message: '共有2户超过阈值。', generator: 'deepagents:deepseek:deepseek-chat', interrupt: null,
      artifacts: [{ type: 'query_result', id: 'qry_1', payload: {
        type: 'query_result', query_id: 'qry_1', source: 'business', sql: 'SELECT user_id, volume_m3 FROM telemetry.scada_observation',
        columns: ['user_id', 'volume_m3'], rows: [{ user_id: 'u1', volume_m3: 1200 }], row_count: 1, truncated: false, elapsed_ms: 8,
      } }], todos: [],
    }
    vi.mocked(api.chatTurnStream).mockImplementation(streamResponse(response))
    render(<ChatPage />)
    fireEvent.change(screen.getByRole('textbox', { name: '对话输入' }), { target: { value: '查询超过1000立方米的用户' } })
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    expect(await screen.findByText('共有2户超过阈值。')).toBeInTheDocument()
    expect(screen.getByText('deepseek-chat')).toBeInTheDocument()
    expect(screen.queryByText('deepagents:deepseek:deepseek-chat')).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '数据与报告' })).not.toBeInTheDocument()
    const openSql = screen.getByRole('button', { name: '查看 SQL 查询（1）' })
    fireEvent.click(openSql)
    expect(screen.getByText('qry_1')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '隐藏' }))
    expect(screen.queryByRole('region', { name: '数据与报告' })).not.toBeInTheDocument()
    await waitFor(() => expect(openSql).toHaveFocus())
    expect(api.chatTurnStream).toHaveBeenCalledTimes(1)
  })

  it('信息不足时补充条件并恢复同一会话', async () => {
    const interrupted: ChatTurnResponse = {
      status: 'interrupted', message: '', generator: 'deepagents:test', artifacts: [],
      interrupt: { kind: 'clarification', question: '请给出阈值', missing_information: ['用气量阈值'], suggestions: ['1000 m³'] },
    }
    vi.mocked(api.chatTurnStream).mockImplementation(streamResponse(interrupted))
    const completed: ChatTurnResponse = { status: 'completed', message: '已按1000 m³查询。', generator: 'deepagents:test', artifacts: [] }
    let completeResume: (() => void) | null = null
    vi.mocked(api.chatResumeStream).mockImplementation(async (_payload, onEvent) => new Promise<void>((resolve) => {
      completeResume = () => {
        onEvent({ event: 'done', data: completed as unknown as Record<string, unknown> })
        resolve()
      }
    }))
    render(<ChatPage />)
    fireEvent.change(screen.getByRole('textbox', { name: '对话输入' }), { target: { value: '查询超标用户' } })
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    expect(await screen.findByText('请给出阈值')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '1000 m³' }))
    fireEvent.click(screen.getByRole('button', { name: '继续处理' }))
    // 提交动作发生后卡片立即退出，不等待后端恢复完成。
    expect(screen.queryByText('需要补充信息')).not.toBeInTheDocument()
    await waitFor(() => expect(api.chatResumeStream).toHaveBeenCalledWith(expect.objectContaining({ kind: 'clarification', decision: 'answer', message: '1000 m³' }), expect.any(Function), expect.any(AbortSignal)))
    await act(async () => { completeResume?.() })
    expect(await screen.findByText('已按1000 m³查询。')).toBeInTheDocument()
  })

  it('工单批准前显示完整参数并显式确认', async () => {
    const interrupted: ChatTurnResponse = {
      status: 'interrupted', message: '', generator: 'deepagents:test', artifacts: [],
      interrupt: { kind: 'work_order_approval', action: { name: 'create_work_order', arguments: { source_module: 'metering', priority: 'P2', title: '现场核查' } }, allowed_decisions: ['approve', 'edit', 'reject'] },
    }
    vi.mocked(api.chatTurnStream).mockImplementation(streamResponse(interrupted))
    vi.mocked(api.chatResumeStream).mockImplementation(resumeStreamResponse({
      status: 'completed', message: '工单已创建。', generator: 'deepagents:test',
      artifacts: [{ type: 'work_order', id: 'wo_1', payload: {
        type: 'work_order', work_order_id: 'wo_1', title: '现场核查', status: 'OPEN', priority: 'P2',
        source_module: 'metering', user_id: 'u1', description: '核查表具异常并恢复缺失数据',
        checklist: ['核对表具参数', '复核采集链路'], source_reference: { diagnosis_date: '2025-01-12', run_id: 'run-1' },
      } }],
    }))
    render(<ChatPage />)
    fireEvent.change(screen.getByRole('textbox', { name: '对话输入' }), { target: { value: '创建工单' } })
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    expect(await screen.findByText('工单等待确认')).toBeInTheDocument()
    expect((screen.getByRole('textbox', { name: '工单内容' }) as HTMLTextAreaElement).value).toContain('现场核查')
    fireEvent.click(screen.getByRole('button', { name: '批准并创建' }))
    await waitFor(() => expect(api.chatResumeStream).toHaveBeenCalledWith(expect.objectContaining({ kind: 'work_order_approval', decision: 'approve' }), expect.any(Function), expect.any(AbortSignal)))
    expect(await screen.findByText('工单已创建。')).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '数据与报告' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '查看工单（1）' }))
    expect(screen.getByText('wo_1')).toBeInTheDocument()
    expect(screen.getByText('核查表具异常并恢复缺失数据')).toBeInTheDocument()
    expect(screen.getByText('核对表具参数')).toBeInTheDocument()
    expect(screen.getByText('复核采集链路')).toBeInTheDocument()
    expect(screen.getByText('metering')).toBeInTheDocument()
    expect(screen.getByText(/"run_id": "run-1"/)).toBeInTheDocument()
  })

  it('保留 Todo，但不展示工具执行流水和相关按钮', async () => {
    vi.mocked(api.chatTurnStream).mockImplementation(async (_threadId, _message, onEvent) => {
      onEvent({ event: 'todo', data: { items: [{ content: '查询每日用气', status: 'in_progress' }] } })
      onEvent({ event: 'tool_start', data: { tool_call_id: 't1', name: 'query_business_data' } })
      onEvent({ event: 'tool_end', data: { tool_call_id: 't1', name: 'query_business_data', status: 'success', elapsed_ms: 12, result: { query_id: 'q1', row_count: 3 } } })
      onEvent({ event: 'done', data: { status: 'completed', message: '查询完成。', generator: 'test', artifacts: [], todos: [{ content: '查询每日用气', status: 'completed' }] } })
    })
    render(<ChatPage />)
    fireEvent.change(screen.getByRole('textbox', { name: '对话输入' }), { target: { value: '查询' } })
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    expect(await screen.findByText('查询完成。')).toBeInTheDocument()
    expect(screen.getByText('查询每日用气')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /执行过程/ })).not.toBeInTheDocument()
    expect(screen.queryByText('工具完成 · query_business_data')).not.toBeInTheDocument()
  })

  it('使用 GFM 渲染报告而不是显示 Markdown 标记', async () => {
    const report = {
      type: 'report' as const, id: 'r1', payload: {
        type: 'report', report_id: 'r1', title: '月报', generated_at: '2025-01-31', summary: '摘要',
        report_markdown: '## 二级标题\n\n- **重点**\n\n<script>window.bad=true</script>', datasets: [], charts: [], recommendations: [],
      },
    }
    vi.mocked(api.chatTurnStream).mockImplementation(streamResponse({ status: 'completed', message: '报告完成', generator: 'test', artifacts: [report] }))
    render(<ChatPage />)
    fireEvent.change(screen.getByRole('textbox', { name: '对话输入' }), { target: { value: '生成报告' } })
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    expect(await screen.findByRole('heading', { name: '二级标题' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '查看报告（1）' })).toBeInTheDocument()
    expect(screen.queryByText('## 二级标题')).not.toBeInTheDocument()
    expect(document.querySelector('script')).toBeNull()
  })

  it('报告面板打开后分隔条支持键盘调整', async () => {
    const report = {
      type: 'report' as const, id: 'split-report', payload: {
        type: 'report', report_id: 'split-report', title: '分栏报告', generated_at: '2025-01-31', summary: '摘要',
        report_markdown: '正文', datasets: [], charts: [], recommendations: [],
      },
    }
    vi.mocked(api.chatTurnStream).mockImplementation(streamResponse({ status: 'completed', message: '报告完成', generator: 'test', artifacts: [report] }))
    render(<ChatPage />)
    await screen.findByText('对话 Agent 已就绪')
    expect(screen.queryByRole('separator', { name: '调整对话区和报告区宽度' })).not.toBeInTheDocument()
    fireEvent.change(screen.getByRole('textbox', { name: '对话输入' }), { target: { value: '生成报告' } })
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    const separator = await screen.findByRole('separator', { name: '调整对话区和报告区宽度' })
    expect(separator).toHaveAttribute('aria-valuenow', '44')
    fireEvent.keyDown(separator, { key: 'ArrowRight' })
    expect(separator).toHaveAttribute('aria-valuenow', '46')
  })

  it('不同回答只打开各自的产物，并按类型切换', async () => {
    const first: ChatTurnResponse = {
      status: 'completed', message: '第一轮完成', generator: 'test', artifacts: [
        { type: 'query_result', id: 'q-first', payload: { type: 'query_result', query_id: 'q-first', source: 'business', sql: 'SELECT 1', columns: [], rows: [], row_count: 0, truncated: false, elapsed_ms: 1 } },
        { type: 'report', id: 'r-first', payload: { type: 'report', report_id: 'r-first', title: '第一轮报告', generated_at: '2025-01-01', summary: '摘要', report_markdown: '第一轮正文', datasets: [], charts: [], recommendations: [] } },
      ],
    }
    const second: ChatTurnResponse = {
      status: 'completed', message: '第二轮完成', generator: 'test', artifacts: [
        { type: 'query_result', id: 'q-second', payload: { type: 'query_result', query_id: 'q-second', source: 'diagnosis', sql: 'SELECT 2', columns: [], rows: [], row_count: 0, truncated: false, elapsed_ms: 1 } },
      ],
    }
    vi.mocked(api.chatTurnStream).mockImplementationOnce(streamResponse(first)).mockImplementationOnce(streamResponse(second))
    render(<ChatPage />)
    const input = screen.getByRole('textbox', { name: '对话输入' })
    fireEvent.change(input, { target: { value: '第一轮' } }); fireEvent.click(screen.getByRole('button', { name: '发送' }))
    await screen.findByText('第一轮完成')
    fireEvent.click(screen.getByRole('button', { name: '隐藏' }))
    fireEvent.change(input, { target: { value: '第二轮' } }); fireEvent.click(screen.getByRole('button', { name: '发送' }))
    await screen.findByText('第二轮完成')
    const sqlButtons = screen.getAllByRole('button', { name: '查看 SQL 查询（1）' })
    fireEvent.click(sqlButtons[1])
    expect(screen.getByText('q-second')).toBeInTheDocument()
    expect(screen.queryByText('q-first')).not.toBeInTheDocument()
    expect(screen.getByRole('tab', { name: '报告 0' })).toBeDisabled()
  })

  it('状态栏不再显示临时会话和长期记忆文案', async () => {
    render(<ChatPage />)
    await screen.findByText('对话 Agent 已就绪')
    expect(screen.queryByText(/临时会话|不保存长期记忆/)).not.toBeInTheDocument()
    expect(screen.queryByText(/工单写入前需要人工确认/)).not.toBeInTheDocument()
    expect(document.querySelector('.chat-toolbar')?.textContent).not.toContain('新建对话')
  })

  it('流式回答期间显示三个依次跳动的状态点', async () => {
    let finish: (() => void) | undefined
    vi.mocked(api.chatTurnStream).mockImplementation(async (_threadId, _message, onEvent) => new Promise<void>((resolve) => {
      finish = () => {
        onEvent({ event: 'done', data: { status: 'completed', message: '回答完成。', generator: 'deepagents:deepseek:deepseek-v4-flash', artifacts: [] } })
        resolve()
      }
    }))
    render(<ChatPage />)
    fireEvent.change(screen.getByRole('textbox', { name: '对话输入' }), { target: { value: '你好' } })
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    const dots = await screen.findByLabelText('正在生成')
    expect(dots.querySelectorAll('i')).toHaveLength(3)
    await act(async () => finish?.())
    expect(await screen.findByText('deepseek-v4-flash')).toBeInTheDocument()
  })
})
