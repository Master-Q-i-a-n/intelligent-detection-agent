import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App, { datesBetween } from './App'
import { api } from './api'

vi.mock('./api', () => ({
  api: {
    users: vi.fn(),
    overview: vi.fn(),
    meteringDiagnosis: vi.fn(),
    meteringHistory: vi.fn(),
    meteringSignals: vi.fn(),
    equipmentDashboard: vi.fn(),
    equipmentWaveform: vi.fn(),
    inspectAgent: vi.fn(),
    securityOverview: vi.fn(),
    securityEvents: vi.fn(),
    securityEvent: vi.fn(),
    securityAction: vi.fn(),
  },
}))

const diagnosis = {
  run_id: 'r1', user_id: 'u1', user_name: '测试企业', diagnosis_date: '2025-01-12', status: 'completed',
  quality_status: 0, model_gas_state: 1, observed_gas_state: 1, observed_volume: 100,
  predicted_normal_volume: 120, baseline_missing_volume: 20, meter_bias_volume: 1, makeup_volume: 21,
  risk_score: 42, risk_level: '中', meter_spec_result: '量程适配', summary: '存在计量偏差', alerts: [],
  anomaly_intervals: [], work_order: null,
  details: { baseline: { history_days: 7 }, data_quality: { pipeline_completeness: { 1: 0.9 } }, meter_spec: { quantity_max: 200, normal_flow_percentage: 100 } },
}

function prepareApi(range: [string, string] | [] = ['2025-01-11', '2025-01-12']) {
  vi.mocked(api.users).mockResolvedValue({
    items: [
      { user_id: 'u1', company_name: '测试企业一', date_range: range as [string, string] },
      { user_id: 'u2', company_name: '测试企业二', date_range: range as [string, string] },
    ],
    count: 2, enterprise_count: 2, date_range: range,
  })
  vi.mocked(api.overview).mockResolvedValue({
    diagnosis_date: '2025-01-12', status: 'completed', diagnosed_enterprises: 1, abnormal_enterprises: 0,
    normal_enterprises: 1, metering_issue_count: 0, equipment_issue_count: 0, issues: [],
  })
  vi.mocked(api.meteringDiagnosis).mockImplementation(async (requestedUserId, date) => ({ ...diagnosis, user_id: requestedUserId, diagnosis_date: date }))
  vi.mocked(api.meteringHistory).mockResolvedValue({ user_id: 'u1', items: [] })
  vi.mocked(api.meteringSignals).mockResolvedValue({ user_id: 'u1', diagnosis_date: '2025-01-12', times: [], resample_frequency: '5min', point_count: 0, pipelines: {} })
  vi.mocked(api.inspectAgent).mockResolvedValue({ generator: 'llm', conclusion: '检查完成' })
  vi.mocked(api.securityOverview).mockResolvedValue({ total: 0, confirmed: 0, review_required: 0, new_count: 0, processing: 0, high_risk: 0, latest_sequence: 0 })
  vi.mocked(api.securityEvents).mockResolvedValue({ items: [] })
}

beforeEach(() => {
  vi.clearAllMocks()
  prepareApi()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Agent 自动调用控制', () => {
  it('初始挂载和重新挂载时开关都关闭', async () => {
    const first = render(<App />)
    const firstSwitch = await screen.findByRole('switch', { name: 'Agent 自动解读' })
    expect(firstSwitch).toHaveAttribute('aria-checked', 'false')
    first.unmount()
    render(<App />)
    expect(await screen.findByRole('switch', { name: 'Agent 自动解读' })).toHaveAttribute('aria-checked', 'false')
  })

  it('关闭状态进入计量详情不会调用 Agent，开启后当前键仅调用一次', async () => {
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: /智能计量详情/ }))
    expect(await screen.findByText('存在计量偏差')).toBeInTheDocument()
    expect(api.inspectAgent).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('switch', { name: 'Agent 自动解读' }))
    await waitFor(() => expect(api.inspectAgent).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole('button', { name: '刷新诊断证据' }))
    await waitFor(() => expect(api.meteringDiagnosis).toHaveBeenCalledTimes(2))
    expect(api.inspectAgent).toHaveBeenCalledTimes(1)
  })

  it('开启后切换日期对新调用键自动请求一次', async () => {
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: /智能计量详情/ }))
    await screen.findByText('存在计量偏差')
    fireEvent.click(screen.getByRole('switch', { name: 'Agent 自动解读' }))
    await waitFor(() => expect(api.inspectAgent).toHaveBeenCalledTimes(1))
    fireEvent.change(screen.getByLabelText('检测日期'), { target: { value: '2025-01-11' } })
    await waitFor(() => expect(api.inspectAgent).toHaveBeenCalledTimes(2))
  })

  it('开启后切换企业对新调用键自动请求一次', async () => {
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: /智能计量详情/ }))
    await screen.findByText('存在计量偏差')
    fireEvent.click(screen.getByRole('switch', { name: 'Agent 自动解读' }))
    await waitFor(() => expect(api.inspectAgent).toHaveBeenCalledTimes(1))
    fireEvent.change(screen.getByLabelText('检测企业'), { target: { value: 'u2' } })
    await waitFor(() => expect(api.inspectAgent).toHaveBeenCalledTimes(2))
  })

  it('关闭状态手动生成仍会调用 Agent', async () => {
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: /智能计量详情/ }))
    await screen.findByText('存在计量偏差')
    fireEvent.click(screen.getByRole('button', { name: '生成智能检查结果' }))
    await waitFor(() => expect(api.inspectAgent).toHaveBeenCalledTimes(1))
    expect(screen.getByRole('switch', { name: 'Agent 自动解读' })).toHaveAttribute('aria-checked', 'false')
  })
})

describe('日期和图表边界', () => {
  it('空日期不产生总览请求并给出可操作错误', async () => {
    prepareApi([])
    render(<App />)
    expect(await screen.findByText('无法进入检测流程')).toBeInTheDocument()
    expect(api.overview).not.toHaveBeenCalled()
  })

  it('日期生成包含首尾且拒绝无效范围', () => {
    expect(datesBetween(['2025-01-11', '2025-01-12'])).toEqual(['2025-01-11', '2025-01-12'])
    expect(datesBetween(['2025-01-12', '2025-01-11'])).toEqual([])
  })
})
