import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api'
import type { BusinessModule, DailyIssue } from '../types'
import { OverviewPage } from './OverviewPage'

vi.mock('../api', () => ({ api: { overview: vi.fn() } }))

function issue(module: BusinessModule, index: number): DailyIssue {
  return {
    module,
    user_id: `${module}-${index}`,
    company_name: `${module === 'metering' ? '计量' : '设备'}企业${index}`,
    risk_level: module === 'metering' ? '中' : '严重',
    issue_type: module === 'metering' ? '计量异常' : '设备异常',
    issue_tags: [module === 'metering' ? '计量异常' : '设备异常'],
    evidence_summary: '测试证据',
    primary_metric_name: '测试指标',
    primary_metric: index,
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api.overview).mockResolvedValue({
    diagnosis_date: '2025-01-12',
    status: 'completed',
    diagnosed_enterprises: 20,
    abnormal_enterprises: 15,
    normal_enterprises: 5,
    metering_issue_count: 7,
    equipment_issue_count: 8,
    high_risk_count: 10,
    risk_distribution: { 严重: 8, 高: 2, 中: 5, 较低: 0, 低: 0 },
    issue_type_distribution: [{ name: '设备异常', value: 8 }, { name: '计量异常', value: 7 }],
    issues: [
      ...Array.from({ length: 5 }, (_, index) => issue('metering', index)),
      ...Array.from({ length: 5 }, (_, index) => issue('equipment', index)),
    ],
  })
})

describe('总览模块前五切换', () => {
  it('默认展示智能计量并移除全部模块选项', async () => {
    render(<OverviewPage active date="2025-01-12" refreshToken={0} onBusyChange={vi.fn()} onOpenIssue={vi.fn()} />)

    const moduleSelect = await screen.findByRole('combobox', { name: '按模块筛选' })
    expect(moduleSelect).toHaveValue('metering')
    expect(screen.queryByRole('option', { name: '全部模块' })).not.toBeInTheDocument()
    expect(screen.getAllByText(/计量企业/)).toHaveLength(5)
    expect(screen.queryByText('设备企业0')).not.toBeInTheDocument()
    expect(screen.getByText('7')).toBeInTheDocument()
    expect(screen.getByText('8')).toBeInTheDocument()
  })

  it('切换设备后仅显示设备前五，搜索只作用于当前模块', async () => {
    render(<OverviewPage active date="2025-01-12" refreshToken={0} onBusyChange={vi.fn()} onOpenIssue={vi.fn()} />)

    const moduleSelect = await screen.findByRole('combobox', { name: '按模块筛选' })
    fireEvent.change(moduleSelect, { target: { value: 'equipment' } })
    expect(screen.getAllByText(/设备企业/)).toHaveLength(5)
    expect(screen.queryByText('计量企业0')).not.toBeInTheDocument()

    fireEvent.change(screen.getByRole('textbox', { name: '搜索异常企业' }), { target: { value: '设备企业3' } })
    expect(screen.getByText('设备企业3')).toBeInTheDocument()
    expect(screen.queryByText('设备企业2')).not.toBeInTheDocument()
  })
})
