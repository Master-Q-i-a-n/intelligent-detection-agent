import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { DateCombobox, EnterpriseCombobox } from './Layout'

const users = [
  { user_id: '1626828', company_name: '江南大学' },
  { user_id: 'C-9001', company_name: '北方能源集团' },
]

function renderCombobox(onUserChange = vi.fn()) {
  render(
    <label>
      <span>检测企业</span>
      <EnterpriseCombobox users={users} userId="1626828" onUserChange={onUserChange} />
    </label>,
  )
  return { input: screen.getByRole('combobox', { name: '检测企业' }), onUserChange }
}

describe('企业搜索下拉', () => {
  it('支持按企业名称和编号片段过滤', () => {
    const { input } = renderCombobox()
    fireEvent.change(input, { target: { value: '能源' } })
    expect(screen.getByRole('option', { name: /北方能源集团/ })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: /江南大学/ })).not.toBeInTheDocument()

    fireEvent.change(input, { target: { value: '268' } })
    expect(screen.getByRole('option', { name: /江南大学/ })).toBeInTheDocument()
  })

  it('无匹配项时显示提示，Escape 恢复当前企业', () => {
    const { input } = renderCombobox()
    fireEvent.change(input, { target: { value: '不存在' } })
    expect(screen.getByText('没有匹配的企业')).toBeInTheDocument()
    fireEvent.keyDown(input, { key: 'Escape' })
    expect(input).toHaveValue('江南大学 · 1626828')
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument()
  })

  it('点击候选项后才提交企业切换', () => {
    const { input, onUserChange } = renderCombobox()
    fireEvent.change(input, { target: { value: '北方' } })
    expect(onUserChange).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('option', { name: /北方能源集团/ }))
    expect(onUserChange).toHaveBeenCalledWith('C-9001')
  })

  it('支持方向键与 Enter 选择，点击外部恢复未确认输入', () => {
    const { input, onUserChange } = renderCombobox()
    fireEvent.click(screen.getByRole('button', { name: '展开企业列表' }))
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(onUserChange).toHaveBeenCalledWith('C-9001')

    fireEvent.change(input, { target: { value: '未确认文字' } })
    fireEvent.pointerDown(document.body)
    expect(input).toHaveValue('江南大学 · 1626828')
  })

  it('通过键盘移出控件时恢复未确认输入', () => {
    const { input } = renderCombobox()
    fireEvent.change(input, { target: { value: '临时关键词' } })
    fireEvent.blur(input, { relatedTarget: document.body })
    expect(input).toHaveValue('江南大学 · 1626828')
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument()
  })
})

describe('日期搜索下拉', () => {
  it('支持输入日期片段筛选并在确认后切换', () => {
    const onDateChange = vi.fn()
    render(
      <label>
        <span>检测日期</span>
        <DateCombobox dates={['2025-01-11', '2025-01-12']} date="2025-01-12" onDateChange={onDateChange} />
      </label>,
    )
    const input = screen.getByRole('combobox', { name: '检测日期' })
    fireEvent.change(input, { target: { value: '01-11' } })
    expect(screen.getByRole('option', { name: '2025-01-11' })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: '2025-01-12' })).not.toBeInTheDocument()
    expect(onDateChange).not.toHaveBeenCalled()
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(onDateChange).toHaveBeenCalledWith('2025-01-11')
  })

  it('日期下拉支持展开选择和无结果提示', () => {
    const onDateChange = vi.fn()
    render(
      <label>
        <span>检测日期</span>
        <DateCombobox dates={['2025-01-11', '2025-01-12']} date="2025-01-12" onDateChange={onDateChange} />
      </label>,
    )
    fireEvent.click(screen.getByRole('button', { name: '展开日期列表' }))
    fireEvent.click(screen.getByRole('option', { name: '2025-01-11' }))
    expect(onDateChange).toHaveBeenCalledWith('2025-01-11')

    const input = screen.getByRole('combobox', { name: '检测日期' })
    fireEvent.change(input, { target: { value: '2099' } })
    expect(screen.getByText('没有匹配的日期')).toBeInTheDocument()
  })
})
