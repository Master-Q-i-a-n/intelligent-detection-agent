import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'
import { requestJson } from '../api'
import { KnowledgePage } from './KnowledgePage'

vi.mock('../api', () => ({ requestJson: vi.fn() }))
const item = { id: 'doc_test', name: '设备手册.pdf', status: 'failed', stage: 'describe', start_page: 9,
  end_page: 12, total_pages: 20, image_count: 4, image_done: 3, chunk_count: 0, indexed_count: 0,
  created_at: '2026-09-09T00:00:00Z', error: '1 张图片识别失败', image_errors: [{ name: 'fig_003', error: '超时' }] }
beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(requestJson).mockImplementation(async (path, options) => {
    if (options?.method === 'POST') return { ...item, duplicate: false }
    return { items: [item], worker_error: '', parser_available: true, upload_limit_mb: 100 }
  })
})

it('上传连续页码范围与 OCR 选项', async () => {
  render(<KnowledgePage />)
  await screen.findByText('设备手册.pdf')
  fireEvent.change(screen.getByLabelText(/选择 PDF 文件/), { target: { files: [new File(['pdf'], '新文档.pdf', { type: 'application/pdf' })] } })
  fireEvent.click(screen.getByLabelText('指定页码范围'))
  fireEvent.change(screen.getByLabelText('起始页'), { target: { value: '9' } })
  fireEvent.change(screen.getByLabelText('结束页'), { target: { value: '12' } })
  fireEvent.click(screen.getByLabelText('强制 OCR'))
  fireEvent.click(screen.getByRole('button', { name: '上传并处理' }))
  await waitFor(() => expect(requestJson).toHaveBeenCalledWith('/rag/documents', expect.objectContaining({ method: 'POST' })))
  const options = vi.mocked(requestJson).mock.calls.find(([, options]) => options?.method === 'POST')?.[1]
  const form = options?.body as FormData
  expect(form.get('start_page')).toBe('9')
  expect(form.get('end_page')).toBe('12')
  expect(form.get('force_ocr')).toBe('true')
})

it('删除先确认，失败详情和重试入口可见', async () => {
  render(<KnowledgePage />)
  fireEvent.click(await screen.findByRole('button', { name: '详情' }))
  expect(screen.getByText('fig_003：超时')).toBeInTheDocument()
  expect(screen.getByRole('link', { name: '查看原 PDF ↗' })).toHaveAttribute('href', '/rag/documents/doc_test/file#page=9')
  fireEvent.click(screen.getByRole('button', { name: '删除' }))
  const dialog = screen.getByRole('dialog')
  expect(requestJson).not.toHaveBeenCalledWith('/rag/documents/doc_test', { method: 'DELETE' })
  fireEvent.click(within(dialog).getByRole('button', { name: '确认删除' }))
  await waitFor(() => expect(requestJson).toHaveBeenCalledWith('/rag/documents/doc_test', { method: 'DELETE' }))
})

it('全文上传不发送页码，重复上传显示提示', async () => {
  vi.mocked(requestJson).mockImplementation(async (_, options) => options?.method === 'POST'
    ? { ...item, duplicate: true } : { items: [item], parser_available: true, upload_limit_mb: 100 })
  render(<KnowledgePage />)
  fireEvent.change(screen.getByLabelText(/选择 PDF 文件/), { target: { files: [new File(['pdf'], 'a.pdf')] } })
  fireEvent.click(screen.getByRole('button', { name: '上传并处理' }))
  expect(await screen.findByText(/相同页码范围已存在/)).toBeInTheDocument()
  const options = vi.mocked(requestJson).mock.calls.find(([, options]) => options?.method === 'POST')?.[1]
  expect((options?.body as FormData).has('start_page')).toBe(false)
})
