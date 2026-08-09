import { expect, test } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.route('**/api/users', (route) => route.fulfill({ json: { items: [{ user_id: 'u1', company_name: '测试企业', date_range: ['2025-01-11', '2025-01-12'] }], count: 1, enterprise_count: 1, date_range: ['2025-01-11', '2025-01-12'] } }))
  await page.route('**/daily/overview/*', (route) => route.fulfill({ json: { diagnosis_date: '2025-01-12', status: 'completed', diagnosed_enterprises: 715, abnormal_enterprises: 5, normal_enterprises: 710, metering_issue_count: 1, equipment_issue_count: 4, issues: [] } }))
  await page.goto('/')
})

test('侧栏控制卡、紧凑图表和页面宽度稳定', async ({ page }) => {
  await expect(page.getByRole('switch', { name: 'Agent 自动解读' })).toBeVisible()
  await expect(page.getByRole('switch', { name: 'Agent 自动解读' })).toHaveAttribute('aria-checked', 'false')
  await expect(page.getByText('问题模块分布')).toBeVisible()

  const viewportWidth = await page.evaluate(() => document.documentElement.clientWidth)
  const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth)
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1)
})

test('问答分栏、执行计划和 Markdown 报告正确显示', async ({ page }) => {
  await page.route('**/chat/status', (route) => route.fulfill({ json: { configured: true, provider: 'deepseek', model: 'deepseek-chat', memory: 'in-process-thread-only', tracing_enabled: false } }))
  const report = {
    type: 'report', id: 'rpt_e2e', payload: {
      type: 'report', report_id: 'rpt_e2e', title: '测试用气报告', generated_at: '2025-01-12 12:00:00', summary: '测试摘要',
      report_markdown: '## 数据结论\n\n- **峰值正常**\n\n| 日期 | 用气量 |\n| --- | ---: |\n| 2025-01-12 | 1200 |',
      datasets: [], charts: [], recommendations: ['继续观察'],
    },
  }
  const query = {
    type: 'query_result', id: 'q_e2e', payload: {
      type: 'query_result', query_id: 'q_e2e', source: 'business', sql: 'SELECT usage_date, volume_m3 FROM telemetry.scada_observation',
      columns: ['usage_date', 'volume_m3'], rows: [{ usage_date: '2025-01-12', volume_m3: 1200 }], row_count: 1, truncated: false, elapsed_ms: 18,
    },
  }
  const stream = [
    ['meta', { run_id: 'run_e2e', thread_id: 'thread_e2e' }],
    ['todo', { items: [{ content: '查询用气数据', status: 'in_progress' }] }],
    ['tool_start', { tool_call_id: 'tool_e2e', name: 'query_business_data' }],
    ['tool_end', { tool_call_id: 'tool_e2e', name: 'query_business_data', status: 'success', elapsed_ms: 18, result: { query_id: 'q_e2e', row_count: 1 } }],
    ['done', { status: 'completed', message: '报告已经生成。', generator: 'e2e', artifacts: [report, query], todos: [{ content: '查询用气数据', status: 'completed' }], interrupt: null }],
  ].map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join('')
  await page.route('**/chat/turns/stream', (route) => route.fulfill({ status: 200, contentType: 'text/event-stream', body: stream }))

  await page.getByRole('button', { name: /智能问答/ }).click()
  await expect(page.getByLabel('数据与报告')).toHaveCount(0)
  await expect(page.getByText(/临时会话|不保存长期记忆/)).toHaveCount(0)
  await page.getByRole('textbox', { name: '对话输入' }).fill('生成测试报告')
  await page.getByRole('button', { name: '发送' }).click()
  await expect(page.getByRole('heading', { name: '数据结论' })).toBeVisible()
  await expect(page.getByText('## 数据结论')).toHaveCount(0)
  await expect(page.getByText('查询用气数据')).toBeVisible()
  await expect(page.getByRole('button', { name: /执行过程/ })).toHaveCount(0)
  await expect(page.getByText('工具完成 · query_business_data')).toHaveCount(0)

  const userMessage = page.locator('.chat-message.user').last()
  const userBubble = userMessage.locator('.chat-message-bubble')
  const userMark = userMessage.locator(':scope > span')
  const [bubbleBox, markBox, messageBox] = await Promise.all([userBubble.boundingBox(), userMark.boundingBox(), userMessage.boundingBox()])
  expect(bubbleBox).not.toBeNull(); expect(markBox).not.toBeNull(); expect(messageBox).not.toBeNull()
  expect(bubbleBox!.x).toBeLessThan(markBox!.x)
  expect(bubbleBox!.width).toBeLessThan(messageBox!.width * .8)

  const separator = page.getByRole('separator', { name: '调整对话区和报告区宽度' })
  if ((page.viewportSize()?.width || 0) > 1050) {
    await expect(separator).toBeVisible()
    await separator.focus()
    await separator.press('ArrowRight')
    await expect(separator).toHaveAttribute('aria-valuenow', '46')
  } else {
    await expect(separator).toBeHidden()
    const panelBox = await page.getByRole('dialog', { name: '数据与报告' }).boundingBox()
    expect(panelBox).not.toBeNull()
    expect(panelBox!.width).toBe(page.viewportSize()!.width)
    expect(panelBox!.height).toBe(page.viewportSize()!.height)
  }

  await page.getByRole('button', { name: '隐藏' }).click()
  const openReport = page.getByRole('button', { name: '查看报告（1）' })
  await openReport.click()
  await page.getByRole('button', { name: '隐藏' }).click()
  await expect(openReport).toBeFocused()
  await page.getByRole('button', { name: '查看 SQL 查询（1）' }).click()
  await expect(page.getByText('q_e2e')).toBeVisible()
  await expect(page.getByRole('heading', { name: '数据结论' })).toHaveCount(0)

  const pageWidth = await page.evaluate(() => ({ client: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth }))
  expect(pageWidth.scroll).toBeLessThanOrEqual(pageWidth.client + 1)
})
