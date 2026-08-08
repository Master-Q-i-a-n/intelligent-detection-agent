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
