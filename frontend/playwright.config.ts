import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  outputDir: 'test-results',
  reporter: 'list',
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:5173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'desktop-1920', use: { ...devices['Desktop Chrome'], viewport: { width: 1920, height: 1080 } } },
    { name: 'desktop-1366', use: { ...devices['Desktop Chrome'], viewport: { width: 1366, height: 768 } } },
    { name: 'narrow-624', use: { ...devices['Desktop Chrome'], viewport: { width: 624, height: 1113 } } },
    { name: 'mobile-390', use: { ...devices['Desktop Chrome'], viewport: { width: 390, height: 844 } } },
  ],
})
