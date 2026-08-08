import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

const backend = 'http://127.0.0.1:8000'

export default defineConfig(({ command }) => ({
  plugins: [react()],
  // 开发服务器使用根路径；生产资源由 FastAPI 挂载在 /static。
  base: command === 'build' ? '/static/' : '/',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    assetsDir: 'assets',
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: Object.fromEntries(
      ['/api', '/daily', '/metering', '/equipment', '/agent', '/users', '/health', '/security', '/internal'].map((prefix) => [
        prefix,
        { target: backend, changeOrigin: true },
      ]),
    ),
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.ts',
    css: true,
    exclude: ['e2e/**', 'node_modules/**', 'dist/**'],
  },
}))
