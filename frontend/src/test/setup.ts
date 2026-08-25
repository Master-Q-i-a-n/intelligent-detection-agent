import '@testing-library/jest-dom/vitest'
import { createElement } from 'react'

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

Object.defineProperty(window, 'ResizeObserver', { value: ResizeObserverMock })
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false,
  }),
})
Object.defineProperty(URL, 'createObjectURL', { writable: true, value: vi.fn(() => 'blob:test-image') })
Object.defineProperty(URL, 'revokeObjectURL', { writable: true, value: vi.fn() })

vi.mock('echarts-for-react', () => ({
  default: ({ option }: { option: unknown }) => createElement('div', {
    'data-testid': 'echart',
    'data-option': JSON.stringify(option),
  }),
}))
