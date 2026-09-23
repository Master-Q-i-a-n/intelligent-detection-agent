import { lazy } from 'react'

// 总览先展示统计和列表，只有实际渲染图表时才下载图表引擎；问答报告共用该缓存。
export const LazyECharts = lazy(() => import('echarts-for-react'))
