import ReactECharts from 'echarts-for-react'
import type { EChartsOption, LineSeriesOption } from 'echarts'

const chartText = '#7f9aac'
const chartGrid = '#1b3444'
const palette = ['#21d4d0', '#f2ad35', '#3f91ff', '#a67cff', '#ff5263']

interface ChartProps {
  option: EChartsOption
  height?: number
  empty?: boolean
  emptyText?: string
  ariaLabel: string
}

export function Chart({ option, height = 280, empty = false, emptyText = '暂无可展示数据', ariaLabel }: ChartProps) {
  if (empty) {
    return (
      <div className="chart-empty" style={{ height }} role="status">
        <span className="empty-radar" aria-hidden="true" />
        <strong>{emptyText}</strong>
        <small>切换企业或日期后重试</small>
      </div>
    )
  }
  return (
    <div role="img" aria-label={ariaLabel} className="chart-frame" style={{ height }}>
      <ReactECharts
        option={option}
        notMerge
        lazyUpdate
        style={{ width: '100%', height: '100%' }}
        opts={{ renderer: 'canvas' }}
      />
    </div>
  )
}

export interface LineDefinition {
  name: string
  data: Array<number | null>
  color?: string
  dashed?: boolean
}

export function buildLineOption(
  labels: Array<string | number>,
  lines: LineDefinition[],
  unit: string,
  bounds?: { min?: number; max?: number },
): EChartsOption {
  const series: LineSeriesOption[] = lines.map((line, index) => ({
    name: line.name,
    type: 'line',
    data: line.data,
    showSymbol: false,
    connectNulls: false,
    smooth: false,
    sampling: 'lttb',
    lineStyle: { width: 2, type: line.dashed ? 'dashed' : 'solid', color: line.color ?? palette[index] },
    itemStyle: { color: line.color ?? palette[index] },
    emphasis: { focus: 'series' },
  }))

  return {
    animationDuration: 450,
    color: palette,
    grid: { top: 42, right: 22, bottom: 38, left: 58, containLabel: false },
    legend: { top: 4, left: 4, textStyle: { color: chartText }, icon: 'roundRect', itemWidth: 16, itemHeight: 3 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#0b1d28ee',
      borderColor: '#24475b',
      textStyle: { color: '#dcecf5' },
      valueFormatter: (value) => `${value ?? '—'}${unit ? ` ${unit}` : ''}`,
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: labels,
      axisLine: { lineStyle: { color: chartGrid } },
      axisTick: { show: false },
      axisLabel: { color: chartText, hideOverlap: true },
    },
    yAxis: {
      type: 'value',
      min: bounds?.min,
      max: bounds?.max,
      name: unit,
      nameTextStyle: { color: chartText, align: 'right' },
      splitLine: { lineStyle: { color: chartGrid, type: 'dashed' } },
      axisLabel: { color: chartText },
    },
    series,
  }
}

export function buildHorizontalBarOption(
  data: Array<{ name: string; value: number; color?: string }>,
  unit = '',
): EChartsOption {
  const maxValue = Math.max(...data.map((item) => item.value), 1)
  return {
    animationDuration: 450,
    grid: { top: 10, right: 48, bottom: 12, left: 94 },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: '#0b1d28ee',
      borderColor: '#24475b',
      textStyle: { color: '#dcecf5' },
      valueFormatter: (value) => `${value ?? 0}${unit}`,
    },
    xAxis: {
      type: 'value',
      max: Math.ceil(maxValue * 1.15),
      show: false,
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: data.map((item) => item.name),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: chartText, width: 86, overflow: 'truncate' },
    },
    series: [
      {
        type: 'bar',
        data: data.map((item, index) => ({
          value: item.value,
          itemStyle: { color: item.color ?? palette[index % palette.length], borderRadius: [0, 2, 2, 0] },
        })),
        barWidth: 11,
        showBackground: true,
        backgroundStyle: { color: '#07141d' },
        label: { show: true, position: 'right', color: '#bcd0dc', formatter: `{c}${unit}` },
      },
    ],
  }
}

export function buildRiskGauge(score: number, label: string): EChartsOption {
  const color = score >= 60 ? '#ff5263' : score >= 35 ? '#f2ad35' : '#32d7a0'
  return {
    series: [
      {
        type: 'gauge',
        startAngle: 210,
        endAngle: -30,
        min: 0,
        max: 100,
        splitNumber: 5,
        radius: '92%',
        progress: { show: true, roundCap: true, width: 13, itemStyle: { color } },
        axisLine: { lineStyle: { width: 13, color: [[1, '#102a38']] } },
        pointer: { show: false },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: { show: false },
        anchor: { show: false },
        title: { show: true, offsetCenter: [0, '38%'], color: chartText, fontSize: 13 },
        detail: {
          valueAnimation: true,
          offsetCenter: [0, '-4%'],
          color: '#ecf7fb',
          fontFamily: 'Bahnschrift, DIN Alternate, sans-serif',
          fontSize: 34,
          formatter: '{value}',
        },
        data: [{ value: Math.max(0, Math.min(100, score)), name: `${label}风险` }],
      },
    ],
  }
}
