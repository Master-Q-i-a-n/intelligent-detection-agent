import { describe, expect, it } from 'vitest'
import { buildHorizontalBarOption, buildLineOption } from './Charts'

describe('图表边界数据', () => {
  it('保留 null 时序断点，不伪装为 0', () => {
    const option = buildLineOption(['00:00', '00:05', '00:10'], [{ name: '压力', data: [1.2, null, 1.3] }], '')
    const series = option.series as Array<{ data: Array<number | null>; connectNulls: boolean }>
    expect(series[0].data).toEqual([1.2, null, 1.3])
    expect(series[0].connectNulls).toBe(false)
  })

  it('全零模块分布仍使用有限坐标范围', () => {
    const option = buildHorizontalBarOption([{ name: '智能计量', value: 0 }, { name: '智能设备', value: 0 }])
    expect((option.xAxis as { max: number }).max).toBeGreaterThan(0)
  })
})
