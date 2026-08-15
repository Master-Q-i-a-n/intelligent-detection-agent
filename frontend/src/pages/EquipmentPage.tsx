import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { AutoAgentPanel, ManualAgentPanel, type AgentRecord, type AutoAgentRequest } from '../components/AgentControls'
import { Chart, buildLineOption } from '../components/Charts'
import { DataTable, EmptyState, ErrorState, KpiGrid, LoadingState, Panel, PercentBars, formatNumber } from '../components/Common'
import type { EquipmentDashboard, EquipmentWaveform } from '../types'

const stageColors: Record<string, string> = {
  H0: '#32d7a0', H1: '#21d4d0', H2: '#3f91ff', H3: '#f2ad35', H4: '#ff5263',
}
const stageNames = ['健康', '轻衰减', '性能衰减', '严重衰减', '故障']

interface EquipmentPageProps {
  active: boolean
  userId: string
  date: string
  refreshToken: number
  autoAgentEnabled: boolean
  autoRecord?: AgentRecord
  onAutoRequest: (request: AutoAgentRequest) => void
  onBusyChange: (busy: boolean) => void
}

export function EquipmentPage(props: EquipmentPageProps) {
  const { active, userId, date, refreshToken, autoAgentEnabled, autoRecord, onAutoRequest, onBusyChange } = props
  const [dashboard, setDashboard] = useState<EquipmentDashboard | null>(null)
  const [waveform, setWaveform] = useState<EquipmentWaveform | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!active) return
    if (!userId || !date) {
      setDashboard(null)
      setError('企业或检测日期为空，未发起设备诊断请求。')
      return
    }
    const controller = new AbortController()
    setLoading(true)
    setError('')
    onBusyChange(true)
    Promise.all([
      api.equipmentDashboard(userId, date, controller.signal),
      api.equipmentWaveform(userId, date, controller.signal),
    ])
      .then(([nextDashboard, nextWaveform]) => {
        setDashboard(nextDashboard)
        setWaveform(nextWaveform)
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '未知错误')
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false)
          onBusyChange(false)
        }
      })
    return () => {
      controller.abort()
      onBusyChange(false)
    }
  }, [active, userId, date, refreshToken, onBusyChange])

  const agentKey = `equipment:${userId}:${date}`
  useEffect(() => {
    if (!active || !autoAgentEnabled || !dashboard || dashboard.current_assessment.date !== date) return
    onAutoRequest({
      key: agentKey,
      payload: {
        module: 'equipment', user_id: userId, diagnosis_date: date, field_text: '',
      },
    })
  }, [active, agentKey, autoAgentEnabled, dashboard, date, onAutoRequest, userId])

  const recent = useMemo(() => dashboard?.daily_history.filter((item) => item.date <= date).slice(-7) || [], [dashboard, date])

  if (loading && !dashboard) return <LoadingState label="正在读取设备健康与振动证据" />
  if (error && !dashboard) return <ErrorState message={error} />
  if (!dashboard) return <EmptyState title="尚无设备诊断" detail="请选择具有振动监测数据的企业和日期。" />

  const current = dashboard.current_assessment
  const trend = dashboard.trend_assessment
  const explanation = dashboard.model_explanation
  const healthTone = current.health_index < 50 ? 'red' : current.health_index < 75 ? 'amber' : 'green'

  return (
    <div className="page-stack">
      {error && <div className="inline-error">部分数据刷新失败，当前显示上次结果：{error}</div>}
      <div className="section-heading"><div><small>EQUIPMENT HEALTH EVIDENCE</small><h2>智能设备详情</h2></div><span className="completion-pill">{current.date} · {current.risk_level}</span></div>
      <KpiGrid items={[
        { label: '健康阶段', value: current.stage, note: current.stage_name, tone: current.stage === 'H4' ? 'red' : current.stage === 'H3' ? 'amber' : 'cyan' },
        { label: '健康指数', value: formatNumber(current.health_index), note: '连续健康度 · HI / 100', tone: healthTone },
        { label: '模型置信度', value: `${formatNumber(current.confidence * 100)}%`, note: '五阶段最大后验概率', tone: 'blue' },
        { label: '演化趋势', value: trend.trend_name, note: `斜率 ${formatNumber(trend.daily_slope, 2)} 点/日`, tone: trend.daily_slope < -0.35 ? 'amber' : 'green' },
      ]} />

      <Panel title="算法原始结论" eyebrow="多尺度形态学、三轴融合与时序约束结果" aside={<span className="algorithm-badge">ALGORITHM</span>}>
        <div className="equipment-conclusion">
          <div className="stage-emblem" style={{ '--stage-color': stageColors[current.stage] || '#21d4d0' } as React.CSSProperties}>
            <span>{current.stage}</span><small>{current.stage_name}</small>
          </div>
          <div><strong>{current.risk_level} · {trend.trend_name}</strong><p>{dashboard.recommended_action}</p><small>最近 {trend.window_days} 天健康指数由 {formatNumber(trend.health_index_start)} 变化至 {formatNumber(trend.health_index_end)}，最大单日下降 {formatNumber(trend.maximum_daily_drop)} 点。</small></div>
        </div>
      </Panel>

      <AutoAgentPanel enabled={autoAgentEnabled} record={autoRecord} />

      <div className="detail-grid equal">
        <Panel title="近 7 天健康指数" eyebrow="时序稳定结果与原始模型对照" aside={<span className="unit-label">HI / 100</span>}>
          <Chart option={buildLineOption(recent.map((item) => item.date.slice(5)), [
            { name: '健康指数', data: recent.map((item) => item.stabilized_health_index ?? item.predicted_health_index), color: '#21d4d0' },
            { name: '原始模型', data: recent.map((item) => item.predicted_health_index), color: '#526e81', dashed: true },
          ], 'HI', { min: 0, max: 100 })} height={300} empty={!recent.length} ariaLabel="近七天设备健康指数曲线" />
        </Panel>
        <Panel title="五阶段后验概率" eyebrow="H0 健康稳定 → H4 故障异常">
          <PercentBars items={Object.entries(current.probabilities).map(([stage, probability]) => ({ label: `${stage} ${stageNames[Number(stage.slice(1))] || ''}`, value: probability * 100, display: `${formatNumber(probability * 100)}%`, tone: stage.toLowerCase() }))} />
        </Panel>
      </div>

      <Panel title="当天三轴振动波形" eyebrow={`后盖 X / Y / Z · 采样率 ${waveform?.sampling_rate_hz || 0} Hz`} aside={<span className="unit-label">加速度</span>}>
        <Chart
          option={buildLineOption(waveform?.indices || [], [
            { name: 'X 轴', data: waveform?.x || [], color: '#21d4d0' },
            { name: 'Y 轴', data: waveform?.y || [], color: '#f2ad35' },
            { name: 'Z 轴', data: waveform?.z || [], color: '#a67cff' },
          ], '')}
          height={330}
          empty={!waveform?.indices.length}
          ariaLabel="当天三轴振动波形"
        />
      </Panel>

      <div className="detail-grid equal">
        <Panel title="三轴自适应权重" eyebrow="模型对各方向振动响应的关注度">
          <PercentBars items={['X', 'Y', 'Z'].map((axis, index) => ({ label: `${axis} 轴`, value: Number(explanation.axis_weights[index] || 0) * 100, display: `${formatNumber(Number(explanation.axis_weights[index] || 0) * 100)}%`, tone: ['cyan', 'amber', 'violet'][index] }))} />
        </Panel>
        <Panel title="形态学尺度权重" eyebrow="3 / 5 / 9 / 17 尺度冲击与包络特征">
          <PercentBars items={explanation.scale_kernel_sizes.map((kernel, index) => ({ label: `k = ${kernel}`, value: Number(explanation.morphological_scale_weights[index] || 0) * 100, display: `${formatNumber(Number(explanation.morphological_scale_weights[index] || 0) * 100)}%`, tone: 'blue' }))} />
        </Panel>
      </div>

      <Panel title="状态演化路径" eyebrow="阶段迁移与当前趋势">
        <div className="stage-timeline">
          {(trend.stage_sequence || []).map((stage, index) => <div className="timeline-node" key={`${stage}-${index}`} style={{ '--stage-color': stageColors[stage] || '#21d4d0' } as React.CSSProperties}><i /><b>{stage}</b><small>{index ? '状态迁移' : '起始阶段'}</small></div>)}
          <div className="timeline-node trend-node"><i /><b>{trend.trend_name}</b><small>最大下降 {formatNumber(trend.maximum_daily_drop)} 点</small></div>
        </div>
      </Panel>

      <Panel title="近 7 天诊断明细" eyebrow="工况、阶段、健康指数与置信度" className="table-panel">
        <DataTable minWidth={850}>
          <thead><tr><th>日期</th><th>稳定阶段</th><th>健康指数</th><th>置信度</th><th>工况</th><th>原始模型</th></tr></thead>
          <tbody>{recent.map((item) => <tr key={item.date}><td>{item.date}</td><td><span className="stage-badge" style={{ color: stageColors[item.temporal_stage || item.stabilized_stage || item.predicted_stage] }}>{item.temporal_stage || item.stabilized_stage || item.predicted_stage}</span></td><td>{formatNumber(item.stabilized_health_index ?? item.predicted_health_index)}</td><td>{formatNumber(item.confidence * 100)}%</td><td>{formatNumber(item.operating_condition, 0)} m³/h</td><td>{item.predicted_stage} / {formatNumber(item.predicted_health_index)}</td></tr>)}</tbody>
        </DataTable>
      </Panel>

      <ManualAgentPanel module="equipment" userId={userId} date={date} />
    </div>
  )
}
