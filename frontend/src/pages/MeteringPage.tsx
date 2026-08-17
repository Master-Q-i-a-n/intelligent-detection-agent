import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { AutoAgentPanel, ManualAgentPanel, type AgentRecord, type AutoAgentRequest } from '../components/AgentControls'
import { Chart, buildLineOption, buildRiskGauge } from '../components/Charts'
import { DataTable, EmptyState, ErrorState, KpiGrid, LoadingState, Panel, PercentBars, formatNumber } from '../components/Common'
import type { MeteringDiagnosis, MeteringHistoryItem, MeteringSignals } from '../types'

interface MeteringPageProps {
  active: boolean
  userId: string
  date: string
  refreshToken: number
  autoAgentEnabled: boolean
  autoRecord?: AgentRecord
  onAutoRequest: (request: AutoAgentRequest) => void
  onBusyChange: (busy: boolean) => void
}

export function MeteringPage(props: MeteringPageProps) {
  const { active, userId, date, refreshToken, autoAgentEnabled, autoRecord, onAutoRequest, onBusyChange } = props
  const [diagnosis, setDiagnosis] = useState<MeteringDiagnosis | null>(null)
  const [history, setHistory] = useState<MeteringHistoryItem[]>([])
  const [signals, setSignals] = useState<MeteringSignals | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!active) return
    if (!userId || !date) {
      setDiagnosis(null)
      setError('企业或检测日期为空，未发起诊断请求。')
      return
    }
    const controller = new AbortController()
    setLoading(true)
    setError('')
    onBusyChange(true)
    Promise.all([
      api.meteringDiagnosis(userId, date, controller.signal),
      api.meteringHistory(userId, date, controller.signal),
      api.meteringSignals(userId, date, controller.signal),
    ])
      .then(([nextDiagnosis, nextHistory, nextSignals]) => {
        setDiagnosis(nextDiagnosis)
        setHistory(nextHistory.items || [])
        setSignals(nextSignals)
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

  const agentKey = `metering:${userId}:${date}`
  useEffect(() => {
    if (!active || !autoAgentEnabled || !diagnosis || diagnosis.user_id !== userId || diagnosis.diagnosis_date !== date) return
    onAutoRequest({
      key: agentKey,
      payload: {
        module: 'metering',
        user_id: userId,
        diagnosis_date: date,
        field_text: '',
      },
    })
  }, [active, agentKey, autoAgentEnabled, date, diagnosis, onAutoRequest, userId])

  if (loading && !diagnosis) return <LoadingState label="正在计算计量诊断与时序证据" />
  if (error && !diagnosis) return <ErrorState message={error} />
  if (!diagnosis) return <EmptyState title="尚无计量诊断" detail="请选择具有 SCADA 数据的企业和日期。" />

  const meterSpec = diagnosis.details?.meter_spec || {}
  const completeness = diagnosis.details?.data_quality?.pipeline_completeness || {}
  const siteResults = diagnosis.details?.site_results || []
  const completenessItems = siteResults.length > 1
    ? siteResults.flatMap((site) => Object.entries(site.pipeline_completeness || {})
      .filter(([, value]) => Number(value || 0) > 0)
      .map(([pipeline, value]) => ({
        label: `${site.site_name} · 管路 ${pipeline}`,
        value: Number(value || 0) * 100,
        tone: Number(value || 0) >= 0.9 ? 'green' as const : 'amber' as const,
        display: `${formatNumber(Number(value || 0) * 100)}%`,
      })))
    : Object.entries(completeness).map(([pipeline, value]) => ({
      label: `管路 ${pipeline}`,
      value: Number(value || 0) * 100,
      tone: Number(value || 0) >= 0.9 ? 'green' as const : 'amber' as const,
      display: `${formatNumber(Number(value || 0) * 100)}%`,
    }))
  const observed = Number(diagnosis.observed_volume || 0)
  const expected = Number(diagnosis.predicted_normal_volume || 0)
  const drop = expected > 0 ? Math.max(0, ((expected - observed) / expected) * 100) : 0
  const isUnmetered = diagnosis.model_gas_state === 1 && diagnosis.observed_gas_state === 0

  return (
    <div className="page-stack">
      {error && <div className="inline-error">部分数据刷新失败，当前显示上次结果：{error}</div>}
      <div className="section-heading"><div><small>METERING DIAGNOSTIC EVIDENCE</small><h2>智能计量详情</h2></div><span className={`risk-pill risk-${diagnosis.risk_level}`}>{diagnosis.risk_level}风险 · {diagnosis.status}</span></div>
      <KpiGrid items={[
        { label: '当日观测用气量', value: formatNumber(diagnosis.observed_volume), note: `${siteResults.length > 1 ? `${siteResults.length} 个厂区汇总 · ` : ''}SCADA 流量积分 · m³`, tone: 'cyan' },
        { label: '预测正常用气量', value: formatNumber(diagnosis.predicted_normal_volume), note: `历史基线 ${diagnosis.details?.baseline?.history_days || 0} 天 · m³`, tone: 'blue' },
        { label: '估算补气量', value: formatNumber(diagnosis.makeup_volume), note: `基线缺口 ${formatNumber(diagnosis.baseline_missing_volume)} + 表误差 ${formatNumber(diagnosis.meter_bias_volume)} m³`, tone: 'amber' },
        { label: '综合风险', value: diagnosis.risk_level, note: `风险评分 ${formatNumber(diagnosis.risk_score, 0)} / 100`, tone: diagnosis.risk_score >= 60 ? 'red' : diagnosis.risk_score >= 35 ? 'amber' : 'green' },
      ]} />

      {siteResults.length > 1 && <Panel title="厂区诊断汇总" eyebrow="企业统一展示，厂区独立计算，风险取最高值" className="table-panel">
        <DataTable minWidth={760}>
          <thead><tr><th>厂区</th><th>当日用气量</th><th>风险</th><th>诊断结论</th></tr></thead>
          <tbody>{siteResults.map((site) => (
            <tr key={site.site_name}>
              <td>{site.site_name}</td>
              <td>{formatNumber(site.observed_volume)} m³</td>
              <td>{site.risk_level} · {formatNumber(site.risk_score, 0)}</td>
              <td>{site.alerts.length ? site.alerts.join('；') : '未发现可推送异常'}</td>
            </tr>
          ))}</tbody>
        </DataTable>
      </Panel>}

      <Panel title="算法原始结论" eyebrow="不依赖 Agent，始终来自计量诊断流程" aside={<span className="algorithm-badge">ALGORITHM</span>}>
        <div className="mechanism-lead"><strong>核心结论</strong><p>{diagnosis.summary}</p></div>
        <div className="mechanism-grid">
          <MechanismStep index="01" title="基线差异" value={`${formatNumber(observed)} / ${formatNumber(expected)} m³`} text={`观测量较预测正常量低 ${formatNumber(drop, 1)}%。正常量来自历史同一 5 分钟时刻的中位数基线。`} />
          <MechanismStep index="02" title="状态交叉验证" value={isUnmetered ? '疑似走气未走字' : '存在远传计量'} text={isUnmetered ? '模型用气状态与远传状态冲突，需核查表体、脉冲和通信链路。' : '远传仍记录到流量；低于基线也可能由停产、减产或阀门状态造成。'} />
          <MechanismStep index="03" title="异常区间" value={`${diagnosis.anomaly_intervals.length} 个区间`} text="期望与观测偏差超过稳健阈值并连续至少两个采样点后，才形成异常区间。" />
          <MechanismStep index="04" title="损失量化" value={`${formatNumber(diagnosis.baseline_missing_volume)} m³`} text={`区间正偏差按 5 分钟积分；检定误差修正量为 ${formatNumber(diagnosis.meter_bias_volume)} m³。`} />
        </div>
        <div className="mechanism-caution"><b>使用边界：</b>估算补气量仅作为现场核查线索。在排除停产、阀门、通信和真实负荷变化前，不可直接作为结算量。</div>
      </Panel>

      <AutoAgentPanel enabled={autoAgentEnabled} record={autoRecord} />

      <div className="detail-grid two-one">
        <Panel title="近 7 天日用气量" eyebrow="日用气量变化与异常日定位" aside={<span className="unit-label">m³</span>}>
          <Chart
            option={buildLineOption(history.map((item) => item.date.slice(5)), [{ name: '日用气量', data: history.map((item) => item.volume), color: '#21d4d0' }], 'm³', { min: 0 })}
            height={285}
            empty={!history.length}
            ariaLabel="近七天日用气量曲线"
          />
        </Panel>
        <Panel title="近 7 天最大瞬时流量" eyebrow="独立纵轴，避免被日累计量压缩" aside={<span className="unit-label">m³/h</span>}>
          <Chart
            option={buildLineOption(history.map((item) => item.date.slice(5)), [{ name: '最大瞬时流量', data: history.map((item) => item.max_flow), color: '#f2ad35' }], 'm³/h', { min: 0 })}
            height={285}
            empty={!history.length}
            ariaLabel="近七天最大瞬时流量曲线"
          />
        </Panel>
        <Panel title="综合风险评分" eyebrow="规则、偏差和量程联合评分">
          <Chart option={buildRiskGauge(diagnosis.risk_score, diagnosis.risk_level)} height={285} ariaLabel="计量综合风险仪表盘" />
        </Panel>
      </div>

      <SignalPanels signals={signals} />

      <div className="detail-grid equal">
        <Panel title="表具量程适配" eyebrow="30 天运行区间分布">
          {meterSpec.quantity_max == null ? (
            <div className="range-missing"><strong>暂时无法判断表具量程是否适配</strong><p>用户表具档案缺少最大量程 quantity_max，因此不计算小流量、正常量程和超量程占比；0 不能视为实际读数。</p></div>
          ) : (
            <PercentBars items={[
              { label: '小流量区', value: Number(meterSpec.small_flow_percentage || 0), tone: 'blue', display: `${formatNumber(meterSpec.small_flow_percentage)}%` },
              { label: '正常量程', value: Number(meterSpec.normal_flow_percentage || 0), tone: 'green', display: `${formatNumber(meterSpec.normal_flow_percentage)}%` },
              { label: '超量程区', value: Number(meterSpec.over_flow_percentage || 0), tone: 'red', display: `${formatNumber(meterSpec.over_flow_percentage)}%` },
            ]} />
          )}
        </Panel>
        <Panel title="数据质量" eyebrow="各管路完整度">
          {completenessItems.length ? <PercentBars items={completenessItems} /> : <div className="panel-empty">没有管路完整度结果</div>}
        </Panel>
      </div>

      <Panel title="异常区间与证据链" eyebrow="定位时段、偏差与估算损失" className="table-panel">
        <DataTable minWidth={920}>
          <thead><tr><th>开始时间</th><th>结束时间</th><th>异常类型</th><th>观测值</th><th>期望值</th><th>估算补量</th><th>风险</th></tr></thead>
          <tbody>{diagnosis.anomaly_intervals.length ? diagnosis.anomaly_intervals.map((item, index) => (
            <tr key={`${item.start_time}-${index}`}><td>{displayTime(item.start_time)}</td><td>{displayTime(item.end_time)}</td><td>{item.anomaly_type || '计量偏差'}</td><td>{formatNumber(item.observed_value)}</td><td>{formatNumber(item.expected_value)}</td><td>{formatNumber(item.estimated_missing_volume)} m³</td><td>{item.severity || '—'}</td></tr>
          )) : <tr><td colSpan={7}><div className="table-empty">当天未定位到连续异常区间</div></td></tr>}</tbody>
        </DataTable>
      </Panel>

      <ManualAgentPanel module="metering" userId={userId} date={date} />
    </div>
  )
}

function MechanismStep({ index, title, value, text }: { index: string; title: string; value: string; text: string }) {
  return <article className="mechanism-step"><small>{index} · {title}</small><strong>{value}</strong><p>{text}</p></article>
}

function SignalPanels({ signals }: { signals: MeteringSignals | null }) {
  const entries = useMemo(() => Object.entries(signals?.pipelines || {}), [signals])
  const labels = signals?.times || []
  const colors = ['#21d4d0', '#f2ad35', '#3f91ff', '#a67cff']
  const definitions = (field: 'flow' | 'pressure' | 'temperature') => entries.map(([pipeline, values], index) => ({
    name: pipeline.includes(' / ') ? pipeline.replace(' / ', ' · 管道 ') : `管道 ${pipeline}`,
    data: values[field],
    color: colors[index % colors.length],
  }))
  return (
    <div className="signal-stack">
      <Panel title="当天各管路瞬时流量曲线" eyebrow="用于识别双管流量失衡、计数时长不符与疑似阻塞" aside={<span className="unit-label">m³/h</span>}>
        <Chart option={buildLineOption(labels, definitions('flow'), 'm³/h', { min: 0 })} height={310} empty={!entries.length} ariaLabel="各管路瞬时流量曲线" />
      </Panel>
      <div className="detail-grid equal">
        <Panel title="当天异常相关压力曲线" eyebrow="空值保留为断点，不补为零" aside={<span className="unit-label">压力</span>}>
          <Chart option={buildLineOption(labels, definitions('pressure'), '')} height={270} empty={!entries.length || entries.every(([, values]) => values.pressure.every((value) => value == null))} ariaLabel="各管路压力曲线" />
        </Panel>
        <Panel title="当天各管路温度曲线" eyebrow="温差比例与样本熵辅助诊断" aside={<span className="unit-label">℃</span>}>
          <Chart option={buildLineOption(labels, definitions('temperature'), '℃')} height={270} empty={!entries.length || entries.every(([, values]) => values.temperature.every((value) => value == null))} ariaLabel="各管路温度曲线" />
        </Panel>
      </div>
    </div>
  )
}

function displayTime(value: string): string {
  return String(value || '—').replace('T', ' ').slice(0, 16)
}
