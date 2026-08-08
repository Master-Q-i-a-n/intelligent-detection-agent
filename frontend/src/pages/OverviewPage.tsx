import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { Chart, buildHorizontalBarOption } from '../components/Charts'
import { DataTable, EmptyState, ErrorState, KpiGrid, LoadingState, Panel, formatNumber } from '../components/Common'
import type { DailyIssue, DailyOverview, PageKey } from '../types'

const riskColors: Record<string, string> = {
  严重: '#ff5263',
  高: '#ff755f',
  中: '#f2ad35',
  较低: '#3f91ff',
  低: '#32d7a0',
}

interface OverviewPageProps {
  active: boolean
  date: string
  refreshToken: number
  onBusyChange: (busy: boolean) => void
  onOpenIssue: (page: PageKey, userId: string) => void
}

export function OverviewPage({ active, date, refreshToken, onBusyChange, onOpenIssue }: OverviewPageProps) {
  const [data, setData] = useState<DailyOverview | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const [moduleFilter, setModuleFilter] = useState('all')
  const [riskFilter, setRiskFilter] = useState('all')

  useEffect(() => {
    if (!active) return
    if (!date) {
      setData(null)
      setError('没有可用检测日期，请先检查数据索引。')
      return
    }
    const controller = new AbortController()
    setLoading(true)
    setError('')
    onBusyChange(true)
    api.overview(date, controller.signal)
      .then(setData)
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
  }, [active, date, refreshToken, onBusyChange])

  const riskDistribution = useMemo(() => {
    const counts: Record<string, number> = { 严重: 0, 高: 0, 中: 0, 较低: 0 }
    data?.issues.forEach((issue) => { counts[issue.risk_level] = (counts[issue.risk_level] || 0) + 1 })
    return Object.entries(counts).map(([name, value]) => ({ name, value, color: riskColors[name] }))
  }, [data])

  const issueTypes = useMemo(() => {
    const counts = new Map<string, number>()
    data?.issues.forEach((issue) => counts.set(issue.issue_type, (counts.get(issue.issue_type) || 0) + 1))
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([name, value]) => ({ name, value, color: '#21d4d0' }))
  }, [data])

  const filteredIssues = useMemo(() => {
    const query = search.trim().toLowerCase()
    return (data?.issues || []).filter((issue) => {
      const matchesModule = moduleFilter === 'all' || issue.module === moduleFilter
      const matchesRisk = riskFilter === 'all' || issue.risk_level === riskFilter
      const content = `${issue.company_name} ${issue.user_id} ${issue.issue_tags.join(' ')} ${issue.issue_type}`.toLowerCase()
      return matchesModule && matchesRisk && (!query || content.includes(query))
    })
  }, [data, moduleFilter, riskFilter, search])

  if (loading && !data) return <LoadingState label="正在汇总每日巡检结果" />
  if (error && !data) return <ErrorState message={error} />
  if (!data) return <EmptyState title="尚无总览数据" detail="选择有效日期后执行每日诊断。" />

  const moduleData = [
    { name: '智能计量', value: data.metering_issue_count, color: '#3f91ff' },
    { name: '智能设备', value: data.equipment_issue_count, color: '#f2ad35' },
  ]
  const highRisk = data.issues.filter((issue) => ['严重', '高'].includes(issue.risk_level)).length

  return (
    <div className="page-stack">
      {error && <div className="inline-error" role="alert">刷新失败，当前仍显示上次结果：{error}</div>}
      <div className="section-heading">
        <div><small>DAILY AUTONOMOUS INSPECTION</small><h2>每日全量自诊断</h2></div>
        <span className="completion-pill">{data.diagnosis_date} · {data.status === 'completed' ? '已完成' : data.status}</span>
      </div>
      <KpiGrid items={[
        { label: '已自动诊断企业', value: data.diagnosed_enterprises, note: '当天全量用户', tone: 'cyan' },
        { label: '异常企业', value: data.abnormal_enterprises, note: `正常企业 ${data.normal_enterprises}`, tone: 'amber' },
        { label: '计量问题事件', value: data.metering_issue_count, note: '用气、量程与数据质量', tone: 'blue' },
        { label: '设备问题事件', value: data.equipment_issue_count, note: `高/严重问题 ${highRisk}`, tone: 'red' },
      ]} />

      <div className="overview-charts">
        <Panel title="问题模块分布" eyebrow="智能计量 / 智能设备">
          <Chart option={buildHorizontalBarOption(moduleData)} height={180} ariaLabel="问题模块分布横向柱状图" />
        </Panel>
        <Panel title="风险等级分布" eyebrow="按问题事件统计">
          <Chart option={buildHorizontalBarOption(riskDistribution)} height={180} ariaLabel="风险等级分布图" />
        </Panel>
        <Panel title="问题类型排行" eyebrow="高频异常自动聚合">
          <Chart
            option={buildHorizontalBarOption(issueTypes)}
            height={180}
            empty={!issueTypes.length}
            ariaLabel="问题类型排行图"
          />
        </Panel>
      </div>

      <Panel title="异常企业清单" eyebrow="点击记录进入对应企业证据页" className="table-panel">
        <div className="table-tools">
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="搜索企业 / 编号 / 问题"
            aria-label="搜索异常企业"
          />
          <select value={moduleFilter} onChange={(event) => setModuleFilter(event.target.value)} aria-label="按模块筛选">
            <option value="all">全部模块</option><option value="metering">智能计量</option><option value="equipment">智能设备</option>
          </select>
          <select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)} aria-label="按风险筛选">
            <option value="all">全部风险</option><option value="严重">严重</option><option value="高">高</option><option value="中">中</option><option value="较低">较低</option><option value="低">低</option>
          </select>
        </div>
        <DataTable minWidth={1120}>
          <thead><tr><th>风险</th><th>企业</th><th>问题模块</th><th>主要问题</th><th>问题证据摘要</th><th>关键指标</th><th>操作</th></tr></thead>
          <tbody>
            {filteredIssues.length ? filteredIssues.map((issue) => (
              <IssueRow key={`${issue.module}-${issue.user_id}-${issue.issue_type}`} issue={issue} onOpenIssue={onOpenIssue} />
            )) : <tr><td colSpan={7}><div className="table-empty">没有符合筛选条件的问题企业</div></td></tr>}
          </tbody>
        </DataTable>
      </Panel>
    </div>
  )
}

function IssueRow({ issue, onOpenIssue }: { issue: DailyIssue; onOpenIssue: OverviewPageProps['onOpenIssue'] }) {
  return (
    <tr>
      <td><span className="risk-label"><i style={{ background: riskColors[issue.risk_level] || '#7f9aac' }} />{issue.risk_level}</span></td>
      <td className="ellipsis-cell" title={`${issue.company_name} · ${issue.user_id}`}><b>{issue.company_name}</b><small>{issue.user_id}</small></td>
      <td><span className="module-badge">{issue.module === 'metering' ? '智能计量' : '智能设备'}</span></td>
      <td className="ellipsis-cell" title={issue.issue_tags.join('、')}>{issue.issue_tags.join('、') || issue.issue_type}</td>
      <td className="evidence-cell" title={issue.evidence_summary}>{issue.evidence_summary}</td>
      <td>{issue.primary_metric_name}：{formatNumber(issue.primary_metric)}</td>
      <td><button className="text-button" type="button" onClick={() => onOpenIssue(issue.module, issue.user_id)}>查看证据 →</button></td>
    </tr>
  )
}
