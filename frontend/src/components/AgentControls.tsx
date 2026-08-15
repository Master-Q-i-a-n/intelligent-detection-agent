import { useEffect, useState } from 'react'
import { api } from '../api'
import type { AgentInspectionPayload, AgentReport, BusinessModule } from '../types'
import { AgentReportView, Panel } from './Common'

export interface AgentRecord {
  status: 'idle' | 'loading' | 'success' | 'error'
  report?: AgentReport
  error?: string
}

export interface AutoAgentRequest {
  key: string
  payload: AgentInspectionPayload
}

export function AutoAgentPanel({ enabled, record }: { enabled: boolean; record?: AgentRecord }) {
  return (
    <Panel
      title="Agent 自动解读"
      eyebrow="结构化算法结果之上的辅助说明"
      aside={<span className={enabled ? 'agent-mode enabled' : 'agent-mode'}>{enabled ? 'AUTO ON' : 'AUTO OFF'}</span>}
      className="agent-panel"
    >
      {!enabled && (
        <div className="agent-off-state">
          <span className="agent-lock" aria-hidden="true">A</span>
          <div><strong>自动调用已关闭</strong><p>算法结论、曲线和证据不受影响；侧栏开启后才会主动调用 LLM。</p></div>
        </div>
      )}
      {enabled && (!record || record.status === 'idle') && <div className="agent-off-state"><span className="agent-lock">A</span><div><strong>等待诊断数据</strong><p>数据就绪后将自动发起一次解读。</p></div></div>}
      {enabled && record?.status === 'loading' && <div className="agent-loading"><span className="scanner" /><div><strong>Agent 正在筛选诊断证据</strong><small>工作流会按风险路径分析，当前指纹不会重复请求</small></div></div>}
      {enabled && record?.status === 'error' && <div className="inline-error" role="alert">Agent 自动解读失败：{record.error}</div>}
      {enabled && record?.status === 'success' && record.report && <AgentReportView report={record.report} />}
    </Panel>
  )
}

export function ManualAgentPanel({
  module,
  userId,
  date,
}: {
  module: BusinessModule
  userId: string
  date: string
}) {
  const [fieldText, setFieldText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [report, setReport] = useState<AgentReport | null>(null)

  useEffect(() => {
    setReport(null)
    setError('')
  }, [module, userId, date])

  async function generate() {
    if (!userId || !date) {
      setError('请先选择企业和有效日期。')
      return
    }
    setLoading(true)
    setError('')
    try {
      const result = await api.inspectAgent({ module, user_id: userId, diagnosis_date: date, field_text: fieldText })
      setReport(result)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '未知错误')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Panel title="人工触发智能检查" eyebrow="手动生成不受自动开关限制" className="manual-agent-panel">
      <label className="field-note">
        <span>现场补充信息（可选）</span>
        <textarea
          value={fieldText}
          onChange={(event) => setFieldText(event.target.value)}
          placeholder="例如：当天停产 2 小时、阀门已复核、机械字轮与远传累计量存在差异……"
          rows={3}
        />
      </label>
      <div className="manual-agent-actions">
        <p><b>注意：</b>点击后会调用 LLM；接口不可用时由后端返回本地规则回退报告。</p>
        <button className="button button-primary" type="button" disabled={loading || !userId || !date} onClick={generate}>
          {loading ? '智能检查生成中…' : '生成智能检查结果'}
        </button>
      </div>
      {error && <div className="inline-error" role="alert">生成失败：{error}</div>}
      {report && <AgentReportView report={report} />}
    </Panel>
  )
}
