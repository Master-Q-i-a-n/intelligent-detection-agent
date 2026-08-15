import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { AgentReportView } from './Common'


describe('AgentReportView', () => {
  it('展示工作流原因、证据、反证、缺口和核查顺序', () => {
    render(<AgentReportView report={{
      generator: 'workflow-llm:deepseek-v4-flash',
      workflow_route: 'high_risk',
      cache_hit: true,
      inspection_conclusion: '建议优先核查远传计量链路。',
      risk_level: '高',
      risk_score: 75,
      decision: '优先核查',
      analysis_confidence: 0.86,
      possible_causes: [{
        rank: 1,
        cause: '远传计量链路异常',
        confidence: 0.82,
        supporting_evidence_ids: ['M-STATE'],
        counter_evidence_ids: [],
      }],
      evidence_items: [{
        evidence_id: 'M-STATE',
        category: '状态交叉验证',
        statement: '模型识别用气，但远传没有形成计量。',
        source: '计量诊断/用气状态',
      }],
      counter_evidence: ['仍需排除停产。'],
      missing_evidence: ['缺少机械字轮读数。'],
      checklist: ['核对机械字轮与远传累计量'],
      work_order_advice: '建议生成核查工单。',
      data_boundary: 'LLM不修改算法原值。',
    }} />)

    expect(screen.getByText('高风险路径')).toBeInTheDocument()
    expect(screen.getByText('已复用相同诊断')).toBeInTheDocument()
    expect(screen.getByText('远传计量链路异常')).toBeInTheDocument()
    expect(screen.getByText('状态交叉验证')).toBeInTheDocument()
    expect(screen.getByText('模型识别用气，但远传没有形成计量。')).toBeInTheDocument()
    expect(screen.queryByText('计量诊断/用气状态')).not.toBeInTheDocument()
    expect(screen.queryByText('分析边界')).not.toBeInTheDocument()
    expect(screen.getByText('反证与边界')).toBeInTheDocument()
    expect(screen.getByText('仍缺少的证据')).toBeInTheDocument()
    expect(screen.getByText('核对机械字轮与远传累计量')).toBeInTheDocument()
  })
})
