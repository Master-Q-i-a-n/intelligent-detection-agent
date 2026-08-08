export type PageKey = 'overview' | 'metering' | 'equipment'
export type BusinessModule = 'metering' | 'equipment'

export interface UserSummary {
  user_id: string
  company_name: string
  date_range?: [string, string]
  [key: string]: unknown
}

export interface UserListResponse {
  items: UserSummary[]
  count: number
  enterprise_count: number
  date_range: [string, string] | []
}

export interface DailyIssue {
  module: BusinessModule
  user_id: string
  company_name: string
  risk_level: string
  issue_type: string
  issue_tags: string[]
  evidence_summary: string
  primary_metric_name: string
  primary_metric: number | null
}

export interface DailyOverview {
  diagnosis_date: string
  status: string
  diagnosed_enterprises: number
  abnormal_enterprises: number
  normal_enterprises: number
  metering_issue_count: number
  equipment_issue_count: number
  high_risk_count?: number
  issues: DailyIssue[]
}

export interface MeteringHistoryItem {
  date: string
  volume: number | null
  max_flow: number | null
}

export interface MeteringDiagnosis {
  run_id: string
  user_id: string
  user_name: string
  diagnosis_date: string
  status: string
  quality_status: number
  model_gas_state: number | null
  observed_gas_state: number | null
  observed_volume: number
  predicted_normal_volume: number
  baseline_missing_volume: number
  meter_bias_volume: number
  makeup_volume: number
  risk_score: number
  risk_level: string
  meter_spec_result: string
  summary: string
  alerts: Array<Record<string, unknown> | string>
  anomaly_intervals: MeteringInterval[]
  work_order: Record<string, unknown> | null
  details: {
    baseline?: { history_days?: number; predicted_normal_volume?: number }
    data_quality?: {
      status?: number
      pipeline_completeness?: Record<string, number | null>
      pipeline_daily_volume?: Record<string, number | null>
    }
    gas_state?: Record<string, unknown>
    diagnostic_evidence?: Record<string, unknown[]>
    meter_error_model?: Record<string, unknown>
    meter_spec?: {
      quantity_min?: number | null
      quantity_max?: number | null
      small_flow_percentage?: number
      normal_flow_percentage?: number
      over_flow_percentage?: number
    }
    [key: string]: unknown
  }
}

export interface MeteringInterval {
  start_time: string
  end_time: string
  anomaly_type?: string
  observed_value?: number
  expected_value?: number
  estimated_missing_volume?: number
  severity?: string
}

export interface PipelineSignals {
  flow: Array<number | null>
  pressure: Array<number | null>
  temperature: Array<number | null>
}

export interface MeteringSignals {
  user_id: string
  diagnosis_date: string
  times: string[]
  resample_frequency: string
  point_count: number
  pipelines: Record<string, PipelineSignals>
}

export interface EquipmentHistoryItem {
  date: string
  operating_condition: number
  predicted_stage: string
  predicted_health_index: number
  stabilized_stage?: string
  stabilized_health_index?: number
  temporal_stage?: string
  confidence: number
}

export interface EquipmentDashboard {
  entity: Record<string, unknown>
  current_assessment: {
    date: string
    stage: string
    stage_name: string
    health_index: number
    confidence: number
    probabilities: Record<string, number>
    risk_level: string
    operating_condition?: number
  }
  trend_assessment: {
    trend_label: string
    trend_name: string
    daily_slope: number
    maximum_daily_drop: number
    health_index_start: number
    health_index_end: number
    stage_sequence: string[]
    window_days: number
  }
  model_explanation: {
    axis_weights: number[]
    morphological_scale_weights: number[]
    scale_kernel_sizes: number[]
  }
  recommended_action: string
  daily_history: EquipmentHistoryItem[]
  model: Record<string, unknown>
  data_notice: Record<string, unknown>
}

export interface EquipmentWaveform {
  sampling_rate_hz: number
  indices: number[]
  x: number[]
  y: number[]
  z: number[]
}

export interface AgentReport {
  generator?: string
  title?: string
  inspection_conclusion?: string
  conclusion?: string
  summary?: string
  risk_level?: string
  evidence_chain?: Array<string | Record<string, unknown>>
  evidence?: Array<string | Record<string, unknown>>
  field_checklist?: string[]
  checklist?: string[]
  recommendations?: string[]
  work_order_advice?: string
  work_order_suggestion?: string
  [key: string]: unknown
}

export interface AgentInspectionPayload {
  module: BusinessModule
  user_id: string
  diagnosis_date: string
  field_text: string
  context: Record<string, unknown>
}
