export type PageKey = 'overview' | 'metering' | 'equipment' | 'safety' | 'chat'
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

export type SecurityDecision = 'CONFIRMED' | 'UNCERTAIN'
export type SecurityHandlingStatus = 'NEW' | 'ACKNOWLEDGED' | 'PROCESSING' | 'CLOSED'

export interface SecurityOverview {
  total: number
  confirmed: number
  review_required: number
  new_count: number
  processing: number
  high_risk: number
  latest_sequence: number
}

export interface SecurityEvidence {
  evidence_id: number
  evidence_type: string
  mime_type: string | null
  file_size?: number | null
  metadata_json?: string | null
}

export interface SecurityAction {
  action_id: string
  action: string
  operator: string
  comment: string | null
  previous_status: string
  new_status: string
  acted_at: string
}

export interface PPEPersonRuleMetrics {
  track_id: number
  track_ids?: number[]
  visible_seconds?: number
  helmet_rule_status?: string
  helmet_seen_worn_at?: number | null
  helmet_last_worn_at?: number | null
  first_no_helmet_at?: number | null
  helmet_violation_at?: number | null
  gloves_rule_status?: string
  goggles_rule_status?: string
  gloves_positive_frames?: number
  goggles_positive_frames?: number
  no_gloves_positive_frames?: number
  no_goggle_positive_frames?: number
  gloves_effective_seconds?: number
  goggles_effective_seconds?: number
  gloves_confirmed_at?: number | null
  goggles_confirmed_at?: number | null
  best_visibility_at?: number | null
}

export interface PPERuleMetrics {
  source_sha256?: string
  anchor_track_id?: number
  anchor_time_seconds?: number
  people?: PPEPersonRuleMetrics[]
}

export interface SecurityEvent {
  notification_sequence: number
  notification_kind: 'CONFIRMED_ALERT' | 'REVIEW_REQUIRED'
  sent_at: string
  event_id: string
  source_system: string
  event_type: string
  camera_id: string
  zone_id: string | null
  primary_track_id: number | null
  lifecycle_status: string
  activated_video_seconds: number | null
  occurred_at: string | null
  severity: string | null
  final_decision: SecurityDecision
  final_reason: string | null
  recommended_action: string | null
  handling_status: SecurityHandlingStatus
  latest_review_helmet_status: string | null
  latest_review_ppe_results?: SecurityEventPerson[] | null
  latest_review_explanation: string | null
  latest_reviewed_at: string | null
  yolo_rule_metrics?: PPERuleMetrics | null
  evidence_count: number
  evidence?: SecurityEvidence[]
  actions?: SecurityAction[]
  people?: SecurityEventPerson[]
}

export interface SecurityEventPerson {
  track_id: number
  helmet_status: string | null
  gloves_status: string | null
  goggles_status: string | null
  zone_id?: string | null
  visibility?: string
  evidence_quality?: string
  visual_reason?: string
  evidence_timestamps?: number[]
}

export type ChatArtifactType = 'query_result' | 'report' | 'work_order'

export interface ChatArtifact {
  type: ChatArtifactType
  id: string
  payload: Record<string, unknown>
}

export interface ChatInterrupt {
  kind: 'clarification' | 'work_order_approval'
  question?: string
  missing_information?: string[]
  suggestions?: string[]
  action?: {
    name?: string
    arguments?: Record<string, unknown>
    args?: Record<string, unknown>
    description?: string
  }
  allowed_decisions?: Array<'approve' | 'edit' | 'reject'>
}

export interface ChatTurnResponse {
  status: 'completed' | 'interrupted'
  message: string
  generator: string
  artifacts: ChatArtifact[]
  todos?: ChatTodo[]
  interrupt?: ChatInterrupt | null
}

export interface ChatTodo {
  content: string
  status: 'pending' | 'in_progress' | 'completed'
}

export type ChatStreamEventName =
  | 'meta'
  | 'status'
  | 'todo'
  | 'tool_start'
  | 'tool_end'
  | 'answer_delta'
  | 'answer_reset'
  | 'artifact'
  | 'interrupt'
  | 'done'
  | 'error'

export interface ChatStreamEvent {
  event: ChatStreamEventName
  data: Record<string, unknown>
}

export interface ChatResumePayload {
  thread_id: string
  kind: ChatInterrupt['kind']
  decision: 'answer' | 'approve' | 'edit' | 'reject'
  message?: string
  edited_action?: Record<string, unknown>
}

export interface ChatStatus {
  configured: boolean
  provider: string
  model: string
  memory: string
  tracing_enabled: boolean
}

export interface AuthUser {
  user_id: string
  username: string
}

export interface ChatThreadSummary {
  thread_id: string
  title: string
  created_at: string
  updated_at: string
  status: 'completed' | 'interrupted' | 'error'
}

export interface ChatHistoryMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  generator?: string | null
  artifact_ids: string[]
  created_at: string
}

export interface ChatThreadDetail {
  thread: Omit<ChatThreadSummary, 'status'>
  messages: ChatHistoryMessage[]
  artifacts: ChatArtifact[]
  todos: ChatTodo[]
  interrupt: ChatInterrupt | null
  last_error: string | null
}
