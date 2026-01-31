// Catchment types
export interface Catchment {
  id: string
  name: string
  gauge_id?: string
  area_km2?: number
  description?: string
  created_at: string
  updated_at: string
  datasets: DatasetSummary[]
  experiments: ExperimentSummary[]
}

export interface CatchmentListItem {
  id: string
  name: string
  gauge_id?: string
  area_km2?: number
  created_at: string
  dataset_count: number
  experiment_count: number
}

export interface CatchmentCreate {
  name: string
  gauge_id?: string
  area_km2?: number
  description?: string
}

// Dataset types
export type DatasetType = 'rainfall' | 'pet' | 'observed_flow'

export interface Dataset {
  id: string
  catchment_id: string
  name: string
  type: DatasetType
  file_path: string
  start_date?: string
  end_date?: string
  record_count?: number
  metadata?: Record<string, any>
  created_at: string
}

export interface DatasetSummary {
  id: string
  name: string
  type: DatasetType
  record_count?: number
  start_date?: string
  end_date?: string
}

export interface DatasetPreview {
  id: string
  name: string
  type: DatasetType
  start_date?: string
  end_date?: string
  record_count: number
  statistics: Record<string, number>
  sample_data: Record<string, any>[]
  missing_count: number
  missing_percentage: number
  warnings: string[]
  errors: string[]
}

// Experiment types
export type ExperimentStatus = 'draft' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
export type ModelType = 'sacramento' | 'gr4j' | 'gr5j' | 'gr6j'

export interface CalibrationPeriod {
  start_date: string
  end_date: string
  warmup_days: number
}

export interface FlowTrimmingConfig {
  enabled: boolean
  min_threshold?: {
    type: 'absolute' | 'percentile'
    value: number
  }
  max_threshold?: {
    type: 'absolute' | 'percentile'
    value: number
  }
}

export interface ObjectiveConfig {
  type: string
  transform: string
  weights?: Record<string, number>
  flow_trimming?: FlowTrimmingConfig
}

export interface AlgorithmConfig {
  method: string
  max_evals: number
  n_complexes?: number
  max_workers: number
  seed?: number
  checkpoint_interval: number
  // Convergence criteria
  max_tolerant_iter: number  // Max iterations without improvement before stopping
  tolerance: number          // Improvement threshold for convergence
}

export interface RoutingConfig {
  enabled: boolean
  K: number
  m: number
  n_subreaches: number
  calibrate_routing: boolean
}

export interface ModelConfig {
  routing?: RoutingConfig
  initial_states?: Record<string, number>
}

export interface Experiment {
  id: string
  catchment_id: string
  name: string
  description?: string
  model_type: ModelType
  model_config?: ModelConfig
  parameter_bounds?: Record<string, [number, number]>
  calibration_period?: CalibrationPeriod
  objective_config?: ObjectiveConfig
  algorithm_config?: AlgorithmConfig
  status: ExperimentStatus
  celery_task_id?: string
  error_message?: string
  created_at: string
  started_at?: string
  completed_at?: string
  runtime_seconds?: number
  has_result: boolean
  best_objective?: number
}

export interface ExperimentSummary {
  id: string
  name: string
  status: ExperimentStatus
  model_type: ModelType
  created_at: string
}

export interface ExperimentCreate {
  catchment_id: string
  name: string
  description?: string
  model_type: ModelType
  model_config?: ModelConfig
  parameter_bounds?: Record<string, [number, number]>
  calibration_period?: CalibrationPeriod
  objective_config?: ObjectiveConfig
  algorithm_config?: AlgorithmConfig
}

export interface ExperimentStatusInfo {
  id: string
  status: ExperimentStatus
  progress?: CalibrationProgress
  error_message?: string
}

// Result types
export interface LogMessage {
  time: string
  message: string
  level: 'info' | 'success' | 'warning' | 'error'
}

export interface ObjectiveHistoryPoint {
  nfev: number
  objective: number
  iteration: number
}

export interface SimulationData {
  hydrograph: {
    dates: string[]
    observed: (number | null)[]
    simulated: (number | null)[]
  }
  fdc: {
    exceedance: number[]
    observed: number[]
    simulated: number[]
  }
  scatter: {
    observed: number[]
    simulated: number[]
  }
  parameters: Record<string, number>
}

export interface CalibrationProgress {
  experiment_id: string
  iteration: number
  nfev?: number
  total_iterations?: number
  best_objective: number
  current_objective: number
  best_parameters: Record<string, number> | number[]
  parameter_names?: string[]
  progress_percent?: number
  elapsed_time?: string
  eta?: string
  no_improve_count?: number
  improved?: boolean
  log_messages?: LogMessage[]
  objective_history?: ObjectiveHistoryPoint[]
  simulation_data?: SimulationData | null
  timestamp?: string
}

export interface CalibrationResult {
  id: string
  experiment_id: string
  best_parameters: Record<string, number>
  best_objective: number
  metrics: Metrics
  runtime_seconds?: number
  iterations_completed?: number
  report_file_path?: string
  created_at: string
}

export interface Metrics {
  NSE?: number
  KGE?: number
  RMSE?: number
  MAE?: number
  PBIAS?: number
  LogNSE?: number
  InvNSE?: number
  SqrtNSE?: number
  KGE_r?: number
  KGE_alpha?: number
  KGE_beta?: number
}

export interface ParameterInfo {
  name: string
  value: number
  min_bound: number
  max_bound: number
  percent_of_range: number
  description?: string
  unit?: string
}

// Reference types
export interface ModelInfo {
  type: string
  name: string
  description: string
  n_parameters: number
  parameters: ParameterDefinition[]
}

export interface ParameterDefinition {
  name: string
  default: number
  min_bound: number
  max_bound: number
  description: string
  unit: string
}

export interface ObjectiveInfo {
  name: string
  description: string
  maximize: boolean
  optimal_value: number
}

export interface AlgorithmInfo {
  id: string
  name: string
  description: string
  parameters: AlgorithmParameter[]
  requires?: string
}

export interface AlgorithmParameter {
  name: string
  type: string
  default: any
  min?: number
  description: string
}

// Plot types
export interface PlotlyFigure {
  data: any[]
  layout: any
}
