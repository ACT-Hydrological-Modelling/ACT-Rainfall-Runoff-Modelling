import axios from 'axios'
import type {
  Catchment,
  CatchmentListItem,
  CatchmentCreate,
  Dataset,
  DatasetPreview,
  DatasetType,
  Experiment,
  ExperimentCreate,
  CalibrationResult,
  ParameterInfo,
  ModelInfo,
  ObjectiveInfo,
  AlgorithmInfo,
  PlotlyFigure,
} from '../types'

// Create axios instance
const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

// ============================================================================
// Catchment API
// ============================================================================

export async function getCatchments(): Promise<CatchmentListItem[]> {
  const response = await api.get('/catchments/')
  return response.data
}

export async function getCatchment(id: string): Promise<Catchment> {
  const response = await api.get(`/catchments/${id}`)
  return response.data
}

export async function createCatchment(data: CatchmentCreate): Promise<Catchment> {
  const response = await api.post('/catchments/', data)
  return response.data
}

export async function updateCatchment(id: string, data: Partial<CatchmentCreate>): Promise<Catchment> {
  const response = await api.put(`/catchments/${id}`, data)
  return response.data
}

export async function deleteCatchment(id: string): Promise<void> {
  await api.delete(`/catchments/${id}`)
}

export interface CatchmentTimeseries {
  dates: string[]
  rainfall: (number | null)[] | null
  pet: (number | null)[] | null
  observed_flow: (number | null)[] | null
  data_range: {
    start: string
    end: string
    total_days: number
  }
  // Calibration range = observed flow range (can't calibrate without observed data)
  calibration_range: {
    start: string
    end: string
    total_days: number
  }
  statistics: {
    rainfall?: { mean: number; max: number; total: number }
    pet?: { mean: number; max: number }
    observed_flow?: { mean: number; max: number; min: number; p10: number; p50: number; p90: number }
  }
}

export async function getCatchmentTimeseries(id: string, maxPoints?: number): Promise<CatchmentTimeseries> {
  const params = maxPoints ? { max_points: maxPoints } : {}
  const response = await api.get(`/catchments/${id}/timeseries`, { params })
  return response.data
}

// ============================================================================
// Dataset API
// ============================================================================

export async function getDatasets(catchmentId: string): Promise<Dataset[]> {
  const response = await api.get(`/datasets/catchments/${catchmentId}`)
  return response.data
}

export async function uploadDataset(
  catchmentId: string,
  name: string,
  type: DatasetType,
  file: File
): Promise<Dataset> {
  const formData = new FormData()
  formData.append('name', name)
  formData.append('type', type)
  formData.append('file', file)
  
  const response = await api.post(`/datasets/catchments/${catchmentId}/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export async function getDatasetPreview(id: string): Promise<DatasetPreview> {
  const response = await api.get(`/datasets/${id}/preview`)
  return response.data
}

export async function getDatasetStatistics(id: string): Promise<any> {
  const response = await api.get(`/datasets/${id}/statistics`)
  return response.data
}

export async function getDatasetPlot(id: string): Promise<PlotlyFigure> {
  const response = await api.get(`/datasets/${id}/plot`)
  return response.data
}

export async function deleteDataset(id: string): Promise<void> {
  await api.delete(`/datasets/${id}`)
}

export interface DataQualityReport {
  total_records: number
  clean_records: number
  sentinel_values: number
  negative_values: number
  nan_values: number
  zero_values: number
  potential_outliers: number
  has_issues: boolean
  issue_percentage: number | null
  issues: string[]
  cleaning_applied: string[]
}

export interface DataQualityResponse {
  dataset_id: string
  dataset_name: string
  dataset_type: string
  quality_report: DataQualityReport
  recommendations: string[]
}

export interface CleaningOptions {
  replace_sentinel?: boolean
  replace_negative?: boolean
  sentinel_values?: number[]
  drop_na?: boolean
  interpolate?: boolean
  max_interpolate_gap?: number
}

export interface CleaningResponse {
  success: boolean
  dataset_id: string
  cleaning_report: {
    original_records: number
    clean_records: number
    records_after_cleaning: number
    operations_applied: string[]
    issue_percentage: number | null
  }
  message: string
}

export async function getDatasetQuality(id: string): Promise<DataQualityResponse> {
  const response = await api.get(`/datasets/${id}/quality`)
  return response.data
}

export async function cleanDataset(id: string, options?: CleaningOptions): Promise<CleaningResponse> {
  const response = await api.post(`/datasets/${id}/clean`, options || {})
  return response.data
}

// ============================================================================
// Experiment API
// ============================================================================

export async function getExperiments(catchmentId?: string): Promise<Experiment[]> {
  const params = catchmentId ? { catchment_id: catchmentId } : {}
  const response = await api.get('/experiments/', { params })
  return response.data
}

export async function getExperiment(id: string): Promise<Experiment> {
  const response = await api.get(`/experiments/${id}`)
  return response.data
}

export async function createExperiment(data: ExperimentCreate): Promise<Experiment> {
  const response = await api.post('/experiments/', data)
  return response.data
}

export async function updateExperiment(id: string, data: Partial<ExperimentCreate>): Promise<Experiment> {
  const response = await api.put(`/experiments/${id}`, data)
  return response.data
}

export async function deleteExperiment(id: string): Promise<void> {
  await api.delete(`/experiments/${id}`)
}

export async function cloneExperiment(id: string, name?: string): Promise<Experiment> {
  const response = await api.post(`/experiments/${id}/clone`, null, {
    params: name ? { name } : {},
  })
  return response.data
}

export async function runExperiment(id: string): Promise<{ id: string; status: string }> {
  const response = await api.post(`/experiments/${id}/run`)
  return response.data
}

export async function cancelExperiment(id: string): Promise<{ id: string; status: string }> {
  const response = await api.post(`/experiments/${id}/cancel`)
  return response.data
}

export async function getExperimentStatus(id: string): Promise<{ id: string; status: string; progress?: any }> {
  const response = await api.get(`/experiments/${id}/status`)
  return response.data
}

// ============================================================================
// Results API
// ============================================================================

export async function getResult(experimentId: string): Promise<CalibrationResult> {
  const response = await api.get(`/results/experiments/${experimentId}`)
  return response.data
}

export async function getParameters(experimentId: string): Promise<ParameterInfo[]> {
  const response = await api.get(`/results/experiments/${experimentId}/parameters`)
  return response.data
}

export async function getMetrics(experimentId: string): Promise<Record<string, { value: number | string; color: string }>> {
  const response = await api.get(`/results/experiments/${experimentId}/metrics`)
  return response.data
}

export async function getHydrographPlot(experimentId: string, logScale = false): Promise<PlotlyFigure> {
  const response = await api.get(`/results/experiments/${experimentId}/plots/hydrograph`, {
    params: { log_scale: logScale },
  })
  return response.data
}

export async function getFdcPlot(experimentId: string, logScale = true): Promise<PlotlyFigure> {
  const response = await api.get(`/results/experiments/${experimentId}/plots/fdc`, {
    params: { log_scale: logScale },
  })
  return response.data
}

export async function getScatterPlot(experimentId: string): Promise<PlotlyFigure> {
  const response = await api.get(`/results/experiments/${experimentId}/plots/scatter`)
  return response.data
}

export async function getParametersPlot(experimentId: string): Promise<PlotlyFigure> {
  const response = await api.get(`/results/experiments/${experimentId}/plots/parameters`)
  return response.data
}

// ============================================================================
// Reference API
// ============================================================================

export async function getModels(): Promise<ModelInfo[]> {
  const response = await api.get('/reference/models')
  return response.data
}

export async function getModelBounds(modelType: string): Promise<Record<string, [number, number]>> {
  const response = await api.get(`/reference/models/${modelType}/bounds`)
  return response.data
}

export async function getObjectives(): Promise<ObjectiveInfo[]> {
  const response = await api.get('/reference/objectives')
  return response.data
}

export async function getAlgorithms(): Promise<AlgorithmInfo[]> {
  const response = await api.get('/reference/algorithms')
  return response.data
}

export async function getTransforms(): Promise<{ id: string; name: string; description: string }[]> {
  const response = await api.get('/reference/transforms')
  return response.data
}

export default api
