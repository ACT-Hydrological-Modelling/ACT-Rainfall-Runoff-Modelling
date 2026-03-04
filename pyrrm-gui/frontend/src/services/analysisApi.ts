import api from './api'
import type {
  SessionSummary,
  SessionDetail,
  GaugeSummary,
  ExperimentInfo,
  DiagnosticsResponse,
  SummaryTableRow,
  PlotlyFigure,
  GeoLayersResponse,
  SignatureComparisonResponse,
  SignatureReferenceResponse,
  ReportCardResponse,
} from '../types/analysis'

export interface AvailableBatch {
  name: string
  path: string
  pkl_count: number
}

export interface BatchesResponse {
  root: string
  batches: AvailableBatch[]
}

// ── Signature reference (static) ────────────────────────────────────────────

export async function getSignatureReference(): Promise<SignatureReferenceResponse> {
  const response = await api.get('/analysis/signatures/reference')
  return response.data
}

// ── Available batches ───────────────────────────────────────────────────────

export async function listAvailableBatches(): Promise<BatchesResponse> {
  const response = await api.get('/analysis/batches')
  return response.data
}

// ── Sessions ────────────────────────────────────────────────────────────────

export async function loadSession(
  folderPath: string,
  name?: string
): Promise<SessionSummary> {
  const response = await api.post('/analysis/sessions', {
    folder_path: folderPath,
    name,
  })
  return response.data
}

export async function listSessions(): Promise<SessionSummary[]> {
  const response = await api.get('/analysis/sessions')
  return response.data
}

export async function getSession(sessionId: string): Promise<SessionDetail> {
  const response = await api.get(`/analysis/sessions/${sessionId}`)
  return response.data
}

export async function deleteSession(sessionId: string): Promise<void> {
  await api.delete(`/analysis/sessions/${sessionId}`)
}

// ── Summary ─────────────────────────────────────────────────────────────────

export async function getSessionSummary(
  sessionId: string
): Promise<SummaryTableRow[]> {
  const response = await api.get(`/analysis/sessions/${sessionId}/summary`)
  return response.data
}

// ── Gauges ──────────────────────────────────────────────────────────────────

export async function listGauges(
  sessionId: string
): Promise<GaugeSummary[]> {
  const response = await api.get(`/analysis/sessions/${sessionId}/gauges`)
  return response.data
}

// ── Experiments ─────────────────────────────────────────────────────────────

export async function getGaugeExperiments(
  sessionId: string,
  gaugeId: string
): Promise<ExperimentInfo[]> {
  const response = await api.get(
    `/analysis/sessions/${sessionId}/gauges/${gaugeId}/experiments`
  )
  return response.data
}

// ── Diagnostics ─────────────────────────────────────────────────────────────

export async function getGaugeDiagnostics(
  sessionId: string,
  gaugeId: string
): Promise<DiagnosticsResponse> {
  const response = await api.get(
    `/analysis/sessions/${sessionId}/gauges/${gaugeId}/diagnostics`
  )
  return response.data
}

// ── Comparison plots ────────────────────────────────────────────────────────

export async function getComparisonHydrograph(
  sessionId: string,
  gaugeId: string,
  logScale = false,
  experiments?: string[]
): Promise<PlotlyFigure> {
  const params: Record<string, string> = { log_scale: String(logScale) }
  if (experiments?.length) params.experiments = experiments.join(',')
  const response = await api.get(
    `/analysis/sessions/${sessionId}/gauges/${gaugeId}/comparison/hydrograph`,
    { params }
  )
  return response.data
}

export async function getComparisonFdc(
  sessionId: string,
  gaugeId: string,
  logScale = true,
  experiments?: string[]
): Promise<PlotlyFigure> {
  const params: Record<string, string> = { log_scale: String(logScale) }
  if (experiments?.length) params.experiments = experiments.join(',')
  const response = await api.get(
    `/analysis/sessions/${sessionId}/gauges/${gaugeId}/comparison/fdc`,
    { params }
  )
  return response.data
}

export async function getComparisonScatter(
  sessionId: string,
  gaugeId: string,
  logScale = false,
  experiments?: string[]
): Promise<PlotlyFigure> {
  const params: Record<string, string> = { log_scale: String(logScale) }
  if (experiments?.length) params.experiments = experiments.join(',')
  const response = await api.get(
    `/analysis/sessions/${sessionId}/gauges/${gaugeId}/comparison/scatter`,
    { params }
  )
  return response.data
}

export async function getComparisonSignatures(
  sessionId: string,
  gaugeId: string,
  experiments?: string[]
): Promise<SignatureComparisonResponse> {
  const params: Record<string, string> = {}
  if (experiments?.length) params.experiments = experiments.join(',')
  const response = await api.get(
    `/analysis/sessions/${sessionId}/gauges/${gaugeId}/comparison/signatures`,
    { params }
  )
  return response.data
}

// ── Geospatial data ─────────────────────────────────────────────────────────

export async function listGeoLayers(): Promise<GeoLayersResponse> {
  const response = await api.get('/geodata/layers')
  return response.data
}

export async function getGeoLayerGeoJSON(layer: string): Promise<any> {
  const response = await api.get(`/geodata/layers/${layer}`)
  return response.data
}

export async function uploadGeoLayer(
  layer: string,
  files: File[]
): Promise<{ layer: string; format: string; n_features: number }> {
  const formData = new FormData()
  for (const f of files) {
    formData.append('files', f)
  }
  const response = await api.post(`/geodata/layers/${layer}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return response.data
}

export async function deleteGeoLayer(layer: string): Promise<void> {
  await api.delete(`/geodata/layers/${layer}`)
}

// ── Report card ─────────────────────────────────────────────────────────────

export async function getExperimentReportCard(
  sessionId: string,
  gaugeId: string,
  expKey: string
): Promise<ReportCardResponse> {
  const response = await api.get(
    `/analysis/sessions/${sessionId}/gauges/${gaugeId}/experiments/${encodeURIComponent(expKey)}/report-card`
  )
  return response.data
}

// ── Report card export ───────────────────────────────────────────────────────

export interface ExportSection {
  id: string
  name: string
  description: string
}

export async function getExportSections(): Promise<ExportSection[]> {
  const response = await api.get('/analysis/export/sections')
  return response.data
}

export async function exportSingleReportCard(
  sessionId: string,
  gaugeId: string,
  expKey: string,
  format: 'pdf' | 'html' | 'interactive' = 'pdf',
  sections?: string[]
): Promise<Blob> {
  const params: Record<string, string> = { format }
  if (sections?.length) params.sections = sections.join(',')
  
  const response = await api.get(
    `/analysis/sessions/${sessionId}/gauges/${gaugeId}/experiments/${encodeURIComponent(expKey)}/report-card/export`,
    { params, responseType: 'blob' }
  )
  return response.data
}

export async function exportBatchReportCard(
  sessionId: string,
  gaugeId: string,
  format: 'pdf' | 'html' | 'interactive' = 'pdf',
  experiments?: string[],
  sections?: string[]
): Promise<Blob> {
  const params: Record<string, string> = { format }
  if (experiments?.length) params.experiments = experiments.join(',')
  if (sections?.length) params.sections = sections.join(',')
  
  const response = await api.get(
    `/analysis/sessions/${sessionId}/gauges/${gaugeId}/report-card/export`,
    { params, responseType: 'blob' }
  )
  return response.data
}
