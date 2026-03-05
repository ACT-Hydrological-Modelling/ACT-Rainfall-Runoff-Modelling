import type { PlotlyFigure } from './index'

export interface SessionSummary {
  id: string
  name: string
  folder_path: string
  gauge_ids: string[]
  total_experiments: number
  total_failures: number
  loaded_at: string
}

export interface GaugeMetadata {
  area_km2: number | null
  gauge_name: string | null
  record_start: string | null
  record_end: string | null
  record_years: number | null
  n_days: number | null
  mean_precip_mm_day: number | null
  mean_pet_mm_day: number | null
  total_precip_mm_yr: number | null
  total_pet_mm_yr: number | null
  mean_flow: number | null
  median_flow: number | null
  aridity_index: number | null
  runoff_ratio: number | null
}

export interface GaugeSummary {
  gauge_id: string
  n_experiments: number
  n_failures: number
  best_by_objective: Record<string, { key: string; value: number | null }>
  metadata: GaugeMetadata | null
}

export interface GeoLayer {
  name: string
  available: boolean
  n_features: number
}

export interface GeoLayersResponse {
  layers: GeoLayer[]
}

export interface SessionDetail extends SessionSummary {
  gauges: GaugeSummary[]
}

export interface ExperimentInfo {
  key: string
  model: string
  objective: string
  algorithm: string
  transformation: string | null
  best_objective: number
  runtime_seconds: number | null
  success: boolean
  headline_metrics: Record<string, number | null>
}

export interface DiagnosticsRow {
  experiment_key: string
  metrics: Record<string, number | null>
}

export interface ClustermapData {
  heatmap_values: (number | null)[][]
  row_labels: string[]
  col_labels: string[]
  row_dendrogram: DendrogramData
  col_dendrogram: DendrogramData
  annotations: (string | null)[][]
}

export interface DendrogramData {
  icoord: number[][]
  dcoord: number[][]
  leaves: number[]
}

export interface DiagnosticsResponse {
  raw_table: DiagnosticsRow[]
  normalised_table: DiagnosticsRow[]
  clustermap: ClustermapData | null
  top_experiments: { experiment_key: string; mean_score: number }[]
  metric_groups: Record<string, string[]>
}

export interface SummaryTableRow {
  key: string
  gauge_id: string
  model: string
  objective: string
  algorithm: string
  transformation: string | null
  best_objective: number | null
  runtime_seconds: number | null
  success: boolean
  parameters: Record<string, number | null>
}

// ── Signature comparison types ──────────────────────────────────────────────

export interface SignatureCategoryData {
  signatures: string[]
  observed: Record<string, number | null>
  experiments: Record<string, Record<string, number | null>>
  percent_errors: Record<string, Record<string, number | null>>
}

export interface SignatureComparisonResponse {
  categories: Record<string, SignatureCategoryData>
  bar_figures: Record<string, PlotlyFigure>
  heatmap_figure: PlotlyFigure | null
}

// ── Signature reference types ───────────────────────────────────────────────

export interface SignatureInfo {
  id: string
  name: string
  category: string
  units: string
  range: (number | string)[] | null
  description: string
  formula: string
  interpretation: string
  related: string[]
  references: string[]
}

export interface SignatureReferenceResponse {
  categories: Record<string, SignatureInfo[]>
  category_order: string[]
  total_signatures: number
}

// ── Report card types ────────────────────────────────────────────────────────

export interface ReportCardHeader {
  catchment_name: string
  gauge_id: string
  area: string | number | null
  method: string
  objective_name: string
  best_objective: number
  period_start: string
  period_end: string
}

export interface ReportCardResponse {
  header: ReportCardHeader
  hydrograph_linear: PlotlyFigure
  hydrograph_log: PlotlyFigure
  metrics_table: PlotlyFigure
  fdc: PlotlyFigure
  scatter: PlotlyFigure
  parameters: PlotlyFigure | null
  signatures_table: PlotlyFigure | null
}

export type { PlotlyFigure }
