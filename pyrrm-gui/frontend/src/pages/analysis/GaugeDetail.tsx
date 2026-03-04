import { useState, useCallback, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import {
  BarChart3,
  GitCompare,
  FileText,
  Trophy,
  Play,
  RefreshCw,
  AlertTriangle,
  Download,
  ChevronDown,
  Settings,
} from 'lucide-react'
import {
  getSession,
  getGaugeExperiments,
  getGaugeDiagnostics,
  getComparisonHydrograph,
  getComparisonFdc,
  getComparisonScatter,
  getExperimentReportCard,
  getExportSections,
  exportSingleReportCard,
  exportBatchReportCard,
  type ExportSection,
} from '../../services/analysisApi'
import PlotlyChart from '../../components/analysis/PlotlyChart'
import DiagnosticsTable from '../../components/analysis/DiagnosticsTable'
import ClustermapChart from '../../components/analysis/ClustermapChart'
import MetricThresholdPanel from '../../components/analysis/MetricThresholdPanel'
import ExperimentSelector from '../../components/analysis/ExperimentSelector'
import SignatureComparisonCharts from '../../components/analysis/SignatureComparisonCharts'
import type { ExperimentInfo, ReportCardResponse } from '../../types/analysis'

type TabId = 'detailed-comparison' | 'rapid-comparison' | 'report-card'

const TABS: { id: TabId; label: string; icon: React.ElementType }[] = [
  { id: 'detailed-comparison', label: 'Detailed Comparison', icon: BarChart3 },
  { id: 'rapid-comparison', label: 'Rapid Comparison', icon: GitCompare },
  { id: 'report-card', label: 'Report Cards', icon: FileText },
]

export default function GaugeDetail() {
  const { sessionId, gaugeId } = useParams<{ sessionId: string; gaugeId: string }>()
  const [activeTab, setActiveTab] = useState<TabId>('detailed-comparison')

  const { data: session } = useQuery({
    queryKey: ['analysis-session', sessionId],
    queryFn: () => getSession(sessionId!),
    enabled: !!sessionId,
  })

  const { data: experiments } = useQuery({
    queryKey: ['gauge-experiments', sessionId, gaugeId],
    queryFn: () => getGaugeExperiments(sessionId!, gaugeId!),
    enabled: !!sessionId && !!gaugeId,
  })

  const gaugeName = gaugeId || 'Unknown'
  const sessionName = session?.name || sessionId

  return (
    <div>
      <div className="mb-6">
        <p className="text-sm text-gray-400">{sessionName}</p>
        <h1 className="text-2xl font-bold text-gray-900">Gauge {gaugeName}</h1>
        {experiments && (
          <p className="text-sm text-gray-500 mt-1">
            {experiments.length} experiments loaded
          </p>
        )}
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex gap-6">
          {TABS.map((tab) => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 pb-3 border-b-2 text-sm font-medium transition-colors ${
                  isActive
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            )
          })}
        </nav>
      </div>

      {/* Tab content */}
      {activeTab === 'rapid-comparison' && (
        <RapidComparisonTab
          sessionId={sessionId!}
          gaugeId={gaugeId!}
          experiments={experiments || []}
        />
      )}
      {activeTab === 'detailed-comparison' && (
        <DetailedComparisonTab sessionId={sessionId!} gaugeId={gaugeId!} />
      )}
      {activeTab === 'report-card' && (
        <ReportCardTab
          sessionId={sessionId!}
          gaugeId={gaugeId!}
          experiments={experiments || []}
        />
      )}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tab 2: Detailed Comparison (formerly Diagnostics)
// ═══════════════════════════════════════════════════════════════════════════════

function DetailedComparisonTab({ sessionId, gaugeId }: { sessionId: string; gaugeId: string }) {
  const { data: diagnostics, isLoading } = useQuery({
    queryKey: ['gauge-diagnostics', sessionId, gaugeId],
    queryFn: () => getGaugeDiagnostics(sessionId, gaugeId),
  })

  const [passingKeys, setPassingKeys] = useState<Set<string>>(new Set())
  const [chartsRequested, setChartsRequested] = useState(false)
  const [chartExperimentKeys, setChartExperimentKeys] = useState<string[]>([])

  const handlePassingKeysChange = useCallback((keys: Set<string>) => {
    setPassingKeys(keys)
  }, [])

  const filtersChanged = chartsRequested && (
    chartExperimentKeys.length !== passingKeys.size ||
    chartExperimentKeys.some((k) => !passingKeys.has(k))
  )

  const handleLoadCharts = useCallback(() => {
    const keys = [...passingKeys]
    setChartExperimentKeys(keys)
    setChartsRequested(true)
  }, [passingKeys])

  if (isLoading) {
    return <div className="text-center py-12 text-gray-400">Computing diagnostics...</div>
  }

  if (!diagnostics) {
    return <div className="text-center py-12 text-red-500">Failed to load diagnostics</div>
  }

  const metricCols = diagnostics.raw_table.length > 0
    ? Object.keys(diagnostics.raw_table[0].metrics)
    : []

  return (
    <div className="space-y-8">
      {/* 1. Top experiments */}
      {diagnostics.top_experiments.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-5">
          <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
            <Trophy className="w-4 h-4 text-amber-500" />
            Top Experiments (by mean normalised score)
          </h3>
          <div className="space-y-2">
            {diagnostics.top_experiments.map((exp, i) => (
              <div key={exp.experiment_key} className="flex items-center gap-3">
                <span className="text-sm font-bold text-gray-400 w-6 text-right">
                  #{i + 1}
                </span>
                <span className="text-sm font-mono text-gray-700 flex-1">
                  {exp.experiment_key}
                </span>
                <span className="text-sm font-mono text-primary-600 font-medium">
                  {exp.mean_score.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 2. Clustermap */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-3">
          Hierarchical Clustermap (normalised, higher = better)
        </h3>
        <ClustermapChart
          data={diagnostics.clustermap}
          height={Math.max(400, (diagnostics.raw_table.length || 1) * 28 + 200)}
        />
      </div>

      {/* 3. Normalised diagnostics table (moved up) */}
      <DiagnosticsTable
        rows={diagnostics.normalised_table}
        metricColumns={metricCols}
        title="Normalised Diagnostics (0 = worst, 1 = best)"
        normalized
      />

      {/* 4. Threshold filter panel */}
      <MetricThresholdPanel
        rows={diagnostics.raw_table}
        metricColumns={metricCols}
        onPassingKeysChange={handlePassingKeysChange}
      />

      {/* 5. Raw diagnostics table (filtered) */}
      <DiagnosticsTable
        rows={diagnostics.raw_table}
        metricColumns={metricCols}
        title="Raw Diagnostics"
        highlightKeys={passingKeys}
      />

      {/* 6. Comparison charts (button-gated) */}
      <div className="bg-white rounded-lg border border-gray-200 p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
            <GitCompare className="w-4 h-4 text-primary-500" />
            Comparison Charts
          </h3>
          <div className="flex items-center gap-3">
            {filtersChanged && (
              <span className="flex items-center gap-1.5 text-xs text-amber-600">
                <AlertTriangle className="w-3.5 h-3.5" />
                Filters changed
              </span>
            )}
            <button
              onClick={handleLoadCharts}
              disabled={passingKeys.size === 0}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                passingKeys.size === 0
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  : filtersChanged
                    ? 'bg-amber-500 text-white hover:bg-amber-600'
                    : 'bg-primary-600 text-white hover:bg-primary-700'
              }`}
            >
              {filtersChanged ? (
                <RefreshCw className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {chartsRequested
                ? `Reload (${passingKeys.size} experiments)`
                : `Load Charts (${passingKeys.size} experiments)`}
            </button>
          </div>
        </div>

        {!chartsRequested ? (
          <div className="text-center py-8 text-gray-400 text-sm">
            Adjust threshold filters above, then click "Load Charts" to compare passing experiments.
          </div>
        ) : (
          <FilteredComparisonCharts
            sessionId={sessionId}
            gaugeId={gaugeId}
            experimentKeys={chartExperimentKeys}
          />
        )}
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Filtered Comparison Charts (used within DiagnosticsTab)
// ═══════════════════════════════════════════════════════════════════════════════

function FilteredComparisonCharts({
  sessionId,
  gaugeId,
  experimentKeys,
}: {
  sessionId: string
  gaugeId: string
  experimentKeys: string[]
}) {
  const [hydroLog, setHydroLog] = useState(false)
  const [fdcLog, setFdcLog] = useState(true)
  const [scatterLog, setScatterLog] = useState(true)

  const { data: hydroFig, isLoading: hydroLoading } = useQuery({
    queryKey: ['diag-hydro', sessionId, gaugeId, hydroLog, experimentKeys],
    queryFn: () => getComparisonHydrograph(sessionId, gaugeId, hydroLog, experimentKeys),
    enabled: experimentKeys.length > 0,
  })

  const { data: fdcFig, isLoading: fdcLoading } = useQuery({
    queryKey: ['diag-fdc', sessionId, gaugeId, fdcLog, experimentKeys],
    queryFn: () => getComparisonFdc(sessionId, gaugeId, fdcLog, experimentKeys),
    enabled: experimentKeys.length > 0,
  })

  const { data: scatterFig, isLoading: scatterLoading } = useQuery({
    queryKey: ['diag-scatter', sessionId, gaugeId, scatterLog, experimentKeys],
    queryFn: () => getComparisonScatter(sessionId, gaugeId, scatterLog, experimentKeys),
    enabled: experimentKeys.length > 0,
  })

  return (
    <div className="space-y-6">
      {/* Row 1: Hydrograph (full width, with scale toggle) */}
      <div className="border border-gray-100 rounded-lg p-3">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-xs font-semibold text-gray-600">Hydrograph</h4>
          <ScaleToggle value={hydroLog} onChange={setHydroLog} />
        </div>
        <PlotlyChart figure={hydroFig} loading={hydroLoading} />
      </div>

      {/* Row 2: FDC + Scatter side by side */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="border border-gray-100 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-semibold text-gray-600">
              Flow Duration Curve
            </h4>
            <ScaleToggle value={fdcLog} onChange={setFdcLog} />
          </div>
          <PlotlyChart figure={fdcFig} loading={fdcLoading} />
        </div>

        <div className="border border-gray-100 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-semibold text-gray-600">
              Observed vs Simulated
            </h4>
            <ScaleToggle value={scatterLog} onChange={setScatterLog} />
          </div>
          <PlotlyChart figure={scatterFig} loading={scatterLoading} />
        </div>
      </div>

      {/* Row 3: Hydrologic Signatures Comparison */}
      <SignatureComparisonCharts
        sessionId={sessionId}
        gaugeId={gaugeId}
        experimentKeys={experimentKeys}
      />
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tab 1: Rapid Comparison (formerly Experiment Comparison)
// ═══════════════════════════════════════════════════════════════════════════════

function RapidComparisonTab({
  sessionId,
  gaugeId,
  experiments,
}: {
  sessionId: string
  gaugeId: string
  experiments: ExperimentInfo[]
}) {
  const [selected, setSelected] = useState<Set<string>>(
    () => new Set(experiments.slice(0, 8).map((e) => e.key))
  )
  const [hydroLog, setHydroLog] = useState(false)
  const [fdcLog, setFdcLog] = useState(true)

  const toggle = useCallback((key: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }, [])

  const selectAll = useCallback(() => {
    setSelected(new Set(experiments.map((e) => e.key)))
  }, [experiments])

  const selectNone = useCallback(() => {
    setSelected(new Set())
  }, [])

  const selectedKeys = useMemo(() => [...selected], [selected])
  const keysParam = selectedKeys.length < experiments.length ? selectedKeys : undefined

  const { data: hydroFig, isLoading: hydroLoading } = useQuery({
    queryKey: ['comparison-hydrograph', sessionId, gaugeId, hydroLog, selectedKeys],
    queryFn: () => getComparisonHydrograph(sessionId, gaugeId, hydroLog, keysParam),
    enabled: selected.size > 0,
  })

  const { data: fdcFig, isLoading: fdcLoading } = useQuery({
    queryKey: ['comparison-fdc', sessionId, gaugeId, fdcLog, selectedKeys],
    queryFn: () => getComparisonFdc(sessionId, gaugeId, fdcLog, keysParam),
    enabled: selected.size > 0,
  })

  const [scatterLog, setScatterLog] = useState(true)

  const { data: scatterFig, isLoading: scatterLoading } = useQuery({
    queryKey: ['comparison-scatter', sessionId, gaugeId, scatterLog, selectedKeys],
    queryFn: () => getComparisonScatter(sessionId, gaugeId, scatterLog, keysParam),
    enabled: selected.size > 0,
  })

  return (
    <div className="space-y-6">
      <ExperimentSelector
        experiments={experiments}
        selected={selected}
        onToggle={toggle}
        onSelectAll={selectAll}
        onSelectNone={selectNone}
      />

      {selected.size === 0 ? (
        <div className="text-center py-12 text-gray-400">
          Select at least one experiment to view comparison plots
        </div>
      ) : (
        <>
          {/* Hydrograph */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-gray-700">Hydrograph Comparison</h3>
              <ScaleToggle value={hydroLog} onChange={setHydroLog} />
            </div>
            <PlotlyChart figure={hydroFig} loading={hydroLoading} />
          </div>

          {/* FDC */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-gray-700">Flow Duration Curve</h3>
              <ScaleToggle value={fdcLog} onChange={setFdcLog} />
            </div>
            <PlotlyChart figure={fdcFig} loading={fdcLoading} />
          </div>

          {/* Scatter */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-gray-700">Observed vs Simulated Scatter</h3>
              <ScaleToggle value={scatterLog} onChange={setScatterLog} />
            </div>
            <PlotlyChart figure={scatterFig} loading={scatterLoading} />
          </div>

          {/* Hydrologic Signatures */}
          <SignatureComparisonCharts
            sessionId={sessionId}
            gaugeId={gaugeId}
            experimentKeys={selectedKeys}
          />
        </>
      )}
    </div>
  )
}

function ScaleToggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center gap-1 text-xs">
      <button
        onClick={() => onChange(false)}
        className={`px-2 py-0.5 rounded ${!value ? 'bg-primary-100 text-primary-700 font-medium' : 'text-gray-500 hover:bg-gray-100'}`}
      >
        Linear
      </button>
      <button
        onClick={() => onChange(true)}
        className={`px-2 py-0.5 rounded ${value ? 'bg-primary-100 text-primary-700 font-medium' : 'text-gray-500 hover:bg-gray-100'}`}
      >
        Log
      </button>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tab 3: Report Cards
// ═══════════════════════════════════════════════════════════════════════════════

const DEFAULT_EXPORT_SECTIONS = [
  'summary', 'metrics', 'hydrograph_linear', 'hydrograph_log',
  'fdc', 'scatter', 'parameters', 'signatures'
]

function ReportCardTab({
  sessionId,
  gaugeId,
  experiments,
}: {
  sessionId: string
  gaugeId: string
  experiments: ExperimentInfo[]
}) {
  const [selectedKey, setSelectedKey] = useState<string>(experiments[0]?.key || '')
  const [scatterLogScale, setScatterLogScale] = useState(false)
  
  // Export state
  const [showExportMenu, setShowExportMenu] = useState(false)
  const [showExportSettings, setShowExportSettings] = useState(false)
  const [exportFormat, setExportFormat] = useState<'pdf' | 'html' | 'interactive'>('pdf')
  const [exportSections, setExportSections] = useState<string[]>(DEFAULT_EXPORT_SECTIONS)
  const [exportBatch, setExportBatch] = useState(false)
  const [isExporting, setIsExporting] = useState(false)

  const { data: reportCard, isLoading } = useQuery<ReportCardResponse>({
    queryKey: ['report-card', sessionId, gaugeId, selectedKey],
    queryFn: () => getExperimentReportCard(sessionId, gaugeId, selectedKey),
    enabled: !!selectedKey,
  })

  const { data: availableSections } = useQuery<ExportSection[]>({
    queryKey: ['export-sections'],
    queryFn: getExportSections,
  })

  const handleExport = async () => {
    if (!exportBatch && !selectedKey) {
      alert('Please select an experiment first')
      return
    }
    
    setIsExporting(true)
    try {
      let blob: Blob
      let filename: string
      
      const ext = exportFormat === 'interactive' ? 'html' : exportFormat
      if (exportBatch) {
        blob = await exportBatchReportCard(sessionId, gaugeId, exportFormat, undefined, exportSections)
        filename = `batch_report${exportFormat === 'interactive' ? '_interactive' : ''}.${ext}`
      } else {
        blob = await exportSingleReportCard(sessionId, gaugeId, selectedKey, exportFormat, exportSections)
        filename = `report_${selectedKey.replace(/[/\\]/g, '_')}${exportFormat === 'interactive' ? '_interactive' : ''}.${ext}`
      }
      
      // Trigger download
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      
      setShowExportMenu(false)
    } catch (error) {
      console.error('Export failed:', error)
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      alert(`Export failed: ${errorMessage}`)
    } finally {
      setIsExporting(false)
    }
  }

  const toggleSection = (sectionId: string) => {
    setExportSections(prev => 
      prev.includes(sectionId) 
        ? prev.filter(s => s !== sectionId)
        : [...prev, sectionId]
    )
  }

  if (!experiments.length) {
    return <div className="text-center py-12 text-gray-400">No experiments available</div>
  }

  const header = reportCard?.header

  return (
    <div className="space-y-4">
      {/* Experiment selector + Export button */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-end justify-between gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Experiment
            </label>
            <select
              value={selectedKey}
              onChange={(e) => setSelectedKey(e.target.value)}
              className="w-full max-w-xl px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              {experiments.map((exp) => {
                const label = [exp.model, exp.objective, exp.transformation, exp.algorithm]
                  .filter(Boolean)
                  .join(' / ')
                return (
                  <option key={exp.key} value={exp.key}>
                    {label} (obj: {exp.best_objective?.toFixed(4) ?? '—'})
                  </option>
                )
              })}
            </select>
          </div>
          
          {/* Export dropdown - temporarily disabled */}
          <div className="relative">
            <button
              disabled
              title="Export temporarily disabled"
              className="flex items-center gap-2 px-4 py-2 bg-gray-400 text-white rounded-lg cursor-not-allowed text-sm font-medium opacity-60"
            >
              <Download className="w-4 h-4" />
              Export Report
              <ChevronDown className="w-4 h-4" />
            </button>
            
            {showExportMenu && (
              <div className="absolute right-0 top-full mt-2 w-80 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
                <div className="p-4 space-y-4">
                  {/* Format selection */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Format</label>
                    <div className="space-y-2">
                      <button
                        onClick={() => setExportFormat('pdf')}
                        className={`w-full px-3 py-2 text-sm rounded-lg border text-left ${exportFormat === 'pdf' ? 'bg-primary-50 border-primary-500 text-primary-700' : 'border-gray-300 text-gray-600 hover:bg-gray-50'}`}
                      >
                        <span className="font-medium">PDF</span>
                        <span className="text-xs ml-2 opacity-75">Landscape A4, single page</span>
                      </button>
                      <button
                        onClick={() => setExportFormat('html')}
                        className={`w-full px-3 py-2 text-sm rounded-lg border text-left ${exportFormat === 'html' ? 'bg-primary-50 border-primary-500 text-primary-700' : 'border-gray-300 text-gray-600 hover:bg-gray-50'}`}
                      >
                        <span className="font-medium">HTML</span>
                        <span className="text-xs ml-2 opacity-75">Static, for printing</span>
                      </button>
                      <button
                        onClick={() => setExportFormat('interactive')}
                        className={`w-full px-3 py-2 text-sm rounded-lg border text-left ${exportFormat === 'interactive' ? 'bg-primary-50 border-primary-500 text-primary-700' : 'border-gray-300 text-gray-600 hover:bg-gray-50'}`}
                      >
                        <span className="font-medium">Interactive HTML</span>
                        <span className="text-xs ml-2 opacity-75">Zoomable Plotly figures</span>
                      </button>
                    </div>
                  </div>
                  
                  {/* Scope selection */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Scope</label>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setExportBatch(false)}
                        className={`flex-1 px-3 py-2 text-sm rounded-lg border ${!exportBatch ? 'bg-primary-50 border-primary-500 text-primary-700' : 'border-gray-300 text-gray-600 hover:bg-gray-50'}`}
                      >
                        Current Experiment
                      </button>
                      <button
                        onClick={() => setExportBatch(true)}
                        className={`flex-1 px-3 py-2 text-sm rounded-lg border ${exportBatch ? 'bg-primary-50 border-primary-500 text-primary-700' : 'border-gray-300 text-gray-600 hover:bg-gray-50'}`}
                      >
                        All Experiments ({experiments.length})
                      </button>
                    </div>
                  </div>
                  
                  {/* Section toggles */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="text-sm font-medium text-gray-700">Sections</label>
                      <button
                        onClick={() => setShowExportSettings(!showExportSettings)}
                        className="text-xs text-primary-600 hover:text-primary-700 flex items-center gap-1"
                      >
                        <Settings className="w-3 h-3" />
                        {showExportSettings ? 'Hide' : 'Customize'}
                      </button>
                    </div>
                    
                    {showExportSettings && availableSections && (
                      <div className="max-h-48 overflow-y-auto border border-gray-200 rounded-lg p-2 space-y-1">
                        {availableSections.map(section => (
                          <label key={section.id} className="flex items-center gap-2 p-1 hover:bg-gray-50 rounded cursor-pointer">
                            <input
                              type="checkbox"
                              checked={exportSections.includes(section.id)}
                              onChange={() => toggleSection(section.id)}
                              className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                            />
                            <span className="text-sm text-gray-700">{section.name}</span>
                          </label>
                        ))}
                      </div>
                    )}
                    
                    {!showExportSettings && (
                      <p className="text-xs text-gray-500">
                        {exportSections.length} of {availableSections?.length ?? 8} sections selected
                      </p>
                    )}
                  </div>
                  
                  {/* Export button */}
                  <button
                    onClick={handleExport}
                    disabled={isExporting || exportSections.length === 0}
                    className="w-full px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-sm font-medium flex items-center justify-center gap-2"
                  >
                    {isExporting ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4" />
                        Download {exportFormat === 'interactive' ? 'Interactive HTML' : exportFormat.toUpperCase()}
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Header info */}
      {header && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h2 className="text-xl font-bold text-gray-900">
            {header.catchment_name}
            {header.gauge_id && <span className="text-gray-500 ml-2">({header.gauge_id})</span>}
          </h2>
          <p className="text-sm text-gray-600 mt-1">
            Method: {header.method} | Objective: {header.objective_name} | 
            Best: {header.best_objective.toFixed(4)}
            {header.area && ` | Area: ${header.area} km²`}
            {` | Period: ${header.period_start} to ${header.period_end}`}
          </p>
        </div>
      )}

      {isLoading && (
        <div className="bg-white rounded-lg border border-gray-200 p-8 text-center text-gray-500">
          Loading report card...
        </div>
      )}

      {reportCard && (
        <>
          {/* Row 1: Hydrographs (stacked) + Metrics Table (full height) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="space-y-4">
              <div className="bg-white rounded-lg border border-gray-200 p-2">
                <PlotlyChart figure={reportCard.hydrograph_linear} />
              </div>
              <div className="bg-white rounded-lg border border-gray-200 p-2">
                <PlotlyChart figure={reportCard.hydrograph_log} />
              </div>
            </div>
            <div className="bg-white rounded-lg border border-gray-200 p-2">
              <PlotlyChart figure={reportCard.metrics_table} />
            </div>
          </div>

          {/* Row 2: FDC + Scatter (with log toggle) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-white rounded-lg border border-gray-200 p-2">
              <PlotlyChart figure={reportCard.fdc} />
            </div>
            <div className="bg-white rounded-lg border border-gray-200 p-2">
              <div className="flex items-center justify-end gap-2 mb-2 px-2">
                <span className="text-xs text-gray-500">Scale:</span>
                <button
                  onClick={() => setScatterLogScale(false)}
                  className={`px-2 py-1 text-xs rounded ${!scatterLogScale ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                >
                  Linear
                </button>
                <button
                  onClick={() => setScatterLogScale(true)}
                  className={`px-2 py-1 text-xs rounded ${scatterLogScale ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                >
                  Log
                </button>
              </div>
              <PlotlyChart 
                figure={reportCard.scatter ? {
                  ...reportCard.scatter,
                  layout: {
                    ...reportCard.scatter.layout,
                    xaxis: { ...reportCard.scatter.layout?.xaxis, type: scatterLogScale ? 'log' : 'linear' },
                    yaxis: { ...reportCard.scatter.layout?.yaxis, type: scatterLogScale ? 'log' : 'linear' },
                  }
                } : undefined} 
              />
            </div>
          </div>

          {/* Row 3: Parameters + Signatures Table (side by side) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {reportCard.parameters && (
              <div className="bg-white rounded-lg border border-gray-200 p-2">
                <PlotlyChart figure={reportCard.parameters} />
              </div>
            )}
            {reportCard.signatures_table && (
              <div className="bg-white rounded-lg border border-gray-200 p-2">
                <PlotlyChart figure={reportCard.signatures_table} />
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
