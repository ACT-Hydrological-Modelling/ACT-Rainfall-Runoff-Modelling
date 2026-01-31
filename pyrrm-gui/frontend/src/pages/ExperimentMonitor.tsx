import { useState, useEffect, useRef, useMemo } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { 
  Play, 
  Pause, 
  CheckCircle2, 
  XCircle, 
  Clock,
  RefreshCw,
  ArrowRight,
  Terminal,
  TrendingUp,
  Timer,
  BarChart3,
  Activity,
  Settings
} from 'lucide-react'
import { 
  getExperiment, 
  runExperiment, 
  cancelExperiment
} from '../services/api'
import type { 
  CalibrationProgress, 
  ExperimentStatus as ExpStatus, 
  LogMessage,
  SimulationData,
  ObjectiveHistoryPoint
} from '../types'

function ProgressBar({ value, max }: { value: number; max?: number }) {
  const percent = max ? (value / max) * 100 : 0
  
  return (
    <div className="w-full bg-gray-200 rounded-full h-4">
      <div 
        className="bg-primary-600 h-4 rounded-full transition-all duration-500"
        style={{ width: `${Math.min(percent, 100)}%` }}
      />
    </div>
  )
}

function StatusIndicator({ status }: { status: ExpStatus }) {
  const config = {
    draft: { color: 'text-gray-500', bg: 'bg-gray-100', icon: Clock, text: 'Draft' },
    queued: { color: 'text-yellow-600', bg: 'bg-yellow-100', icon: Clock, text: 'Queued' },
    running: { color: 'text-blue-600', bg: 'bg-blue-100', icon: RefreshCw, text: 'Running' },
    completed: { color: 'text-green-600', bg: 'bg-green-100', icon: CheckCircle2, text: 'Completed' },
    failed: { color: 'text-red-600', bg: 'bg-red-100', icon: XCircle, text: 'Failed' },
    cancelled: { color: 'text-gray-500', bg: 'bg-gray-100', icon: XCircle, text: 'Cancelled' },
  }[status]
  
  const Icon = config.icon
  
  return (
    <div className={`inline-flex items-center px-4 py-2 rounded-lg ${config.bg}`}>
      <Icon className={`w-5 h-5 mr-2 ${config.color} ${status === 'running' ? 'animate-spin' : ''}`} />
      <span className={`font-medium ${config.color}`}>{config.text}</span>
    </div>
  )
}

// Real-time visualization charts
function ObjectiveEvolutionChart({ history }: { history: ObjectiveHistoryPoint[] }) {
  const data = useMemo(() => [{
    x: history.map(h => h.nfev),
    y: history.map(h => h.objective),
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: 'Best Objective',
    line: { color: '#3b82f6', width: 2 }
  }], [history])

  return (
    <Plot
      data={data}
      layout={{
        height: 200,
        margin: { l: 50, r: 20, t: 10, b: 40 },
        xaxis: { title: { text: 'Function Evaluations' }, gridcolor: '#e5e7eb' },
        yaxis: { title: { text: 'Objective' }, gridcolor: '#e5e7eb' },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { size: 11 }
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}

function HydrographChart({ data }: { data: SimulationData['hydrograph'] }) {
  const plotData = useMemo(() => [
    {
      x: data.dates,
      y: data.observed,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Observed',
      line: { color: '#1f77b4', width: 1.5 }
    },
    {
      x: data.dates,
      y: data.simulated,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Simulated',
      line: { color: '#ff7f0e', width: 1.5 }
    }
  ], [data])

  return (
    <Plot
      data={plotData}
      layout={{
        height: 250,
        margin: { l: 60, r: 20, t: 10, b: 40 },
        xaxis: { title: { text: 'Date' }, gridcolor: '#e5e7eb', tickangle: -45 },
        yaxis: { title: { text: 'Flow (ML/day)' }, gridcolor: '#e5e7eb' },
        legend: { orientation: 'h', y: 1.1 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { size: 11 }
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}

function FDCChart({ data }: { data: SimulationData['fdc'] }) {
  const plotData = useMemo(() => [
    {
      x: data.exceedance,
      y: data.observed,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Observed',
      line: { color: '#1f77b4', width: 2 }
    },
    {
      x: data.exceedance,
      y: data.simulated,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Simulated',
      line: { color: '#ff7f0e', width: 2 }
    }
  ], [data])

  return (
    <Plot
      data={plotData}
      layout={{
        height: 250,
        margin: { l: 60, r: 20, t: 10, b: 40 },
        xaxis: { title: { text: 'Exceedance (%)' }, gridcolor: '#e5e7eb' },
        yaxis: { title: { text: 'Flow (ML/day)' }, type: 'log', gridcolor: '#e5e7eb' },
        legend: { orientation: 'h', y: 1.1 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { size: 11 }
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}

function ScatterChart({ data }: { data: SimulationData['scatter'] }) {
  const maxVal = Math.max(
    ...data.observed.filter(v => v != null),
    ...data.simulated.filter(v => v != null)
  )
  
  const plotData = useMemo(() => [
    {
      x: data.observed,
      y: data.simulated,
      type: 'scatter' as const,
      mode: 'markers' as const,
      name: 'Data',
      marker: { color: '#3b82f6', size: 4, opacity: 0.5 }
    },
    {
      x: [0, maxVal],
      y: [0, maxVal],
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: '1:1 Line',
      line: { color: '#dc2626', dash: 'dash' as const, width: 1 }
    }
  ], [data, maxVal])

  return (
    <Plot
      data={plotData}
      layout={{
        height: 250,
        margin: { l: 60, r: 20, t: 10, b: 40 },
        xaxis: { title: { text: 'Observed (ML/day)' }, gridcolor: '#e5e7eb' },
        yaxis: { title: { text: 'Simulated (ML/day)' }, gridcolor: '#e5e7eb' },
        showlegend: false,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { size: 11 }
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}

function ParameterTable({ 
  parameters, 
  parameterNames 
}: { 
  parameters: Record<string, number> | number[]
  parameterNames?: string[] 
}) {
  const entries = Array.isArray(parameters) 
    ? parameters.map((val, idx) => [parameterNames?.[idx] || `p${idx}`, val])
    : Object.entries(parameters)

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
      {entries.map(([name, value], idx) => (
        <div key={idx} className="bg-gray-50 rounded-lg p-3 border border-gray-100">
          <p className="text-xs text-gray-500 truncate" title={String(name)}>{name}</p>
          <p className="text-sm font-mono font-semibold text-gray-900">
            {typeof value === 'number' ? value.toFixed(4) : String(value)}
          </p>
        </div>
      ))}
    </div>
  )
}

export default function ExperimentMonitor() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [progress, setProgress] = useState<CalibrationProgress | null>(null)
  const [logMessages, setLogMessages] = useState<LogMessage[]>([])
  const [objectiveHistory, setObjectiveHistory] = useState<ObjectiveHistoryPoint[]>([])
  const [simulationData, setSimulationData] = useState<SimulationData | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const logEndRef = useRef<HTMLDivElement>(null)
  
  const { data: experiment, isLoading } = useQuery({
    queryKey: ['experiment', id],
    queryFn: () => getExperiment(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      const data = query.state.data
      return data?.status === 'running' || data?.status === 'queued' ? 5000 : false
    },
  })
  
  const runMutation = useMutation({
    mutationFn: () => runExperiment(id!),
    onSuccess: () => {
      // Reset state for new run
      setProgress(null)
      setLogMessages([])
      setObjectiveHistory([])
      setSimulationData(null)
      queryClient.invalidateQueries({ queryKey: ['experiment', id] })
    },
  })
  
  const cancelMutation = useMutation({
    mutationFn: () => cancelExperiment(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['experiment', id] })
    },
  })
  
  // WebSocket connection for real-time progress
  useEffect(() => {
    const status = experiment?.status
    if (status === 'running' || status === 'queued') {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/experiments/${id}/progress`)
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if (data.type === 'progress') {
          setProgress(data.data)
          
          // Update log messages
          if (data.data.log_messages) {
            setLogMessages(data.data.log_messages)
          }
          
          // Update objective history (accumulate)
          if (data.data.objective_history) {
            setObjectiveHistory(prev => {
              const newHistory = data.data.objective_history as ObjectiveHistoryPoint[]
              // Merge: keep existing entries not in new batch, add new entries
              const existingNfevs = new Set(newHistory.map((h: ObjectiveHistoryPoint) => h.nfev))
              const filtered = prev.filter(p => !existingNfevs.has(p.nfev))
              const merged = [...filtered, ...newHistory].sort((a, b) => a.nfev - b.nfev)
              // Keep last 500 points
              return merged.slice(-500)
            })
          }
          
          // Update simulation data when available
          if (data.data.simulation_data) {
            setSimulationData(data.data.simulation_data)
          }
        } else if (data.type === 'started') {
          setLogMessages([{
            time: new Date().toISOString(),
            message: "Calibration started...",
            level: 'info'
          }])
        } else if (data.type === 'completed') {
          const successMsg = data.data.success 
            ? `Calibration completed successfully! Best objective: ${data.data.best_objective?.toFixed(6)}`
            : `Calibration failed: ${data.data.error}`
          setLogMessages(prev => [...prev, {
            time: new Date().toISOString(),
            message: successMsg,
            level: data.data.success ? 'success' : 'error'
          }])
          queryClient.invalidateQueries({ queryKey: ['experiment', id] })
          if (data.data.success) {
            setTimeout(() => navigate(`/experiments/${id}/results`), 2000)
          }
        }
      }
      
      wsRef.current = ws
      
      return () => {
        ws.close()
      }
    }
  }, [experiment?.status, id, queryClient, navigate])
  
  // Auto-scroll log to bottom
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logMessages])
  
  if (isLoading) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin w-8 h-8 border-4 border-primary-500 border-t-transparent rounded-full mx-auto"></div>
        <p className="mt-4 text-gray-500">Loading experiment...</p>
      </div>
    )
  }
  
  if (!experiment) {
    return (
      <div className="text-center py-12">
        <XCircle className="w-12 h-12 mx-auto text-red-500" />
        <p className="mt-4 text-gray-900">Experiment not found</p>
      </div>
    )
  }
  
  const isRunning = experiment.status === 'running'
  const isQueued = experiment.status === 'queued'
  const isCompleted = experiment.status === 'completed'
  const canStart = experiment.status === 'draft' || experiment.status === 'failed' || experiment.status === 'cancelled'
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{experiment.name}</h1>
          <p className="text-gray-500 mt-1">
            {experiment.model_type.toUpperCase()} • 
            Created {new Date(experiment.created_at).toLocaleDateString()}
          </p>
        </div>
        <StatusIndicator status={experiment.status as ExpStatus} />
      </div>
      
      {/* Progress Section */}
      {(isRunning || isQueued) && (
        <>
          {/* Top Row: Progress Overview + Objective Evolution */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Progress Overview Card */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                  <Activity className="w-5 h-5 mr-2 text-primary-600" />
                  Progress
                </h2>
                {progress && (
                  <div className="flex items-center space-x-3 text-sm text-gray-500">
                    <span className="flex items-center">
                      <Timer className="w-4 h-4 mr-1" />
                      {progress.elapsed_time || '0s'}
                    </span>
                    {progress.eta && (
                      <span className="flex items-center">
                        <Clock className="w-4 h-4 mr-1" />
                        ETA: {progress.eta}
                      </span>
                    )}
                  </div>
                )}
              </div>
              
              {progress ? (
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm text-gray-600 mb-2">
                      <span>Evaluations: {progress.nfev?.toLocaleString() || 0} / {(progress.total_iterations || 50000).toLocaleString()}</span>
                      <span className="font-medium">{(progress.progress_percent || 0).toFixed(1)}%</span>
                    </div>
                    <ProgressBar 
                      value={progress.nfev || 0} 
                      max={progress.total_iterations || 50000} 
                    />
                  </div>
                  
                  <div className="grid grid-cols-3 gap-3">
                    <div className={`rounded-lg p-3 ${progress.improved ? 'bg-green-50 border border-green-200' : 'bg-gray-50'}`}>
                      <div className="flex items-center space-x-1">
                        <TrendingUp className={`w-4 h-4 ${progress.improved ? 'text-green-600' : 'text-gray-400'}`} />
                        <p className="text-xs text-gray-500">Best Objective</p>
                      </div>
                      <p className={`text-xl font-bold ${progress.improved ? 'text-green-700' : 'text-gray-900'}`}>
                        {progress.best_objective?.toFixed(4) || '—'}
                      </p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3">
                      <p className="text-xs text-gray-500">Iteration</p>
                      <p className="text-xl font-bold text-gray-900">
                        {progress.iteration?.toLocaleString() || 0}
                      </p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3">
                      <p className="text-xs text-gray-500">No Improve</p>
                      <p className={`text-xl font-bold ${(progress.no_improve_count || 0) > 50 ? 'text-amber-600' : 'text-gray-900'}`}>
                        {progress.no_improve_count || 0}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-6">
                  <RefreshCw className="w-8 h-8 mx-auto text-gray-400 animate-spin" />
                  <p className="mt-2 text-gray-500">
                    {isQueued ? 'Waiting in queue...' : 'Connecting...'}
                  </p>
                </div>
              )}
            </div>
            
            {/* Objective Evolution Chart */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center mb-4">
                <BarChart3 className="w-5 h-5 mr-2 text-primary-600" />
                Objective Evolution
              </h2>
              {objectiveHistory.length > 1 ? (
                <ObjectiveEvolutionChart history={objectiveHistory} />
              ) : (
                <div className="h-48 flex items-center justify-center text-gray-400">
                  Waiting for data...
                </div>
              )}
            </div>
          </div>
          
          {/* Middle Row: Hydrograph and FDC */}
          {simulationData && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Hydrograph */}
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">
                  Hydrograph Comparison
                </h2>
                <HydrographChart data={simulationData.hydrograph} />
              </div>
              
              {/* FDC */}
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">
                  Flow Duration Curve
                </h2>
                <FDCChart data={simulationData.fdc} />
              </div>
            </div>
          )}
          
          {/* Bottom Row: Scatter Plot and Parameters */}
          {simulationData && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Scatter Plot */}
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">
                  Observed vs Simulated
                </h2>
                <ScatterChart data={simulationData.scatter} />
              </div>
              
              {/* Parameters */}
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center mb-4">
                  <Settings className="w-5 h-5 mr-2 text-primary-600" />
                  Current Best Parameters
                </h2>
                <ParameterTable 
                  parameters={simulationData.parameters} 
                  parameterNames={progress?.parameter_names}
                />
              </div>
            </div>
          )}
          
          {/* Log Console */}
          <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
            <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
              <div className="flex items-center space-x-2">
                <Terminal className="w-4 h-4 text-green-400" />
                <span className="text-sm font-medium text-gray-200">Optimization Log</span>
              </div>
              <div className="flex items-center space-x-4">
                <span className="text-xs text-gray-400">
                  {logMessages.length} messages
                </span>
                <button
                  onClick={() => cancelMutation.mutate()}
                  disabled={cancelMutation.isPending}
                  className="px-3 py-1 text-xs font-medium text-red-400 hover:text-red-300 border border-red-700 rounded hover:bg-red-900/30"
                >
                  <Pause className="w-3 h-3 inline mr-1" />
                  Cancel
                </button>
              </div>
            </div>
            <div className="h-48 overflow-y-auto p-4 font-mono text-sm">
              {logMessages.length === 0 ? (
                <div className="text-gray-500 text-center py-8">
                  Waiting for log messages...
                </div>
              ) : (
                <div className="space-y-1">
                  {logMessages.map((log, idx) => (
                    <div 
                      key={idx} 
                      className={`
                        ${log.level === 'success' ? 'text-green-400' : ''}
                        ${log.level === 'error' ? 'text-red-400' : ''}
                        ${log.level === 'warning' ? 'text-yellow-400' : ''}
                        ${log.level === 'info' ? 'text-gray-300' : ''}
                      `}
                    >
                      <span className="text-gray-500 text-xs mr-2">
                        {new Date(log.time).toLocaleTimeString()}
                      </span>
                      {log.message}
                    </div>
                  ))}
                  <div ref={logEndRef} />
                </div>
              )}
            </div>
          </div>
        </>
      )}
      
      {/* Error Message */}
      {experiment.status === 'failed' && experiment.error_message && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-red-900 mb-2">Calibration Failed</h2>
          <pre className="text-sm text-red-700 whitespace-pre-wrap overflow-auto max-h-48">
            {experiment.error_message}
          </pre>
        </div>
      )}
      
      {/* Configuration Summary */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Configuration</h2>
        
        <dl className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div>
            <dt className="text-sm text-gray-500">Model</dt>
            <dd className="font-medium text-gray-900">{experiment.model_type.toUpperCase()}</dd>
          </div>
          <div>
            <dt className="text-sm text-gray-500">Objective</dt>
            <dd className="font-medium text-gray-900">
              {experiment.objective_config?.type}
              {experiment.objective_config?.transform !== 'none' && 
                ` (${experiment.objective_config?.transform})`}
            </dd>
          </div>
          <div>
            <dt className="text-sm text-gray-500">Algorithm</dt>
            <dd className="font-medium text-gray-900">{experiment.algorithm_config?.method}</dd>
          </div>
          <div>
            <dt className="text-sm text-gray-500">Max Evaluations</dt>
            <dd className="font-medium text-gray-900">
              {experiment.algorithm_config?.max_evals?.toLocaleString()}
            </dd>
          </div>
          <div>
            <dt className="text-sm text-gray-500">Calibration Period</dt>
            <dd className="font-medium text-gray-900">
              {experiment.calibration_period?.start_date} to {experiment.calibration_period?.end_date}
            </dd>
          </div>
          <div>
            <dt className="text-sm text-gray-500">Warmup</dt>
            <dd className="font-medium text-gray-900">
              {experiment.calibration_period?.warmup_days} days
            </dd>
          </div>
        </dl>
      </div>
      
      {/* Actions */}
      <div className="flex justify-between">
        <Link
          to={`/catchments/${experiment.catchment_id}`}
          className="text-sm text-primary-600 hover:text-primary-700"
        >
          ← Back to Catchment
        </Link>
        
        <div className="flex gap-3">
          {canStart && (
            <button
              onClick={() => runMutation.mutate()}
              disabled={runMutation.isPending}
              className="inline-flex items-center px-4 py-2 border border-transparent rounded-lg text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 disabled:opacity-50"
            >
              <Play className="w-4 h-4 mr-2" />
              {runMutation.isPending ? 'Starting...' : 'Start Calibration'}
            </button>
          )}
          
          {isCompleted && (
            <Link
              to={`/experiments/${id}/results`}
              className="inline-flex items-center px-4 py-2 border border-transparent rounded-lg text-sm font-medium text-white bg-green-600 hover:bg-green-700"
            >
              View Results
              <ArrowRight className="w-4 h-4 ml-2" />
            </Link>
          )}
        </div>
      </div>
    </div>
  )
}
