import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { 
  Download, 
  ChevronDown,
  BarChart3,
  LineChart,
  ScatterChart
} from 'lucide-react'
import { 
  getExperiment,
  getResult,
  getParameters,
  getMetrics,
  getHydrographPlot,
  getFdcPlot,
  getScatterPlot,
  getParametersPlot
} from '../services/api'

function MetricsCard({ metrics }: { metrics: Record<string, { value: number | string; color: string }> }) {
  return (
    <div className="grid grid-cols-3 gap-4">
      {Object.entries(metrics).map(([name, { value, color }]) => (
        <div key={name} className="bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-sm text-gray-500">{name}</p>
          <p className={`text-2xl font-bold ${
            color === 'green' ? 'text-green-600' :
            color === 'orange' ? 'text-yellow-600' :
            color === 'red' ? 'text-red-600' : 'text-gray-600'
          }`}>
            {typeof value === 'number' ? value.toFixed(4) : value}
          </p>
        </div>
      ))}
    </div>
  )
}

function PlotCard({ 
  title, 
  icon: Icon, 
  children 
}: { 
  title: string
  icon: React.ElementType
  children: React.ReactNode 
}) {
  const [expanded, setExpanded] = useState(true)
  
  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-6 py-4 flex items-center justify-between border-b border-gray-200"
      >
        <div className="flex items-center">
          <Icon className="w-5 h-5 mr-2 text-gray-500" />
          <h3 className="font-semibold text-gray-900">{title}</h3>
        </div>
        <ChevronDown className={`w-5 h-5 text-gray-500 transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>
      {expanded && (
        <div className="p-4">
          {children}
        </div>
      )}
    </div>
  )
}

export default function ResultsPage() {
  const { id } = useParams<{ id: string }>()
  const [logScale, setLogScale] = useState(false)
  
  const { data: experiment } = useQuery({
    queryKey: ['experiment', id],
    queryFn: () => getExperiment(id!),
    enabled: !!id,
  })
  
  const { data: result } = useQuery({
    queryKey: ['result', id],
    queryFn: () => getResult(id!),
    enabled: !!id,
  })
  
  const { data: parameters } = useQuery({
    queryKey: ['parameters', id],
    queryFn: () => getParameters(id!),
    enabled: !!id,
  })
  
  const { data: metrics } = useQuery({
    queryKey: ['metrics', id],
    queryFn: () => getMetrics(id!),
    enabled: !!id,
  })
  
  const { data: hydrographData } = useQuery({
    queryKey: ['hydrograph', id, logScale],
    queryFn: () => getHydrographPlot(id!, logScale),
    enabled: !!id,
  })
  
  const { data: fdcData } = useQuery({
    queryKey: ['fdc', id],
    queryFn: () => getFdcPlot(id!),
    enabled: !!id,
  })
  
  const { data: scatterData } = useQuery({
    queryKey: ['scatter', id],
    queryFn: () => getScatterPlot(id!),
    enabled: !!id,
  })
  
  const { data: paramsPlotData } = useQuery({
    queryKey: ['paramsPlot', id],
    queryFn: () => getParametersPlot(id!),
    enabled: !!id,
  })
  
  if (!experiment || !result) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin w-8 h-8 border-4 border-primary-500 border-t-transparent rounded-full mx-auto"></div>
        <p className="mt-4 text-gray-500">Loading results...</p>
      </div>
    )
  }
  
  return (
    <div>
      {/* Header */}
      <div className="flex justify-between items-start mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{experiment.name}</h1>
          <p className="text-gray-500 mt-1">
            Calibration Results • {experiment.model_type.toUpperCase()} • 
            {result.runtime_seconds ? ` Runtime: ${(result.runtime_seconds / 60).toFixed(1)} min` : ''}
          </p>
        </div>
        <div className="flex gap-3">
          <button
            className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
          >
            <Download className="w-4 h-4 mr-2" />
            Export Report
          </button>
        </div>
      </div>
      
      {/* Best Result Summary */}
      <div className="bg-gradient-to-r from-primary-500 to-primary-600 rounded-lg p-6 mb-8 text-white">
        <p className="text-primary-100 text-sm">Best {experiment.objective_config?.type}</p>
        <p className="text-4xl font-bold mt-1">{result.best_objective.toFixed(6)}</p>
      </div>
      
      {/* Metrics */}
      {metrics && (
        <div className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h2>
          <MetricsCard metrics={metrics} />
        </div>
      )}
      
      {/* Plots */}
      <div className="space-y-6">
        {/* Hydrograph */}
        <PlotCard title="Hydrograph Comparison" icon={LineChart}>
          <div className="mb-4">
            <label className="inline-flex items-center">
              <input
                type="checkbox"
                checked={logScale}
                onChange={(e) => setLogScale(e.target.checked)}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
              <span className="ml-2 text-sm text-gray-600">Log Scale</span>
            </label>
          </div>
          {hydrographData && (
            <Plot
              data={hydrographData.data}
              layout={{
                ...hydrographData.layout,
                autosize: true,
                margin: { l: 60, r: 20, t: 40, b: 40 }
              }}
              config={{ responsive: true }}
              style={{ width: '100%', height: '400px' }}
            />
          )}
        </PlotCard>
        
        {/* FDC and Scatter side by side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <PlotCard title="Flow Duration Curve" icon={BarChart3}>
            {fdcData && (
              <Plot
                data={fdcData.data}
                layout={{
                  ...fdcData.layout,
                  autosize: true,
                  margin: { l: 60, r: 20, t: 40, b: 40 }
                }}
                config={{ responsive: true }}
                style={{ width: '100%', height: '350px' }}
              />
            )}
          </PlotCard>
          
          <PlotCard title="Observed vs Simulated" icon={ScatterChart}>
            {scatterData && (
              <Plot
                data={scatterData.data}
                layout={{
                  ...scatterData.layout,
                  autosize: true,
                  margin: { l: 60, r: 20, t: 40, b: 40 }
                }}
                config={{ responsive: true }}
                style={{ width: '100%', height: '350px' }}
              />
            )}
          </PlotCard>
        </div>
        
        {/* Parameters */}
        <PlotCard title="Calibrated Parameters" icon={BarChart3}>
          {paramsPlotData && (
            <Plot
              data={paramsPlotData.data}
              layout={{
                ...paramsPlotData.layout,
                autosize: true,
                margin: { l: 100, r: 20, t: 40, b: 40 }
              }}
              config={{ responsive: true }}
              style={{ width: '100%', height: '400px' }}
            />
          )}
          
          {/* Parameters table */}
          {parameters && (
            <div className="mt-6 overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Parameter</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Value</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Min</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Max</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">% of Range</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {parameters.map((param) => (
                    <tr key={param.name}>
                      <td className="px-4 py-3 text-sm font-medium text-gray-900">{param.name}</td>
                      <td className="px-4 py-3 text-sm text-gray-900">{param.value.toFixed(4)}</td>
                      <td className="px-4 py-3 text-sm text-gray-500">{param.min_bound.toFixed(2)}</td>
                      <td className="px-4 py-3 text-sm text-gray-500">{param.max_bound.toFixed(2)}</td>
                      <td className="px-4 py-3 text-sm">
                        <div className="flex items-center">
                          <div className="w-24 bg-gray-200 rounded-full h-2 mr-2">
                            <div 
                              className="bg-primary-600 h-2 rounded-full"
                              style={{ width: `${param.percent_of_range}%` }}
                            />
                          </div>
                          <span className="text-gray-600">{param.percent_of_range.toFixed(1)}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </PlotCard>
      </div>
      
      {/* Back link */}
      <div className="mt-8">
        <Link
          to={`/catchments/${experiment.catchment_id}`}
          className="text-sm text-primary-600 hover:text-primary-700"
        >
          ← Back to Catchment
        </Link>
      </div>
    </div>
  )
}
