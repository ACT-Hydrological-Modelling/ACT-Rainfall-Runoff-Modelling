import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { 
  X, 
  BarChart3
} from 'lucide-react'
import { 
  getExperiments,
  getResult
} from '../services/api'
import type { Experiment, CalibrationResult } from '../types'

interface ComparisonExperiment {
  experiment: Experiment
  result: CalibrationResult
}

export default function ComparisonPage() {
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  
  const { data: experiments } = useQuery({
    queryKey: ['experiments'],
    queryFn: () => getExperiments(),
  })
  
  const completedExperiments = experiments?.filter(e => e.status === 'completed') || []
  
  // Load results for selected experiments
  const { data: comparisonData } = useQuery({
    queryKey: ['comparison', selectedIds],
    queryFn: async () => {
      const results = await Promise.all(
        selectedIds.map(async (id) => {
          const exp = completedExperiments.find(e => e.id === id)
          if (!exp) return null
          const result = await getResult(id)
          return { experiment: exp, result }
        })
      )
      return results.filter(Boolean) as ComparisonExperiment[]
    },
    enabled: selectedIds.length > 0,
  })
  
  const addExperiment = (id: string) => {
    if (!selectedIds.includes(id) && selectedIds.length < 5) {
      setSelectedIds([...selectedIds, id])
    }
  }
  
  const removeExperiment = (id: string) => {
    setSelectedIds(selectedIds.filter(i => i !== id))
  }
  
  return (
    <div>
      <div className="flex justify-between items-start mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Compare Experiments</h1>
          <p className="text-gray-500 mt-1">
            Select up to 5 completed experiments to compare
          </p>
        </div>
      </div>
      
      {/* Experiment selector */}
      <div className="bg-white rounded-lg border border-gray-200 p-6 mb-8">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Select Experiments</h2>
        
        <div className="flex flex-wrap gap-2 mb-4">
          {selectedIds.map((id) => {
            const exp = completedExperiments.find(e => e.id === id)
            if (!exp) return null
            return (
              <span
                key={id}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-primary-100 text-primary-700"
              >
                {exp.name}
                <button
                  onClick={() => removeExperiment(id)}
                  className="ml-2 hover:text-primary-900"
                >
                  <X className="w-4 h-4" />
                </button>
              </span>
            )
          })}
        </div>
        
        <select
          onChange={(e) => {
            if (e.target.value) {
              addExperiment(e.target.value)
              e.target.value = ''
            }
          }}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
          disabled={selectedIds.length >= 5}
        >
          <option value="">Add an experiment...</option>
          {completedExperiments
            .filter(e => !selectedIds.includes(e.id))
            .map((exp) => (
              <option key={exp.id} value={exp.id}>
                {exp.name} ({exp.model_type.toUpperCase()})
              </option>
            ))}
        </select>
      </div>
      
      {/* Comparison results */}
      {comparisonData && comparisonData.length > 0 && (
        <div className="space-y-6">
          {/* Metrics comparison table */}
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">
                <BarChart3 className="w-5 h-5 inline mr-2" />
                Metrics Comparison
              </h2>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Experiment
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Model
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Objective
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      NSE
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      KGE
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      RMSE
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      PBIAS
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {comparisonData.map(({ experiment, result }) => (
                    <tr key={experiment.id}>
                      <td className="px-6 py-4 text-sm font-medium text-gray-900">
                        {experiment.name}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-500">
                        {experiment.model_type.toUpperCase()}
                      </td>
                      <td className="px-6 py-4 text-sm font-medium text-primary-600">
                        {result.best_objective.toFixed(4)}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-900">
                        {result.metrics.NSE?.toFixed(4) || 'N/A'}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-900">
                        {result.metrics.KGE?.toFixed(4) || 'N/A'}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-900">
                        {result.metrics.RMSE?.toFixed(2) || 'N/A'}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-900">
                        {result.metrics.PBIAS?.toFixed(2) || 'N/A'}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          {/* Bar chart comparison */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Objective Function Comparison
            </h2>
            <Plot
              data={[
                {
                  x: comparisonData.map(d => d.experiment.name),
                  y: comparisonData.map(d => d.result.best_objective),
                  type: 'bar',
                  marker: { color: '#3b82f6' }
                }
              ]}
              layout={{
                yaxis: { title: { text: 'Best Objective Value' } },
                margin: { l: 60, r: 20, t: 20, b: 100 },
                height: 300
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
          
          {/* Parameter comparison */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Parameter Values Comparison
            </h2>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                      Parameter
                    </th>
                    {comparisonData.map(({ experiment }) => (
                      <th key={experiment.id} className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                        {experiment.name.substring(0, 20)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {comparisonData[0] && Object.keys(comparisonData[0].result.best_parameters).map((param) => (
                    <tr key={param}>
                      <td className="px-4 py-2 text-sm font-medium text-gray-900">{param}</td>
                      {comparisonData.map(({ experiment, result }) => (
                        <td key={experiment.id} className="px-4 py-2 text-sm text-gray-600">
                          {result.best_parameters[param]?.toFixed(4) || 'N/A'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
      
      {selectedIds.length === 0 && (
        <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
          <BarChart3 className="w-12 h-12 mx-auto text-gray-300" />
          <p className="mt-4 text-gray-500">Select experiments to compare their results</p>
        </div>
      )}
    </div>
  )
}
