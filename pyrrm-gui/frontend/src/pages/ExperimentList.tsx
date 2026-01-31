import { Link, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getExperiments, deleteExperiment, getCatchments } from '../services/api'
import { Plus, Trash2, Play, Eye, FlaskConical } from 'lucide-react'
import type { Experiment } from '../types'

const statusColors: Record<string, string> = {
  draft: 'bg-gray-100 text-gray-800',
  queued: 'bg-yellow-100 text-yellow-800',
  running: 'bg-blue-100 text-blue-800',
  completed: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
  cancelled: 'bg-gray-100 text-gray-600',
}

export default function ExperimentList() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: experiments, isLoading } = useQuery({
    queryKey: ['experiments'],
    queryFn: () => getExperiments(),
  })

  const { data: catchments } = useQuery({
    queryKey: ['catchments'],
    queryFn: getCatchments,
  })

  const deleteMutation = useMutation({
    mutationFn: deleteExperiment,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['experiments'] })
    },
  })

  const handleDelete = (id: string, name: string, e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (confirm(`Are you sure you want to delete experiment "${name}"? This action cannot be undone.`)) {
      deleteMutation.mutate(id)
    }
  }

  const getCatchmentName = (catchmentId: string) => {
    const catchment = catchments?.find(c => c.id === catchmentId)
    return catchment?.name || 'Unknown'
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-600">Loading experiments...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Experiments</h1>
          <p className="text-gray-500 mt-1">Manage and monitor your calibration experiments</p>
        </div>
        <Link
          to="/experiments/new"
          className="flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          <Plus className="h-5 w-5" />
          <span>New Experiment</span>
        </Link>
      </div>

      {/* Experiments Table */}
      {!experiments || experiments.length === 0 ? (
        <div className="bg-white rounded-lg border p-12 text-center">
          <FlaskConical className="h-12 w-12 mx-auto text-gray-300 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No experiments yet</h3>
          <p className="text-gray-500 mb-4">Create your first calibration experiment to get started</p>
          <Link
            to="/experiments/new"
            className="inline-flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
          >
            <Plus className="h-5 w-5" />
            <span>New Experiment</span>
          </Link>
        </div>
      ) : (
        <div className="bg-white rounded-lg border overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Experiment
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Catchment
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Model
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Best Objective
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Created
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {experiments.map((exp: Experiment) => (
                <tr
                  key={exp.id}
                  className="hover:bg-gray-50 cursor-pointer"
                  onClick={() => {
                    if (exp.status === 'completed') {
                      navigate(`/experiments/${exp.id}/results`)
                    } else {
                      navigate(`/experiments/${exp.id}`)
                    }
                  }}
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">{exp.name}</div>
                    <div className="text-xs text-gray-500">{exp.id.slice(0, 8)}...</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {getCatchmentName(exp.catchment_id)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="text-sm text-gray-900 uppercase">{exp.model_type}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${statusColors[exp.status]}`}>
                      {exp.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {exp.best_objective !== null && exp.best_objective !== undefined ? (
                      <span className={exp.best_objective > 0 ? 'text-green-600' : 'text-red-600'}>
                        {exp.best_objective.toFixed(4)}
                      </span>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(exp.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <div className="flex items-center justify-end space-x-2">
                      {exp.status === 'completed' && (
                        <Link
                          to={`/experiments/${exp.id}/results`}
                          onClick={(e) => e.stopPropagation()}
                          className="p-1 text-blue-600 hover:bg-blue-50 rounded"
                          title="View Results"
                        >
                          <Eye className="h-4 w-4" />
                        </Link>
                      )}
                      {exp.status === 'draft' && (
                        <Link
                          to={`/experiments/${exp.id}`}
                          onClick={(e) => e.stopPropagation()}
                          className="p-1 text-green-600 hover:bg-green-50 rounded"
                          title="Run Experiment"
                        >
                          <Play className="h-4 w-4" />
                        </Link>
                      )}
                      <button
                        onClick={(e) => handleDelete(exp.id, exp.name, e)}
                        disabled={exp.status === 'running' || deleteMutation.isPending}
                        className={`p-1 rounded ${
                          exp.status === 'running'
                            ? 'text-gray-300 cursor-not-allowed'
                            : 'text-red-600 hover:bg-red-50'
                        }`}
                        title={exp.status === 'running' ? 'Cannot delete running experiment' : 'Delete'}
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
