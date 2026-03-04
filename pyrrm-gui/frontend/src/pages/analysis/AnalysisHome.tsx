import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  FolderOpen,
  Trash2,
  ArrowRight,
  Loader2,
  AlertCircle,
  Package,
} from 'lucide-react'
import {
  listSessions,
  loadSession,
  deleteSession,
  listAvailableBatches,
} from '../../services/analysisApi'

export default function AnalysisHome() {
  const queryClient = useQueryClient()
  const [showLoad, setShowLoad] = useState(false)
  const [selectedBatch, setSelectedBatch] = useState('')
  const [sessionName, setSessionName] = useState('')
  const [loadError, setLoadError] = useState<string | null>(null)

  const { data: sessions, isLoading } = useQuery({
    queryKey: ['analysis-sessions'],
    queryFn: listSessions,
  })

  const { data: batchesData, isLoading: batchesLoading } = useQuery({
    queryKey: ['available-batches'],
    queryFn: listAvailableBatches,
    enabled: showLoad,
  })

  const loadMutation = useMutation({
    mutationFn: () => {
      const batch = batchesData?.batches.find((b) => b.name === selectedBatch)
      if (!batch) throw new Error('No batch selected')
      return loadSession(batch.path, sessionName || batch.name)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['analysis-sessions'] })
      setShowLoad(false)
      setSelectedBatch('')
      setSessionName('')
      setLoadError(null)
    },
    onError: (err: any) => {
      setLoadError(
        err?.response?.data?.detail || err.message || 'Failed to load session'
      )
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteSession,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['analysis-sessions'] })
    },
  })

  const batches = batchesData?.batches ?? []
  const batchesWithPkl = batches.filter((b) => b.pkl_count > 0)
  const selectedInfo = batches.find((b) => b.name === selectedBatch)

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Batch Analysis</h1>
          <p className="text-gray-500 mt-1">
            Load and diagnose batch calibration results
          </p>
        </div>
        <button
          onClick={() => {
            setShowLoad(true)
            setLoadError(null)
            setSelectedBatch('')
            setSessionName('')
          }}
          className="inline-flex items-center px-4 py-2 border border-transparent rounded-lg text-sm font-medium text-white bg-primary-600 hover:bg-primary-700"
        >
          <FolderOpen className="w-4 h-4 mr-2" />
          Load Batch Results
        </button>
      </div>

      {/* Load modal */}
      {showLoad && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl max-w-lg w-full mx-4 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Load Batch Results
            </h2>
            <div className="space-y-4">
              {/* Batch dropdown */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Available Batches
                </label>
                {batchesLoading ? (
                  <div className="flex items-center gap-2 text-sm text-gray-500 py-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Scanning server for batch results…
                  </div>
                ) : batches.length === 0 ? (
                  <div className="text-sm text-gray-500 bg-gray-50 rounded-lg px-3 py-3">
                    <p className="font-medium text-gray-700 mb-1">
                      No batches found on the server
                    </p>
                    <p>
                      Place batch result folders in{' '}
                      <code className="text-xs bg-gray-200 px-1.5 py-0.5 rounded">
                        {batchesData?.root ?? 'pyrrm-gui/batch_results/'}
                      </code>
                    </p>
                  </div>
                ) : (
                  <>
                    <select
                      value={selectedBatch}
                      onChange={(e) => setSelectedBatch(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    >
                      <option value="">Select a batch…</option>
                      {batches.map((b) => (
                        <option
                          key={b.name}
                          value={b.name}
                          disabled={b.pkl_count === 0}
                        >
                          {b.name}
                          {b.pkl_count > 0
                            ? ` (${b.pkl_count} gauge${b.pkl_count !== 1 ? 's' : ''})`
                            : ' (no results found)'}
                        </option>
                      ))}
                    </select>
                    {selectedInfo && selectedInfo.pkl_count > 0 && (
                      <div className="flex items-center gap-2 mt-2 text-sm text-emerald-700 bg-emerald-50 rounded-lg px-3 py-2">
                        <Package className="w-4 h-4" />
                        {selectedInfo.pkl_count} batch_result.pkl file
                        {selectedInfo.pkl_count !== 1 ? 's' : ''} found
                      </div>
                    )}
                    {batchesWithPkl.length === 0 && (
                      <p className="text-xs text-amber-600 mt-1">
                        None of the folders contain batch_result.pkl files yet.
                      </p>
                    )}
                  </>
                )}
              </div>

              {/* Session name */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Session Name (optional)
                </label>
                <input
                  type="text"
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value)}
                  placeholder={selectedBatch || 'e.g. LBG Headwater Calibrations'}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
              </div>

              {loadError && (
                <div className="flex items-start gap-2 text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2">
                  <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  {loadError}
                </div>
              )}
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => {
                  setShowLoad(false)
                  setLoadError(null)
                }}
                className="px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={() => loadMutation.mutate()}
                disabled={
                  !selectedBatch ||
                  !selectedInfo ||
                  selectedInfo.pkl_count === 0 ||
                  loadMutation.isPending
                }
                className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-lg disabled:opacity-50"
              >
                {loadMutation.isPending && (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                )}
                Load
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Sessions list */}
      {isLoading ? (
        <div className="text-center py-12 text-gray-500">Loading sessions…</div>
      ) : !sessions?.length ? (
        <div className="text-center py-16">
          <FolderOpen className="w-16 h-16 mx-auto mb-4 text-gray-300" />
          <p className="text-gray-500">No analysis sessions loaded</p>
          <p className="text-sm text-gray-400 mt-1">
            Click "Load Batch Results" to get started
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {sessions.map((s) => (
            <div
              key={s.id}
              className="bg-white rounded-lg border border-gray-200 p-5 hover:border-primary-200 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <Link
                    to={`/analysis/sessions/${s.id}`}
                    className="text-lg font-semibold text-gray-900 hover:text-primary-600"
                  >
                    {s.name}
                  </Link>
                  <div className="flex gap-6 mt-2 text-sm text-gray-500">
                    <span>
                      {s.gauge_ids.length} gauge
                      {s.gauge_ids.length !== 1 ? 's' : ''}
                    </span>
                    <span>{s.total_experiments} experiments</span>
                    {s.total_failures > 0 && (
                      <span className="text-amber-600">
                        {s.total_failures} failures
                      </span>
                    )}
                    <span>
                      Loaded {new Date(s.loaded_at).toLocaleString()}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <button
                    onClick={() => deleteMutation.mutate(s.id)}
                    className="p-2 text-gray-400 hover:text-red-500 rounded-lg hover:bg-red-50"
                    title="Unload session"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                  <Link
                    to={`/analysis/sessions/${s.id}`}
                    className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-primary-600 hover:bg-primary-50 rounded-lg"
                  >
                    Open <ArrowRight className="w-4 h-4 ml-1" />
                  </Link>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
