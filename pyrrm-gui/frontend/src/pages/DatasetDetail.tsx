import { useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { 
  getDatasetPreview, 
  getDatasetStatistics, 
  getDatasetPlot, 
  deleteDataset,
  cleanDataset,
  CleaningOptions 
} from '../services/api'
import { 
  ArrowLeft, 
  Trash2, 
  Database, 
  Calendar, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  Sparkles,
  RefreshCw
} from 'lucide-react'

export default function DatasetDetail() {
  const { datasetId, catchmentId } = useParams<{ datasetId: string; catchmentId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  
  // Cleaning options state
  const [showCleaningModal, setShowCleaningModal] = useState(false)
  const [cleaningOptions, setCleaningOptions] = useState<CleaningOptions>({
    replace_sentinel: true,
    replace_negative: true,
    sentinel_values: [-9999, -999, -99, -1],
    drop_na: false,
    interpolate: false,
    max_interpolate_gap: 3
  })

  const { data: preview, isLoading: previewLoading } = useQuery({
    queryKey: ['dataset-preview', datasetId],
    queryFn: () => getDatasetPreview(datasetId!),
    enabled: !!datasetId,
  })

  const { data: statistics, isLoading: statsLoading } = useQuery({
    queryKey: ['dataset-statistics', datasetId],
    queryFn: () => getDatasetStatistics(datasetId!),
    enabled: !!datasetId,
  })

  const { data: plot, isLoading: plotLoading } = useQuery({
    queryKey: ['dataset-plot', datasetId],
    queryFn: () => getDatasetPlot(datasetId!),
    enabled: !!datasetId,
  })

  const deleteMutation = useMutation({
    mutationFn: () => deleteDataset(datasetId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      queryClient.invalidateQueries({ queryKey: ['catchment'] })
      navigate(`/catchments/${catchmentId}`)
    },
  })

  const cleanMutation = useMutation({
    mutationFn: (options: CleaningOptions) => cleanDataset(datasetId!, options),
    onSuccess: () => {
      // Refresh all dataset data
      queryClient.invalidateQueries({ queryKey: ['dataset-preview', datasetId] })
      queryClient.invalidateQueries({ queryKey: ['dataset-statistics', datasetId] })
      queryClient.invalidateQueries({ queryKey: ['dataset-plot', datasetId] })
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      queryClient.invalidateQueries({ queryKey: ['catchment'] })
      setShowCleaningModal(false)
    },
  })

  const handleDelete = () => {
    if (confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) {
      deleteMutation.mutate()
    }
  }

  const handleClean = () => {
    cleanMutation.mutate(cleaningOptions)
  }

  if (previewLoading || statsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-600">Loading dataset...</span>
      </div>
    )
  }

  const typeColors: Record<string, string> = {
    rainfall: 'bg-blue-100 text-blue-800',
    pet: 'bg-orange-100 text-orange-800',
    observed_flow: 'bg-green-100 text-green-800',
  }

  const typeLabels: Record<string, string> = {
    rainfall: 'Rainfall',
    pet: 'PET',
    observed_flow: 'Observed Flow',
  }

  // Get data quality info from statistics
  const dataQuality = statistics?.data_quality
  const hasQualityIssues = dataQuality?.has_issues

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Link
            to={`/catchments/${catchmentId}`}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{preview?.name}</h1>
            <div className="flex items-center space-x-3 mt-1">
              <span className={`px-2 py-1 text-xs font-medium rounded-full ${typeColors[preview?.type || 'rainfall']}`}>
                {typeLabels[preview?.type || 'rainfall']}
              </span>
              <span className="text-sm text-gray-500">
                {preview?.record_count?.toLocaleString()} records
              </span>
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {hasQualityIssues && (
            <button
              onClick={() => setShowCleaningModal(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Sparkles className="h-4 w-4" />
              <span>Clean Data</span>
            </button>
          )}
          <button
            onClick={handleDelete}
            disabled={deleteMutation.isPending}
            className="flex items-center space-x-2 px-4 py-2 text-red-600 border border-red-200 rounded-lg hover:bg-red-50 transition-colors"
          >
            <Trash2 className="h-4 w-4" />
            <span>{deleteMutation.isPending ? 'Deleting...' : 'Delete Dataset'}</span>
          </button>
        </div>
      </div>

      {/* Date Range & Data Quality Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white rounded-lg border p-4">
          <div className="flex items-center space-x-2 text-gray-600 mb-2">
            <Calendar className="h-4 w-4" />
            <span className="text-sm font-medium">Date Range</span>
          </div>
          <p className="text-lg font-semibold">
            {preview?.start_date} to {preview?.end_date}
          </p>
        </div>
        <div className={`rounded-lg border p-4 ${hasQualityIssues ? 'bg-amber-50 border-amber-200' : 'bg-green-50 border-green-200'}`}>
          <div className="flex items-center space-x-2 mb-2">
            {hasQualityIssues ? (
              <AlertTriangle className="h-4 w-4 text-amber-600" />
            ) : (
              <CheckCircle className="h-4 w-4 text-green-600" />
            )}
            <span className={`text-sm font-medium ${hasQualityIssues ? 'text-amber-700' : 'text-green-700'}`}>
              Data Quality
            </span>
          </div>
          <p className={`text-lg font-semibold ${hasQualityIssues ? 'text-amber-900' : 'text-green-900'}`}>
            {dataQuality?.clean_records?.toLocaleString() || 0}/{dataQuality?.total_records?.toLocaleString() || 0} clean records
            {dataQuality?.issue_percentage != null && dataQuality.issue_percentage > 0 && (
              <span className="text-sm font-normal ml-2">
                ({dataQuality.issue_percentage.toFixed(1)}% have issues)
              </span>
            )}
          </p>
        </div>
      </div>

      {/* Data Quality Issues Alert */}
      {hasQualityIssues && dataQuality && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="h-5 w-5 text-amber-600 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="font-semibold text-amber-800">Data Quality Issues Detected</h3>
              <p className="text-sm text-amber-700 mt-1">
                This dataset contains values that need to be cleaned before calibration.
              </p>
              
              {/* Issue breakdown */}
              <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3">
                {dataQuality.sentinel_values > 0 && (
                  <div className="bg-white rounded-lg p-2 border border-amber-200">
                    <div className="flex items-center space-x-2">
                      <XCircle className="h-4 w-4 text-red-500" />
                      <span className="text-sm font-medium text-gray-800">{dataQuality.sentinel_values.toLocaleString()}</span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Sentinel values (-9999, etc.)</div>
                  </div>
                )}
                {dataQuality.negative_values > 0 && (
                  <div className="bg-white rounded-lg p-2 border border-amber-200">
                    <div className="flex items-center space-x-2">
                      <XCircle className="h-4 w-4 text-red-500" />
                      <span className="text-sm font-medium text-gray-800">{dataQuality.negative_values.toLocaleString()}</span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Negative values</div>
                  </div>
                )}
                {dataQuality.nan_values > 0 && (
                  <div className="bg-white rounded-lg p-2 border border-amber-200">
                    <div className="flex items-center space-x-2">
                      <Database className="h-4 w-4 text-gray-500" />
                      <span className="text-sm font-medium text-gray-800">{dataQuality.nan_values.toLocaleString()}</span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Missing values (NaN)</div>
                  </div>
                )}
                {dataQuality.potential_outliers > 0 && (
                  <div className="bg-white rounded-lg p-2 border border-amber-200">
                    <div className="flex items-center space-x-2">
                      <AlertTriangle className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm font-medium text-gray-800">{dataQuality.potential_outliers.toLocaleString()}</span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Potential outliers</div>
                  </div>
                )}
              </div>

              {/* Issues list */}
              {dataQuality.issues && dataQuality.issues.length > 0 && (
                <ul className="mt-3 list-disc list-inside text-sm text-amber-700 space-y-1">
                  {dataQuality.issues.slice(0, 5).map((issue: string, i: number) => (
                    <li key={i}>{issue}</li>
                  ))}
                </ul>
              )}

              {/* Clean Data CTA */}
              <div className="mt-4">
                <button
                  onClick={() => setShowCleaningModal(true)}
                  className="inline-flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Sparkles className="h-4 w-4" />
                  <span>Clean Data Now</span>
                </button>
                <p className="text-xs text-amber-600 mt-2">
                  Cleaning will replace problematic values with NaN and optionally remove them.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Cleaning Applied Info */}
      {dataQuality?.cleaning_applied && dataQuality.cleaning_applied.length > 0 && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 text-green-800 mb-2">
            <CheckCircle className="h-4 w-4" />
            <span className="font-medium">Data Cleaning Applied</span>
          </div>
          <ul className="list-disc list-inside text-sm text-green-700 space-y-1">
            {dataQuality.cleaning_applied.map((op: string, i: number) => (
              <li key={i}>{op}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Legacy Warnings */}
      {preview?.warnings && preview.warnings.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 text-yellow-800 mb-2">
            <AlertTriangle className="h-4 w-4" />
            <span className="font-medium">Warnings</span>
          </div>
          <ul className="list-disc list-inside text-sm text-yellow-700 space-y-1">
            {preview.warnings.map((w: string, i: number) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Time Series Plot */}
      <div className="bg-white rounded-lg border p-4">
        <h2 className="text-lg font-semibold mb-4">Time Series</h2>
        {plotLoading ? (
          <div className="h-80 flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        ) : plot ? (
          <Plot
            data={plot.data}
            layout={{ ...plot.layout, autosize: true }}
            config={{ responsive: true }}
            style={{ width: '100%', height: '400px' }}
          />
        ) : (
          <div className="h-80 flex items-center justify-center text-gray-500">
            Failed to load plot
          </div>
        )}
      </div>

      {/* Statistics */}
      <div className="bg-white rounded-lg border p-4">
        <h2 className="text-lg font-semibold mb-4">Statistics</h2>
        {statistics?.statistics && (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {Object.entries(statistics.statistics).map(([key, value]) => (
              <div key={key} className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-500 uppercase">{key.replace('_', ' ')}</div>
                <div className="text-lg font-semibold">
                  {typeof value === 'number' ? value.toFixed(2) : String(value ?? 'N/A')}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Hydrologic Signatures (for flow data) */}
      {statistics?.signatures && Object.keys(statistics.signatures).length > 0 && (
        <div className="bg-white rounded-lg border p-4">
          <h2 className="text-lg font-semibold mb-4">Hydrologic Signatures</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(statistics.signatures).map(([key, value]) => (
              <div key={key} className="bg-blue-50 rounded-lg p-3">
                <div className="text-xs text-blue-600 uppercase">{key.replace(/_/g, ' ')}</div>
                <div className="text-lg font-semibold text-blue-900">
                  {typeof value === 'number' ? value.toFixed(4) : String(value ?? 'N/A')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Seasonal Statistics (for rainfall/PET) */}
      {statistics?.seasonal && Object.keys(statistics.seasonal).length > 0 && (
        <div className="bg-white rounded-lg border p-4">
          <h2 className="text-lg font-semibold mb-4">Monthly Averages</h2>
          <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-12 gap-2">
            {Object.entries(statistics.seasonal).map(([month, value]) => (
              <div key={month} className="bg-gray-50 rounded-lg p-2 text-center">
                <div className="text-xs text-gray-500">{month}</div>
                <div className="text-sm font-semibold">
                  {typeof value === 'number' ? value.toFixed(1) : String(value ?? '-')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sample Data */}
      {preview?.sample_data && preview.sample_data.length > 0 && (
        <div className="bg-white rounded-lg border p-4">
          <h2 className="text-lg font-semibold mb-4">Sample Data (First 10 rows)</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Value</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {preview.sample_data.map((row: Record<string, unknown>, i: number) => (
                  <tr key={i}>
                    <td className="px-4 py-2 text-sm text-gray-600">{String(row.date ?? row.Date ?? '')}</td>
                    <td className="px-4 py-2 text-sm font-mono">
                      {typeof row.value === 'number' ? row.value.toFixed(4) : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Cleaning Modal */}
      {showCleaningModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-lg w-full mx-4 p-6">
            <h3 className="text-xl font-semibold mb-4">Clean Dataset</h3>
            <p className="text-gray-600 mb-4">
              Configure how to handle problematic values in your dataset.
            </p>

            <div className="space-y-4">
              {/* Replace Sentinel Values */}
              <label className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  checked={cleaningOptions.replace_sentinel}
                  onChange={(e) => setCleaningOptions({
                    ...cleaningOptions,
                    replace_sentinel: e.target.checked
                  })}
                  className="mt-1 h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <div>
                  <span className="font-medium text-gray-900">Replace sentinel values</span>
                  <p className="text-sm text-gray-500">
                    Replace -9999, -999, -99, -1 with NaN (these are common missing data indicators)
                  </p>
                </div>
              </label>

              {/* Replace Negative Values */}
              <label className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  checked={cleaningOptions.replace_negative}
                  onChange={(e) => setCleaningOptions({
                    ...cleaningOptions,
                    replace_negative: e.target.checked
                  })}
                  className="mt-1 h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <div>
                  <span className="font-medium text-gray-900">Replace negative values</span>
                  <p className="text-sm text-gray-500">
                    Replace negative values with NaN (physically impossible for rainfall, PET, and flow)
                  </p>
                </div>
              </label>

              {/* Interpolate */}
              <label className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  checked={cleaningOptions.interpolate}
                  onChange={(e) => setCleaningOptions({
                    ...cleaningOptions,
                    interpolate: e.target.checked
                  })}
                  className="mt-1 h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <div>
                  <span className="font-medium text-gray-900">Interpolate short gaps</span>
                  <p className="text-sm text-gray-500">
                    Fill missing values using linear interpolation (only for gaps ≤ {cleaningOptions.max_interpolate_gap} days)
                  </p>
                </div>
              </label>

              {/* Drop NA */}
              <label className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  checked={cleaningOptions.drop_na}
                  onChange={(e) => setCleaningOptions({
                    ...cleaningOptions,
                    drop_na: e.target.checked
                  })}
                  className="mt-1 h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <div>
                  <span className="font-medium text-gray-900">Drop rows with missing values</span>
                  <p className="text-sm text-gray-500">
                    Remove all rows containing NaN values after cleaning
                  </p>
                </div>
              </label>
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowCleaningModal(false)}
                className="px-4 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleClean}
                disabled={cleanMutation.isPending}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                {cleanMutation.isPending ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    <span>Cleaning...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    <span>Clean Data</span>
                  </>
                )}
              </button>
            </div>

            {cleanMutation.isError && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                Failed to clean dataset. Please try again.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
