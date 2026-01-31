import { useState, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  Upload, 
  Database, 
  FlaskConical, 
  Plus, 
  Trash2,
  FileSpreadsheet,
  CheckCircle2,
  AlertCircle
} from 'lucide-react'
import { 
  getCatchment, 
  uploadDataset, 
  deleteDataset,
  getExperiments
} from '../services/api'
import type { DatasetType, ExperimentStatus } from '../types'

const DATASET_TYPES: { value: DatasetType; label: string; description: string }[] = [
  { value: 'rainfall', label: 'Rainfall', description: 'Daily precipitation (mm/day)' },
  { value: 'pet', label: 'PET', description: 'Potential evapotranspiration (mm/day)' },
  { value: 'observed_flow', label: 'Observed Flow', description: 'Observed streamflow (ML/day)' },
]

function UploadModal({ 
  isOpen, 
  onClose,
  catchmentId
}: { 
  isOpen: boolean
  onClose: () => void
  catchmentId: string
}) {
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [name, setName] = useState('')
  const [type, setType] = useState<DatasetType>('rainfall')
  const [file, setFile] = useState<File | null>(null)
  
  const mutation = useMutation({
    mutationFn: () => uploadDataset(catchmentId, name, type, file!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['catchment', catchmentId] })
      onClose()
      setName('')
      setFile(null)
    },
  })
  
  if (!isOpen) return null
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Upload Dataset</h2>
        </div>
        
        <form onSubmit={(e) => {
          e.preventDefault()
          if (file) mutation.mutate()
        }}>
          <div className="px-6 py-4 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Dataset Name *
              </label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="e.g., SILO Rainfall 1985-2024"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Data Type *
              </label>
              <select
                value={type}
                onChange={(e) => setType(e.target.value as DatasetType)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                {DATASET_TYPES.map((dt) => (
                  <option key={dt.value} value={dt.value}>
                    {dt.label} - {dt.description}
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                CSV File *
              </label>
              <input
                type="file"
                ref={fileInputRef}
                accept=".csv"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="w-full px-4 py-8 border-2 border-dashed border-gray-300 rounded-lg hover:border-primary-400 transition-colors"
              >
                {file ? (
                  <div className="flex items-center justify-center text-gray-700">
                    <FileSpreadsheet className="w-6 h-6 mr-2" />
                    {file.name}
                  </div>
                ) : (
                  <div className="text-gray-500">
                    <Upload className="w-8 h-8 mx-auto mb-2" />
                    <p>Click to select a CSV file</p>
                  </div>
                )}
              </button>
            </div>
            
            {mutation.isError && (
              <div className="flex items-center text-red-600 text-sm">
                <AlertCircle className="w-4 h-4 mr-2" />
                {(mutation.error as Error)?.message || 'Upload failed'}
              </div>
            )}
          </div>
          
          <div className="px-6 py-4 border-t border-gray-200 flex justify-end gap-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 rounded-lg"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!file || mutation.isPending}
              className="px-4 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-lg disabled:opacity-50"
            >
              {mutation.isPending ? 'Uploading...' : 'Upload Dataset'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function StatusBadge({ status }: { status: ExperimentStatus }) {
  const config = {
    draft: { color: 'bg-gray-100 text-gray-700' },
    queued: { color: 'bg-yellow-100 text-yellow-700' },
    running: { color: 'bg-blue-100 text-blue-700' },
    completed: { color: 'bg-green-100 text-green-700' },
    failed: { color: 'bg-red-100 text-red-700' },
    cancelled: { color: 'bg-gray-100 text-gray-700' },
  }[status] || { color: 'bg-gray-100 text-gray-700' }
  
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${config.color}`}>
      {status}
    </span>
  )
}

export default function CatchmentDetail() {
  const { id } = useParams<{ id: string }>()
  const [showUploadModal, setShowUploadModal] = useState(false)
  const queryClient = useQueryClient()
  
  const { data: catchment, isLoading } = useQuery({
    queryKey: ['catchment', id],
    queryFn: () => getCatchment(id!),
    enabled: !!id,
  })
  
  const { data: experiments } = useQuery({
    queryKey: ['experiments', id],
    queryFn: () => getExperiments(id),
    enabled: !!id,
  })
  
  const deleteMutation = useMutation({
    mutationFn: deleteDataset,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['catchment', id] })
    },
  })
  
  if (isLoading) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin w-8 h-8 border-4 border-primary-500 border-t-transparent rounded-full mx-auto"></div>
        <p className="mt-4 text-gray-500">Loading catchment...</p>
      </div>
    )
  }
  
  if (!catchment) {
    return (
      <div className="text-center py-12">
        <AlertCircle className="w-12 h-12 mx-auto text-red-500" />
        <p className="mt-4 text-gray-900">Catchment not found</p>
      </div>
    )
  }
  
  return (
    <div>
      {/* Header */}
      <div className="flex justify-between items-start mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{catchment.name}</h1>
          <div className="flex items-center gap-4 mt-2 text-gray-500">
            {catchment.gauge_id && <span>Gauge: {catchment.gauge_id}</span>}
            {catchment.area_km2 && <span>Area: {catchment.area_km2.toFixed(2)} km²</span>}
          </div>
        </div>
        <Link
          to={`/experiments/new?catchment=${id}`}
          className="inline-flex items-center px-4 py-2 border border-transparent rounded-lg text-sm font-medium text-white bg-primary-600 hover:bg-primary-700"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Experiment
        </Link>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Datasets */}
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
            <h2 className="text-lg font-semibold text-gray-900">
              <Database className="w-5 h-5 inline mr-2" />
              Datasets
            </h2>
            <button
              onClick={() => setShowUploadModal(true)}
              className="text-sm text-primary-600 hover:text-primary-700 font-medium"
            >
              <Upload className="w-4 h-4 inline mr-1" />
              Upload
            </button>
          </div>
          
          {catchment.datasets.length === 0 ? (
            <div className="p-6 text-center text-gray-500">
              <Database className="w-8 h-8 mx-auto mb-2 text-gray-300" />
              <p>No datasets uploaded</p>
              <button
                onClick={() => setShowUploadModal(true)}
                className="mt-2 text-sm text-primary-600 hover:text-primary-700"
              >
                Upload your first dataset
              </button>
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {catchment.datasets.map((dataset) => (
                <Link
                  key={dataset.id}
                  to={`/catchments/${id}/datasets/${dataset.id}`}
                  className="block px-6 py-4 hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-900">{dataset.name}</p>
                      <p className="text-sm text-gray-500">
                        {dataset.type} • {dataset.record_count?.toLocaleString()} records
                      </p>
                      <p className="text-xs text-gray-400 mt-1">
                        {dataset.start_date} to {dataset.end_date}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-5 h-5 text-green-500" />
                      <button
                        onClick={(e) => {
                          e.preventDefault()
                          e.stopPropagation()
                          if (confirm('Delete this dataset?')) {
                            deleteMutation.mutate(dataset.id)
                          }
                        }}
                        className="p-1 text-gray-400 hover:text-red-600"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
        
        {/* Experiments */}
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">
              <FlaskConical className="w-5 h-5 inline mr-2" />
              Experiments
            </h2>
          </div>
          
          {!experiments || experiments.length === 0 ? (
            <div className="p-6 text-center text-gray-500">
              <FlaskConical className="w-8 h-8 mx-auto mb-2 text-gray-300" />
              <p>No experiments yet</p>
              <Link
                to={`/experiments/new?catchment=${id}`}
                className="mt-2 text-sm text-primary-600 hover:text-primary-700"
              >
                Create your first experiment
              </Link>
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {experiments.map((exp) => (
                <Link
                  key={exp.id}
                  to={`/experiments/${exp.id}${exp.status === 'completed' ? '/results' : ''}`}
                  className="block px-6 py-4 hover:bg-gray-50"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-900">{exp.name}</p>
                      <p className="text-sm text-gray-500">
                        {exp.model_type.toUpperCase()} • 
                        {new Date(exp.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <StatusBadge status={exp.status as ExperimentStatus} />
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
      
      <UploadModal
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        catchmentId={id!}
      />
    </div>
  )
}
