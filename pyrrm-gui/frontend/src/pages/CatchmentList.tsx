import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { Plus, Map, Database, FlaskConical, Trash2 } from 'lucide-react'
import { getCatchments, createCatchment, deleteCatchment } from '../services/api'
import type { CatchmentCreate } from '../types'

function CreateCatchmentModal({ 
  isOpen, 
  onClose 
}: { 
  isOpen: boolean
  onClose: () => void 
}) {
  const queryClient = useQueryClient()
  const [formData, setFormData] = useState<CatchmentCreate>({
    name: '',
    gauge_id: '',
    area_km2: undefined,
    description: '',
  })
  
  const mutation = useMutation({
    mutationFn: createCatchment,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['catchments'] })
      onClose()
      setFormData({ name: '', gauge_id: '', area_km2: undefined, description: '' })
    },
  })
  
  if (!isOpen) return null
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Create Catchment</h2>
        </div>
        
        <form onSubmit={(e) => {
          e.preventDefault()
          mutation.mutate(formData)
        }}>
          <div className="px-6 py-4 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Name *
              </label>
              <input
                type="text"
                required
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="e.g., Queanbeyan River at Queanbeyan"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Gauge ID
              </label>
              <input
                type="text"
                value={formData.gauge_id || ''}
                onChange={(e) => setFormData({ ...formData, gauge_id: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="e.g., 410734"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Area (km²)
              </label>
              <input
                type="number"
                step="0.01"
                value={formData.area_km2 || ''}
                onChange={(e) => setFormData({ ...formData, area_km2: e.target.value ? parseFloat(e.target.value) : undefined })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="e.g., 516.63"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                rows={3}
                value={formData.description || ''}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="Optional notes about this catchment..."
              />
            </div>
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
              disabled={mutation.isPending}
              className="px-4 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-lg disabled:opacity-50"
            >
              {mutation.isPending ? 'Creating...' : 'Create Catchment'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function CatchmentList() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const queryClient = useQueryClient()
  
  const { data: catchments, isLoading } = useQuery({
    queryKey: ['catchments'],
    queryFn: getCatchments,
  })
  
  const deleteMutation = useMutation({
    mutationFn: deleteCatchment,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['catchments'] })
    },
  })
  
  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Catchments</h1>
          <p className="text-gray-500 mt-1">
            Manage your hydrological catchments and their data
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="inline-flex items-center px-4 py-2 border border-transparent rounded-lg text-sm font-medium text-white bg-primary-600 hover:bg-primary-700"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Catchment
        </button>
      </div>
      
      {isLoading ? (
        <div className="text-center py-12">
          <div className="animate-spin w-8 h-8 border-4 border-primary-500 border-t-transparent rounded-full mx-auto"></div>
          <p className="mt-4 text-gray-500">Loading catchments...</p>
        </div>
      ) : catchments?.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
          <Map className="w-12 h-12 mx-auto text-gray-300" />
          <h3 className="mt-4 text-lg font-medium text-gray-900">No catchments yet</h3>
          <p className="mt-2 text-gray-500">Create a catchment to get started with calibration</p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="mt-4 inline-flex items-center px-4 py-2 border border-transparent rounded-lg text-sm font-medium text-white bg-primary-600 hover:bg-primary-700"
          >
            <Plus className="w-4 h-4 mr-2" />
            Create Your First Catchment
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {catchments?.map((catchment) => (
            <div
              key={catchment.id}
              className="bg-white rounded-lg border border-gray-200 hover:shadow-lg transition-shadow"
            >
              <Link to={`/catchments/${catchment.id}`} className="block p-6">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{catchment.name}</h3>
                    {catchment.gauge_id && (
                      <p className="text-sm text-gray-500">Gauge: {catchment.gauge_id}</p>
                    )}
                  </div>
                  <Map className="w-8 h-8 text-hydro" />
                </div>
                
                {catchment.area_km2 && (
                  <p className="mt-2 text-sm text-gray-600">
                    Area: {catchment.area_km2.toFixed(2)} km²
                  </p>
                )}
                
                <div className="mt-4 flex items-center gap-4 text-sm text-gray-500">
                  <span className="inline-flex items-center">
                    <Database className="w-4 h-4 mr-1" />
                    {catchment.dataset_count} datasets
                  </span>
                  <span className="inline-flex items-center">
                    <FlaskConical className="w-4 h-4 mr-1" />
                    {catchment.experiment_count} experiments
                  </span>
                </div>
              </Link>
              
              <div className="px-6 py-3 border-t border-gray-100 flex justify-end">
                <button
                  onClick={() => {
                    if (confirm('Are you sure you want to delete this catchment?')) {
                      deleteMutation.mutate(catchment.id)
                    }
                  }}
                  className="text-sm text-red-600 hover:text-red-700"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
      
      <CreateCatchmentModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
      />
    </div>
  )
}
