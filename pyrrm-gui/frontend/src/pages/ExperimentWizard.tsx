import { useState, useEffect, useMemo } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { 
  ChevronRight, 
  ChevronLeft, 
  Play,
  CheckCircle2,
  AlertCircle,
  Scissors,
  Info
} from 'lucide-react'
import { 
  getCatchments,
  getCatchment,
  getModels,
  getObjectives,
  getAlgorithms,
  getModelBounds,
  createExperiment,
  runExperiment,
  getDatasetStatistics,
  getCatchmentTimeseries
} from '../services/api'
import type { 
  ModelType, 
  ExperimentCreate,
  FlowTrimmingConfig
} from '../types'
import CalibrationPeriodSelector from '../components/CalibrationPeriodSelector'

const STEPS = [
  { id: 'basic', title: 'Basic Info' },
  { id: 'model', title: 'Model' },
  { id: 'objective', title: 'Objective' },
  { id: 'algorithm', title: 'Algorithm' },
  { id: 'review', title: 'Review' },
]

export default function ExperimentWizard() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const initialCatchmentId = searchParams.get('catchment') || ''
  
  const [step, setStep] = useState(0)
  const [formData, setFormData] = useState<Partial<ExperimentCreate>>({
    catchment_id: initialCatchmentId,
    name: '',
    model_type: 'sacramento' as ModelType,
    calibration_period: {
      start_date: '1990-01-01',
      end_date: '2020-12-31',
      warmup_days: 365,
    },
    objective_config: {
      type: 'NSE',
      transform: 'none',
      flow_trimming: {
        enabled: false,
        min_threshold: undefined,
        max_threshold: undefined,
      }
    },
    algorithm_config: {
      method: 'sceua_direct',
      max_evals: 10000,  // Match notebooks default
      max_workers: 1,
      checkpoint_interval: 5000,
      max_tolerant_iter: 100,  // Match notebooks - allow 100 iters without improvement
      tolerance: 1e-4,          // Match notebooks - more lenient threshold
    },
  })
  
  // Load reference data
  const { data: catchments } = useQuery({
    queryKey: ['catchments'],
    queryFn: getCatchments,
  })
  
  const { data: selectedCatchment } = useQuery({
    queryKey: ['catchment', formData.catchment_id],
    queryFn: () => getCatchment(formData.catchment_id!),
    enabled: !!formData.catchment_id,
  })
  
  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
  })
  
  const { data: objectives } = useQuery({
    queryKey: ['objectives'],
    queryFn: getObjectives,
  })
  
  const { data: algorithms } = useQuery({
    queryKey: ['algorithms'],
    queryFn: getAlgorithms,
  })
  
  const { data: parameterBounds } = useQuery({
    queryKey: ['bounds', formData.model_type],
    queryFn: () => getModelBounds(formData.model_type!),
    enabled: !!formData.model_type,
  })
  
  // Get observed flow dataset for flow statistics
  const observedFlowDataset = useMemo(() => {
    return selectedCatchment?.datasets?.find(d => d.type === 'observed_flow')
  }, [selectedCatchment])
  
  const { data: flowStats } = useQuery({
    queryKey: ['dataset-statistics', observedFlowDataset?.id],
    queryFn: () => getDatasetStatistics(observedFlowDataset!.id),
    enabled: !!observedFlowDataset?.id,
  })
  
  // Get catchment timeseries for calibration period selection
  const { data: catchmentTimeseries, isLoading: isLoadingTimeseries } = useQuery({
    queryKey: ['catchment-timeseries', formData.catchment_id],
    queryFn: () => getCatchmentTimeseries(formData.catchment_id!, 1000),
    enabled: !!formData.catchment_id,
  })
  
  // Set initial dates from calibration range (observed flow range) when timeseries loads
  useEffect(() => {
    if (catchmentTimeseries?.calibration_range && formData.calibration_period?.start_date === '1990-01-01') {
      // Use full calibration range (observed flow range) as default
      setFormData(prev => ({
        ...prev,
        calibration_period: {
          ...prev.calibration_period!,
          start_date: catchmentTimeseries.calibration_range.start,
          end_date: catchmentTimeseries.calibration_range.end,
        }
      }))
    }
  }, [catchmentTimeseries])
  
  // Set default bounds when model changes
  useEffect(() => {
    if (parameterBounds && !formData.parameter_bounds) {
      setFormData(prev => ({ ...prev, parameter_bounds: parameterBounds }))
    }
  }, [parameterBounds])
  
  const createMutation = useMutation({
    mutationFn: createExperiment,
  })
  
  const runMutation = useMutation({
    mutationFn: runExperiment,
  })
  
  const handleNext = () => {
    if (step < STEPS.length - 1) {
      setStep(step + 1)
    }
  }
  
  const handleBack = () => {
    if (step > 0) {
      setStep(step - 1)
    }
  }
  
  const handleSubmit = async (startImmediately: boolean) => {
    try {
      const experiment = await createMutation.mutateAsync(formData as ExperimentCreate)
      
      if (startImmediately) {
        await runMutation.mutateAsync(experiment.id)
        navigate(`/experiments/${experiment.id}`)
      } else {
        navigate(`/experiments/${experiment.id}`)
      }
    } catch (error) {
      console.error('Failed to create experiment:', error)
    }
  }
  
  const renderStep = () => {
    switch (step) {
      case 0: // Basic Info
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Catchment *
                </label>
                <select
                  value={formData.catchment_id}
                  onChange={(e) => setFormData({ ...formData, catchment_id: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                >
                  <option value="">Select a catchment...</option>
                  {catchments?.map((c) => (
                    <option key={c.id} value={c.id}>{c.name}</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Experiment Name *
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  placeholder="e.g., Sacramento NSE Calibration"
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                rows={2}
                value={formData.description || ''}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                placeholder="Optional notes about this experiment..."
              />
            </div>
            
            {/* Calibration Period Selector */}
            <div className="border-t pt-4">
              <h3 className="text-sm font-medium text-gray-700 mb-3">
                Calibration Period
              </h3>
              <CalibrationPeriodSelector
                timeseries={catchmentTimeseries || null}
                isLoading={isLoadingTimeseries}
                startDate={formData.calibration_period?.start_date || ''}
                endDate={formData.calibration_period?.end_date || ''}
                warmupDays={formData.calibration_period?.warmup_days || 365}
                onPeriodChange={(start, end) => setFormData({
                  ...formData,
                  calibration_period: {
                    ...formData.calibration_period!,
                    start_date: start,
                    end_date: end,
                  }
                })}
                onWarmupChange={(days) => setFormData({
                  ...formData,
                  calibration_period: {
                    ...formData.calibration_period!,
                    warmup_days: days,
                  }
                })}
              />
            </div>
          </div>
        )
      
      case 1: // Model
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model Type
              </label>
              <div className="grid grid-cols-2 gap-4">
                {models?.map((model) => (
                  <button
                    key={model.type}
                    type="button"
                    onClick={() => setFormData({ ...formData, model_type: model.type as ModelType })}
                    className={`p-4 border rounded-lg text-left transition-colors ${
                      formData.model_type === model.type
                        ? 'border-primary-500 bg-primary-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <p className="font-medium text-gray-900">{model.name}</p>
                    <p className="text-sm text-gray-500">{model.n_parameters} parameters</p>
                  </button>
                ))}
              </div>
            </div>
            
            {parameterBounds && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Parameter Bounds
                </label>
                <p className="text-sm text-gray-500 mb-4">
                  Adjust the search bounds for each parameter
                </p>
                <div className="max-h-80 overflow-y-auto border border-gray-200 rounded-lg">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50 sticky top-0">
                      <tr>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Parameter</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Min</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Max</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {Object.entries(parameterBounds).map(([name, [min, max]]) => (
                        <tr key={name}>
                          <td className="px-4 py-2 text-sm font-medium text-gray-900">{name}</td>
                          <td className="px-4 py-2">
                            <input
                              type="number"
                              step="any"
                              value={formData.parameter_bounds?.[name]?.[0] ?? min}
                              onChange={(e) => setFormData({
                                ...formData,
                                parameter_bounds: {
                                  ...formData.parameter_bounds,
                                  [name]: [parseFloat(e.target.value), formData.parameter_bounds?.[name]?.[1] ?? max]
                                }
                              })}
                              className="w-24 px-2 py-1 border border-gray-300 rounded text-sm"
                            />
                          </td>
                          <td className="px-4 py-2">
                            <input
                              type="number"
                              step="any"
                              value={formData.parameter_bounds?.[name]?.[1] ?? max}
                              onChange={(e) => setFormData({
                                ...formData,
                                parameter_bounds: {
                                  ...formData.parameter_bounds,
                                  [name]: [formData.parameter_bounds?.[name]?.[0] ?? min, parseFloat(e.target.value)]
                                }
                              })}
                              className="w-24 px-2 py-1 border border-gray-300 rounded text-sm"
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )
      
      case 2: // Objective
        const trimConfig = formData.objective_config?.flow_trimming
        const stats = flowStats?.statistics || {}
        
        // Helper to calculate percentile value from stats
        const getPercentileValue = (percentile: number): number => {
          const key = `p${percentile}`
          if (stats[key] !== undefined) return stats[key]
          // Approximate from available percentiles
          if (percentile <= 5) return stats.p5 || stats.min || 0
          if (percentile <= 10) return stats.p10 || stats.p5 || 0
          if (percentile <= 25) return stats.p25 || stats.p10 || 0
          if (percentile <= 50) return stats.p50 || stats.median || 0
          if (percentile <= 75) return stats.p75 || stats.p50 || 0
          if (percentile <= 90) return stats.p90 || stats.p75 || 0
          if (percentile <= 95) return stats.p95 || stats.p90 || 0
          return stats.max || 1000
        }
        
        // Calculate actual threshold values for preview
        const getEffectiveMin = () => {
          if (!trimConfig?.min_threshold) return null
          if (trimConfig.min_threshold.type === 'absolute') return trimConfig.min_threshold.value
          return getPercentileValue(trimConfig.min_threshold.value)
        }
        
        const getEffectiveMax = () => {
          if (!trimConfig?.max_threshold) return null
          if (trimConfig.max_threshold.type === 'absolute') return trimConfig.max_threshold.value
          return getPercentileValue(trimConfig.max_threshold.value)
        }
        
        const effectiveMin = getEffectiveMin()
        const effectiveMax = getEffectiveMax()
        
        return (
          <div className="space-y-6">
            {/* Objective Function */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Objective Function
              </label>
              <div className="grid grid-cols-2 gap-4">
                {objectives?.map((obj) => (
                  <button
                    key={obj.name}
                    type="button"
                    onClick={() => setFormData({ 
                      ...formData, 
                      objective_config: { ...formData.objective_config!, type: obj.name }
                    })}
                    className={`p-4 border rounded-lg text-left transition-colors ${
                      formData.objective_config?.type === obj.name
                        ? 'border-primary-500 bg-primary-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <p className="font-medium text-gray-900">{obj.name}</p>
                    <p className="text-sm text-gray-500">
                      {obj.maximize ? 'Maximize' : 'Minimize'} (optimal: {obj.optimal_value})
                    </p>
                  </button>
                ))}
              </div>
            </div>
            
            {/* Flow Transformation */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Flow Transformation
              </label>
              <select
                value={formData.objective_config?.transform}
                onChange={(e) => setFormData({
                  ...formData,
                  objective_config: { ...formData.objective_config!, transform: e.target.value }
                })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              >
                <option value="none">None (emphasizes high flows)</option>
                <option value="log">Logarithm (emphasizes low flows)</option>
                <option value="sqrt">Square Root (balanced)</option>
                <option value="inverse">Inverse (strong low flow emphasis)</option>
              </select>
            </div>
            
            {/* Flow Range Trimming */}
            <div className="border-t pt-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <Scissors className="w-5 h-5 text-gray-600" />
                  <label className="text-sm font-medium text-gray-700">
                    Flow Range Trimming
                  </label>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={trimConfig?.enabled || false}
                    onChange={(e) => setFormData({
                      ...formData,
                      objective_config: {
                        ...formData.objective_config!,
                        flow_trimming: {
                          ...trimConfig,
                          enabled: e.target.checked,
                        } as FlowTrimmingConfig
                      }
                    })}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
                </label>
              </div>
              
              {trimConfig?.enabled && (
                <div className="space-y-6 bg-gray-50 rounded-lg p-4">
                  {/* Info Box */}
                  <div className="flex items-start space-x-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                    <div className="text-sm text-blue-800">
                      <p className="font-medium">Trim calibration to specific flow ranges</p>
                      <p className="mt-1 text-blue-700">
                        Exclude very low flows (measurement error) or extreme events that may bias calibration.
                        Only observations within the specified range will be used for objective function calculation.
                      </p>
                    </div>
                  </div>
                  
                  {/* Flow Statistics Preview */}
                  {flowStats && (
                    <div className="grid grid-cols-5 gap-2 text-center text-xs">
                      <div className="bg-white rounded p-2 border">
                        <p className="text-gray-500">Min</p>
                        <p className="font-medium">{stats.min?.toFixed(1)}</p>
                      </div>
                      <div className="bg-white rounded p-2 border">
                        <p className="text-gray-500">P10</p>
                        <p className="font-medium">{(stats.p10 || stats.min)?.toFixed(1)}</p>
                      </div>
                      <div className="bg-white rounded p-2 border">
                        <p className="text-gray-500">Median</p>
                        <p className="font-medium">{(stats.median || stats.p50)?.toFixed(1)}</p>
                      </div>
                      <div className="bg-white rounded p-2 border">
                        <p className="text-gray-500">P90</p>
                        <p className="font-medium">{(stats.p90 || stats.max)?.toFixed(1)}</p>
                      </div>
                      <div className="bg-white rounded p-2 border">
                        <p className="text-gray-500">Max</p>
                        <p className="font-medium">{stats.max?.toFixed(1)}</p>
                      </div>
                    </div>
                  )}
                  
                  {/* Minimum Threshold */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium text-gray-700">
                        Minimum Flow Threshold
                      </label>
                      <label className="flex items-center space-x-2 text-sm">
                        <input
                          type="checkbox"
                          checked={!!trimConfig?.min_threshold}
                          onChange={(e) => setFormData({
                            ...formData,
                            objective_config: {
                              ...formData.objective_config!,
                              flow_trimming: {
                                ...trimConfig!,
                                min_threshold: e.target.checked 
                                  ? { type: 'absolute', value: 1 }
                                  : undefined
                              }
                            }
                          })}
                          className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                        />
                        <span className="text-gray-600">Enable</span>
                      </label>
                    </div>
                    
                    {trimConfig?.min_threshold && (
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-gray-500 mb-1">Type</label>
                          <select
                            value={trimConfig.min_threshold.type}
                            onChange={(e) => setFormData({
                              ...formData,
                              objective_config: {
                                ...formData.objective_config!,
                                flow_trimming: {
                                  ...trimConfig!,
                                  min_threshold: {
                                    ...trimConfig.min_threshold!,
                                    type: e.target.value as 'absolute' | 'percentile',
                                    value: e.target.value === 'percentile' ? 5 : 1
                                  }
                                }
                              }
                            })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          >
                            <option value="absolute">Absolute (ML/day)</option>
                            <option value="percentile">Percentile (%)</option>
                          </select>
                        </div>
                        <div>
                          <label className="block text-xs text-gray-500 mb-1">
                            {trimConfig.min_threshold.type === 'absolute' ? 'Value (ML/day)' : 'Percentile (%)'}
                          </label>
                          <input
                            type="number"
                            step={trimConfig.min_threshold.type === 'percentile' ? '1' : '0.1'}
                            min={trimConfig.min_threshold.type === 'percentile' ? 0 : 0}
                            max={trimConfig.min_threshold.type === 'percentile' ? 100 : undefined}
                            value={trimConfig.min_threshold.value}
                            onChange={(e) => setFormData({
                              ...formData,
                              objective_config: {
                                ...formData.objective_config!,
                                flow_trimming: {
                                  ...trimConfig!,
                                  min_threshold: {
                                    ...trimConfig.min_threshold!,
                                    value: parseFloat(e.target.value) || 0
                                  }
                                }
                              }
                            })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          />
                        </div>
                        {trimConfig.min_threshold.type === 'percentile' && (
                          <div className="col-span-2">
                            <input
                              type="range"
                              min="0"
                              max="50"
                              step="1"
                              value={trimConfig.min_threshold.value}
                              onChange={(e) => setFormData({
                                ...formData,
                                objective_config: {
                                  ...formData.objective_config!,
                                  flow_trimming: {
                                    ...trimConfig!,
                                    min_threshold: {
                                      ...trimConfig.min_threshold!,
                                      value: parseInt(e.target.value)
                                    }
                                  }
                                }
                              })}
                              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                            />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  
                  {/* Maximum Threshold */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium text-gray-700">
                        Maximum Flow Threshold
                      </label>
                      <label className="flex items-center space-x-2 text-sm">
                        <input
                          type="checkbox"
                          checked={!!trimConfig?.max_threshold}
                          onChange={(e) => setFormData({
                            ...formData,
                            objective_config: {
                              ...formData.objective_config!,
                              flow_trimming: {
                                ...trimConfig!,
                                max_threshold: e.target.checked 
                                  ? { type: 'absolute', value: 1000 }
                                  : undefined
                              }
                            }
                          })}
                          className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                        />
                        <span className="text-gray-600">Enable</span>
                      </label>
                    </div>
                    
                    {trimConfig?.max_threshold && (
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-gray-500 mb-1">Type</label>
                          <select
                            value={trimConfig.max_threshold.type}
                            onChange={(e) => setFormData({
                              ...formData,
                              objective_config: {
                                ...formData.objective_config!,
                                flow_trimming: {
                                  ...trimConfig!,
                                  max_threshold: {
                                    ...trimConfig.max_threshold!,
                                    type: e.target.value as 'absolute' | 'percentile',
                                    value: e.target.value === 'percentile' ? 95 : 1000
                                  }
                                }
                              }
                            })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          >
                            <option value="absolute">Absolute (ML/day)</option>
                            <option value="percentile">Percentile (%)</option>
                          </select>
                        </div>
                        <div>
                          <label className="block text-xs text-gray-500 mb-1">
                            {trimConfig.max_threshold.type === 'absolute' ? 'Value (ML/day)' : 'Percentile (%)'}
                          </label>
                          <input
                            type="number"
                            step={trimConfig.max_threshold.type === 'percentile' ? '1' : '0.1'}
                            min={trimConfig.max_threshold.type === 'percentile' ? 0 : 0}
                            max={trimConfig.max_threshold.type === 'percentile' ? 100 : undefined}
                            value={trimConfig.max_threshold.value}
                            onChange={(e) => setFormData({
                              ...formData,
                              objective_config: {
                                ...formData.objective_config!,
                                flow_trimming: {
                                  ...trimConfig!,
                                  max_threshold: {
                                    ...trimConfig.max_threshold!,
                                    value: parseFloat(e.target.value) || 0
                                  }
                                }
                              }
                            })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                          />
                        </div>
                        {trimConfig.max_threshold.type === 'percentile' && (
                          <div className="col-span-2">
                            <input
                              type="range"
                              min="50"
                              max="100"
                              step="1"
                              value={trimConfig.max_threshold.value}
                              onChange={(e) => setFormData({
                                ...formData,
                                objective_config: {
                                  ...formData.objective_config!,
                                  flow_trimming: {
                                    ...trimConfig!,
                                    max_threshold: {
                                      ...trimConfig.max_threshold!,
                                      value: parseInt(e.target.value)
                                    }
                                  }
                                }
                              })}
                              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                            />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  
                  {/* Preview of effective range */}
                  {(effectiveMin !== null || effectiveMax !== null) && (
                    <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                      <p className="text-sm font-medium text-amber-800">Effective Flow Range</p>
                      <p className="text-sm text-amber-700 mt-1">
                        Only flows between{' '}
                        <span className="font-mono font-medium">
                          {effectiveMin !== null ? effectiveMin.toFixed(1) : '0'}
                        </span>
                        {' '}and{' '}
                        <span className="font-mono font-medium">
                          {effectiveMax !== null ? effectiveMax.toFixed(1) : '∞'}
                        </span>
                        {' '}ML/day will be used for calibration.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )
      
      case 3: // Algorithm
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Algorithm
              </label>
              <select
                value={formData.algorithm_config?.method}
                onChange={(e) => setFormData({
                  ...formData,
                  algorithm_config: { ...formData.algorithm_config!, method: e.target.value }
                })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              >
                {algorithms?.map((alg) => (
                  <option key={alg.id} value={alg.id}>{alg.name}</option>
                ))}
              </select>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Max Evaluations
                </label>
                <input
                  type="number"
                  value={formData.algorithm_config?.max_evals}
                  onChange={(e) => setFormData({
                    ...formData,
                    algorithm_config: { ...formData.algorithm_config!, max_evals: parseInt(e.target.value) }
                  })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Recommended: 10,000 for quick tests, 50,000+ for production
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Parallel Workers
                </label>
                <input
                  type="number"
                  min={1}
                  max={8}
                  value={formData.algorithm_config?.max_workers}
                  onChange={(e) => setFormData({
                    ...formData,
                    algorithm_config: { ...formData.algorithm_config!, max_workers: parseInt(e.target.value) }
                  })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                />
              </div>
            </div>
            
            {/* Convergence Settings */}
            <div className="border-t pt-4">
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900 flex items-center">
                  <span>Convergence Settings (Advanced)</span>
                  <span className="ml-2 text-xs text-gray-500">Click to expand</span>
                </summary>
                <div className="mt-4 space-y-4 bg-gray-50 rounded-lg p-4">
                  <div className="text-sm text-gray-600 mb-4">
                    These settings control when the algorithm stops. Default values match the tutorial notebooks.
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Max Iterations Without Improvement
                      </label>
                      <input
                        type="number"
                        min={10}
                        max={500}
                        value={formData.algorithm_config?.max_tolerant_iter || 100}
                        onChange={(e) => setFormData({
                          ...formData,
                          algorithm_config: { 
                            ...formData.algorithm_config!, 
                            max_tolerant_iter: parseInt(e.target.value) 
                          }
                        })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Stop after this many iterations without finding a better solution (default: 100)
                      </p>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Improvement Threshold
                      </label>
                      <select
                        value={formData.algorithm_config?.tolerance || 1e-4}
                        onChange={(e) => setFormData({
                          ...formData,
                          algorithm_config: { 
                            ...formData.algorithm_config!, 
                            tolerance: parseFloat(e.target.value) 
                          }
                        })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                      >
                        <option value={1e-2}>1e-2 (Very lenient)</option>
                        <option value={1e-3}>1e-3 (Lenient)</option>
                        <option value={1e-4}>1e-4 (Default - Balanced)</option>
                        <option value={1e-5}>1e-5 (Strict)</option>
                        <option value={1e-6}>1e-6 (Very strict)</option>
                      </select>
                      <p className="text-xs text-gray-500 mt-1">
                        Minimum improvement to count as progress (smaller = stricter)
                      </p>
                    </div>
                  </div>
                </div>
              </details>
            </div>
          </div>
        )
      
      case 4: // Review
        return (
          <div className="space-y-6">
            <div className="bg-gray-50 rounded-lg p-6">
              <h3 className="font-medium text-gray-900 mb-4">Experiment Summary</h3>
              
              <dl className="space-y-3">
                <div className="flex justify-between">
                  <dt className="text-gray-500">Name</dt>
                  <dd className="font-medium text-gray-900">{formData.name}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Catchment</dt>
                  <dd className="font-medium text-gray-900">{selectedCatchment?.name}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Model</dt>
                  <dd className="font-medium text-gray-900">{formData.model_type?.toUpperCase()}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Objective</dt>
                  <dd className="font-medium text-gray-900">
                    {formData.objective_config?.type}
                    {formData.objective_config?.transform !== 'none' && 
                      ` (${formData.objective_config?.transform})`}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Algorithm</dt>
                  <dd className="font-medium text-gray-900">
                    {algorithms?.find(a => a.id === formData.algorithm_config?.method)?.name}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Max Evaluations</dt>
                  <dd className="font-medium text-gray-900">
                    {formData.algorithm_config?.max_evals?.toLocaleString()}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Period</dt>
                  <dd className="font-medium text-gray-900">
                    {formData.calibration_period?.start_date} to {formData.calibration_period?.end_date}
                  </dd>
                </div>
                {formData.objective_config?.flow_trimming?.enabled && (
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Flow Trimming</dt>
                    <dd className="font-medium text-gray-900">
                      {formData.objective_config.flow_trimming.min_threshold && (
                        <>
                          Min: {formData.objective_config.flow_trimming.min_threshold.value}
                          {formData.objective_config.flow_trimming.min_threshold.type === 'percentile' ? '%ile' : ' ML/day'}
                        </>
                      )}
                      {formData.objective_config.flow_trimming.min_threshold && 
                       formData.objective_config.flow_trimming.max_threshold && ' | '}
                      {formData.objective_config.flow_trimming.max_threshold && (
                        <>
                          Max: {formData.objective_config.flow_trimming.max_threshold.value}
                          {formData.objective_config.flow_trimming.max_threshold.type === 'percentile' ? '%ile' : ' ML/day'}
                        </>
                      )}
                    </dd>
                  </div>
                )}
              </dl>
            </div>
            
            {(createMutation.isError || runMutation.isError) && (
              <div className="flex items-center p-4 bg-red-50 text-red-700 rounded-lg">
                <AlertCircle className="w-5 h-5 mr-2" />
                {(createMutation.error as Error)?.message || (runMutation.error as Error)?.message}
              </div>
            )}
          </div>
        )
    }
  }
  
  const canProceed = () => {
    switch (step) {
      case 0:
        return formData.catchment_id && formData.name
      case 1:
        return formData.model_type
      case 2:
        return formData.objective_config?.type
      case 3:
        return formData.algorithm_config?.method
      default:
        return true
    }
  }
  
  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-8">New Experiment</h1>
      
      {/* Steps indicator */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {STEPS.map((s, index) => (
            <div key={s.id} className="flex items-center">
              <button
                onClick={() => index < step && setStep(index)}
                className={`flex items-center justify-center w-10 h-10 rounded-full border-2 transition-colors ${
                  index < step
                    ? 'bg-primary-600 border-primary-600 text-white'
                    : index === step
                    ? 'border-primary-600 text-primary-600'
                    : 'border-gray-300 text-gray-400'
                }`}
              >
                {index < step ? <CheckCircle2 className="w-5 h-5" /> : index + 1}
              </button>
              <span className={`ml-2 text-sm font-medium ${
                index <= step ? 'text-gray-900' : 'text-gray-400'
              }`}>
                {s.title}
              </span>
              {index < STEPS.length - 1 && (
                <div className={`w-12 h-0.5 mx-4 ${
                  index < step ? 'bg-primary-600' : 'bg-gray-300'
                }`} />
              )}
            </div>
          ))}
        </div>
      </div>
      
      {/* Step content */}
      <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
        {renderStep()}
      </div>
      
      {/* Navigation */}
      <div className="flex justify-between">
        <button
          onClick={handleBack}
          disabled={step === 0}
          className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ChevronLeft className="w-4 h-4 mr-2" />
          Back
        </button>
        
        <div className="flex gap-3">
          {step === STEPS.length - 1 ? (
            <>
              <button
                onClick={() => handleSubmit(false)}
                disabled={createMutation.isPending}
                className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
              >
                Save as Draft
              </button>
              <button
                onClick={() => handleSubmit(true)}
                disabled={createMutation.isPending || runMutation.isPending}
                className="inline-flex items-center px-4 py-2 border border-transparent rounded-lg text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 disabled:opacity-50"
              >
                <Play className="w-4 h-4 mr-2" />
                {createMutation.isPending || runMutation.isPending ? 'Starting...' : 'Start Calibration'}
              </button>
            </>
          ) : (
            <button
              onClick={handleNext}
              disabled={!canProceed()}
              className="inline-flex items-center px-4 py-2 border border-transparent rounded-lg text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
              <ChevronRight className="w-4 h-4 ml-2" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
