import { useState, useMemo } from 'react'
import Plot from 'react-plotly.js'
import { Calendar, ZoomIn, Maximize2 } from 'lucide-react'
import type { CatchmentTimeseries } from '../services/api'

interface CalibrationPeriodSelectorProps {
  timeseries: CatchmentTimeseries | null
  isLoading: boolean
  startDate: string
  endDate: string
  warmupDays: number
  onPeriodChange: (start: string, end: string) => void
  onWarmupChange: (days: number) => void
}

export default function CalibrationPeriodSelector({
  timeseries,
  isLoading,
  startDate,
  endDate,
  warmupDays,
  onPeriodChange,
  onWarmupChange,
}: CalibrationPeriodSelectorProps) {
  const [showFullRange, setShowFullRange] = useState(false)
  
  // Calculate warmup end date
  const warmupEndDate = useMemo(() => {
    if (!startDate) return null
    const start = new Date(startDate)
    start.setDate(start.getDate() + warmupDays)
    return start.toISOString().split('T')[0]
  }, [startDate, warmupDays])
  
  // Calculate statistics for selected period based on actual calendar days
  const periodStats = useMemo(() => {
    if (!startDate || !endDate) return null
    
    // Calculate actual calendar days between dates
    const start = new Date(startDate)
    const end = new Date(endDate)
    const diffTime = end.getTime() - start.getTime()
    const selectedDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24)) + 1 // +1 to include both start and end
    
    if (selectedDays <= 0) return null
    
    const actualWarmupDays = Math.min(warmupDays, selectedDays)
    const calibrationDays = Math.max(0, selectedDays - actualWarmupDays)
    
    return {
      totalDays: selectedDays,
      warmupDays: actualWarmupDays,
      calibrationDays,
    }
  }, [startDate, endDate, warmupDays])
  
  if (isLoading) {
    return (
      <div className="h-96 flex items-center justify-center bg-gray-50 rounded-lg border">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-4 border-primary-500 border-t-transparent rounded-full mx-auto"></div>
          <p className="mt-2 text-gray-500">Loading time series data...</p>
        </div>
      </div>
    )
  }
  
  if (!timeseries) {
    return (
      <div className="h-64 flex items-center justify-center bg-gray-50 rounded-lg border">
        <div className="text-center text-gray-500">
          <Calendar className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>Select a catchment to view time series</p>
        </div>
      </div>
    )
  }
  
  // Prepare plot data
  const rainfallTrace = timeseries.rainfall ? {
    x: timeseries.dates,
    y: timeseries.rainfall,
    type: 'bar' as const,
    name: 'Rainfall',
    marker: { color: '#3b82f6' },
    yaxis: 'y3',
  } : null
  
  const petTrace = timeseries.pet ? {
    x: timeseries.dates,
    y: timeseries.pet,
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: 'PET',
    line: { color: '#f97316', width: 1 },
    yaxis: 'y2',
  } : null
  
  const flowTrace = timeseries.observed_flow ? {
    x: timeseries.dates,
    y: timeseries.observed_flow,
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: 'Observed Flow',
    line: { color: '#1f77b4', width: 1.5 },
  } : null
  
  const traces = [rainfallTrace, petTrace, flowTrace].filter(Boolean)
  
  // Add selection shapes - build array without nulls
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const shapes: any[] = []
  
  // Warmup period (yellow/amber)
  if (startDate && warmupEndDate && warmupEndDate <= endDate) {
    shapes.push({
      type: 'rect',
      xref: 'x',
      yref: 'paper',
      x0: startDate,
      x1: warmupEndDate,
      y0: 0,
      y1: 1,
      fillcolor: 'rgba(251, 191, 36, 0.2)',
      line: { width: 0 },
      layer: 'below',
    })
  }
  
  // Calibration period (green)
  if (warmupEndDate && endDate) {
    shapes.push({
      type: 'rect',
      xref: 'x',
      yref: 'paper',
      x0: warmupEndDate > startDate ? warmupEndDate : startDate,
      x1: endDate,
      y0: 0,
      y1: 1,
      fillcolor: 'rgba(34, 197, 94, 0.15)',
      line: { width: 0 },
      layer: 'below',
    })
  }
  
  // Start line
  if (startDate) {
    shapes.push({
      type: 'line',
      xref: 'x',
      yref: 'paper',
      x0: startDate,
      x1: startDate,
      y0: 0,
      y1: 1,
      line: { color: '#16a34a', width: 2 },
    })
  }
  
  // End line
  if (endDate) {
    shapes.push({
      type: 'line',
      xref: 'x',
      yref: 'paper',
      x0: endDate,
      x1: endDate,
      y0: 0,
      y1: 1,
      line: { color: '#dc2626', width: 2 },
    })
  }
  
  // Warmup end line
  if (warmupEndDate && warmupEndDate > startDate && warmupEndDate < endDate) {
    shapes.push({
      type: 'line',
      xref: 'x',
      yref: 'paper',
      x0: warmupEndDate,
      x1: warmupEndDate,
      y0: 0,
      y1: 1,
      line: { color: '#f59e0b', width: 1.5, dash: 'dash' },
    })
  }
  
  const layout = {
    height: 400,
    margin: { l: 60, r: 60, t: 30, b: 80 },
    showlegend: true,
    legend: { orientation: 'h' as const, y: 1.12 },
    xaxis: {
      title: { text: 'Date' },
      rangeslider: { visible: true, thickness: 0.1 },
      type: 'date' as const,
      range: showFullRange 
        ? [timeseries.data_range.start, timeseries.data_range.end]
        : [startDate, endDate],
    },
    yaxis: {
      title: { text: 'Flow (ML/day)' },
      domain: [0, 0.55],
      gridcolor: '#e5e7eb',
    },
    yaxis2: {
      title: { text: 'PET (mm)' },
      domain: [0.6, 0.75],
      gridcolor: '#e5e7eb',
    },
    yaxis3: {
      title: { text: 'Rainfall (mm)' },
      domain: [0.8, 1],
      autorange: 'reversed' as const,
      gridcolor: '#e5e7eb',
    },
    shapes,
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { size: 11 },
    annotations: warmupEndDate && warmupEndDate > startDate && warmupEndDate < endDate ? [{
      x: warmupEndDate,
      y: 1.02,
      xref: 'x' as const,
      yref: 'paper' as const,
      text: 'Warmup ends',
      showarrow: false,
      font: { size: 10, color: '#f59e0b' },
    }] : [],
  }
  
  // Handle plot selection
  const handleRelayout = (event: any) => {
    // Check if this is a range selection event
    if (event['xaxis.range[0]'] && event['xaxis.range[1]']) {
      const newStart = event['xaxis.range[0]'].split('T')[0]
      const newEnd = event['xaxis.range[1]'].split('T')[0]
      onPeriodChange(newStart, newEnd)
    }
  }
  
  return (
    <div className="space-y-4">
      {/* Info and controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-amber-200 border border-amber-400 rounded"></div>
            <span className="text-gray-600">Warmup Period</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-green-200 border border-green-400 rounded"></div>
            <span className="text-gray-600">Calibration Period</span>
          </div>
        </div>
        <button
          onClick={() => setShowFullRange(!showFullRange)}
          className="flex items-center space-x-1 text-sm text-primary-600 hover:text-primary-700"
        >
          {showFullRange ? <ZoomIn className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          <span>{showFullRange ? 'Zoom to Selection' : 'Show Full Range'}</span>
        </button>
      </div>
      
      {/* Plot */}
      <div className="border rounded-lg overflow-hidden bg-white">
        <Plot
          data={traces as any}
          layout={layout}
          config={{ 
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            displaylogo: false,
            responsive: true,
          }}
          style={{ width: '100%' }}
          onRelayout={handleRelayout}
        />
      </div>
      
      {/* Date inputs and stats */}
      <div className="grid grid-cols-4 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Start Date
          </label>
          <input
            type="date"
            value={startDate}
            min={timeseries.calibration_range.start}
            max={timeseries.calibration_range.end}
            onChange={(e) => onPeriodChange(e.target.value, endDate)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 text-sm"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            End Date
          </label>
          <input
            type="date"
            value={endDate}
            min={timeseries.calibration_range.start}
            max={timeseries.calibration_range.end}
            onChange={(e) => onPeriodChange(startDate, e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 text-sm"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Warmup (days)
          </label>
          <input
            type="number"
            value={warmupDays}
            min={0}
            max={730}
            onChange={(e) => onWarmupChange(parseInt(e.target.value) || 0)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 text-sm"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Calibration Days
          </label>
          <div className="px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm">
            {periodStats ? (
              <span className="font-medium text-gray-900">
                {periodStats.calibrationDays.toLocaleString()}
              </span>
            ) : (
              <span className="text-gray-400">—</span>
            )}
          </div>
        </div>
      </div>
      
      {/* Quick select buttons */}
      <div className="flex items-center space-x-2">
        <span className="text-sm text-gray-500">Quick select:</span>
        <button
          onClick={() => onPeriodChange(timeseries.calibration_range.start, timeseries.calibration_range.end)}
          className="px-3 py-1 text-sm border border-gray-300 rounded-lg hover:bg-gray-50"
        >
          Full Range
        </button>
        {/* Generate decade buttons based on calibration range */}
        {(() => {
          const startYear = parseInt(timeseries.calibration_range.start.split('-')[0])
          const endYear = parseInt(timeseries.calibration_range.end.split('-')[0])
          const buttons: React.ReactNode[] = []
          
          // Last N years buttons
          const yearsOptions = [5, 10, 20]
          yearsOptions.forEach(years => {
            const yearStart = Math.max(startYear, endYear - years)
            if (yearStart < endYear) {
              buttons.push(
                <button
                  key={`last-${years}`}
                  onClick={() => onPeriodChange(`${yearStart}-01-01`, timeseries.calibration_range.end)}
                  className="px-3 py-1 text-sm border border-gray-300 rounded-lg hover:bg-gray-50"
                >
                  Last {years}y
                </button>
              )
            }
          })
          
          return buttons
        })()}
      </div>
      
      {/* Data range info */}
      <div className="text-xs text-gray-500 flex flex-col space-y-1">
        <span>
          Observed flow data: {timeseries.calibration_range.start} to {timeseries.calibration_range.end} ({timeseries.calibration_range.total_days.toLocaleString()} days)
        </span>
        {timeseries.statistics.observed_flow && (
          <span>
            Flow range: {timeseries.statistics.observed_flow.min?.toFixed(1)} - {timeseries.statistics.observed_flow.max?.toFixed(0)} ML/day 
            (median: {timeseries.statistics.observed_flow.p50?.toFixed(1)})
          </span>
        )}
      </div>
    </div>
  )
}
