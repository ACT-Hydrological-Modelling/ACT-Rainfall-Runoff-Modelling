import { useMemo, useRef } from 'react'
import Plot from 'react-plotly.js'
import type { PlotlyFigure } from '../../types/analysis'

interface PlotlyChartProps {
  figure: PlotlyFigure | null | undefined
  loading?: boolean
  className?: string
  /** When true, the chart height is driven by the figure's layout.height
   *  instead of filling 100% of the CSS container. This prevents overflow
   *  when the server-side figure height is larger than a fixed CSS class. */
  autoHeight?: boolean
}

export default function PlotlyChart({ figure, loading, className, autoHeight }: PlotlyChartProps) {
  const revisionRef = useRef(0)
  const prevFigureRef = useRef<PlotlyFigure | null | undefined>(null)

  if (figure !== prevFigureRef.current) {
    prevFigureRef.current = figure
    revisionRef.current += 1
  }

  const layout = useMemo(() => {
    if (!figure?.layout) return {}
    return {
      ...figure.layout,
      autosize: true,
    }
  }, [figure?.layout])

  const figureHeight = autoHeight
    ? (figure?.layout?.height as number | undefined) ?? 450
    : undefined

  if (loading) {
    return (
      <div className={`flex items-center justify-center h-64 bg-white rounded-lg border border-gray-200 ${className || ''}`}>
        <div className="text-gray-400 text-sm">Loading chart...</div>
      </div>
    )
  }

  if (!figure?.data?.length) {
    return (
      <div className={`flex items-center justify-center h-64 bg-white rounded-lg border border-gray-200 ${className || ''}`}>
        <div className="text-gray-400 text-sm">No data available</div>
      </div>
    )
  }

  return (
    <div className={className} style={figureHeight ? { height: figureHeight } : undefined}>
      <Plot
        data={figure.data}
        layout={layout}
        revision={revisionRef.current}
        config={{
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          displaylogo: false,
          scrollZoom: true,
        }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}
