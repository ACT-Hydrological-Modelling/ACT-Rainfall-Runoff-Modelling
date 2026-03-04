import { useMemo } from 'react'
import Plot from 'react-plotly.js'
import type { ClustermapData } from '../../types/analysis'

interface ClustermapChartProps {
  data: ClustermapData | null
  height?: number
}

export default function ClustermapChart({ data, height = 600 }: ClustermapChartProps) {
  const plotData = useMemo(() => {
    if (!data) return null

    const nRows = data.row_labels.length
    const nCols = data.col_labels.length

    const annotationText = data.annotations.map(row =>
      row.map(v => v ?? '')
    )

    // Dendrogram leaf positions: scipy uses 5, 15, 25, ... (i.e. i*10 + 5)
    const rowLeafPositions = data.row_labels.map((_, i) => i * 10 + 5)
    const colLeafPositions = data.col_labels.map((_, i) => i * 10 + 5)

    const rowMin = rowLeafPositions[0] - 5
    const rowMax = rowLeafPositions[nRows - 1] + 5
    const colMin = colLeafPositions[0] - 5
    const colMax = colLeafPositions[nCols - 1] + 5

    // Layout domains:
    //   Left: row labels (handled by yaxis ticklabels)
    //   Centre: heatmap + column dendrogram above
    //   Right: row dendrogram
    //   Bottom: column labels (xaxis ticklabels)
    const heatLeft = 0.0
    const heatRight = 0.88
    const dendRight_left = 0.88
    const dendRight_right = 1.0
    const heatBottom = 0.0
    const heatTop = 0.78
    const dendTop_bottom = 0.78
    const dendTop_top = 1.0

    // Green (1.0) → Yellow (0.5) → Red (0.0)
    const colorscale: [number, string][] = [
      [0.0, '#d73027'],
      [0.25, '#fc8d59'],
      [0.5, '#fee08b'],
      [0.75, '#d9ef8b'],
      [1.0, '#1a9850'],
    ]

    const heatmap: Plotly.Data = {
      z: data.heatmap_values,
      x: data.col_labels,
      y: data.row_labels,
      type: 'heatmap' as const,
      colorscale,
      zmin: 0,
      zmax: 1,
      text: annotationText as any,
      texttemplate: '%{text}',
      textfont: { size: 8 },
      hovertemplate: 'Experiment: %{y}<br>Metric: %{x}<br>Score: %{z:.3f}<extra></extra>',
      xaxis: 'x',
      yaxis: 'y',
      showscale: true,
      colorbar: {
        title: { text: 'Normalised Score', side: 'right' as const },
        len: 0.5,
        y: 0.4,
        x: 1.01,
        thickness: 12,
        tickvals: [0, 0.25, 0.5, 0.75, 1.0],
        ticktext: ['0.0', '0.25', '0.5', '0.75', '1.0'],
      },
    }

    // Column dendrogram (top) — x maps to column leaf positions
    const colDendTraces: Plotly.Data[] = data.col_dendrogram.icoord.map((ic, i) => ({
      x: ic,
      y: data.col_dendrogram.dcoord[i],
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: '#555', width: 1.2 },
      xaxis: 'x2',
      yaxis: 'y2',
      showlegend: false,
      hoverinfo: 'skip' as const,
    }))

    // Row dendrogram (right) — y maps to row leaf positions, x is distance
    const rowDendTraces: Plotly.Data[] = data.row_dendrogram.icoord.map((ic, i) => ({
      x: data.row_dendrogram.dcoord[i],
      y: ic,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: '#555', width: 1.2 },
      xaxis: 'x3',
      yaxis: 'y3',
      showlegend: false,
      hoverinfo: 'skip' as const,
    }))

    const layout: Partial<Plotly.Layout> = {
      height,
      margin: { l: 220, r: 20, t: 10, b: 120 },

      // Heatmap axes
      xaxis: {
        domain: [heatLeft, heatRight],
        anchor: 'y',
        side: 'bottom',
        tickangle: -50,
        tickfont: { size: 9, family: 'monospace' },
        showgrid: false,
      },
      yaxis: {
        domain: [heatBottom, heatTop],
        anchor: 'x',
        side: 'left',
        tickfont: { size: 8, family: 'monospace' },
        autorange: 'reversed' as const,
        showgrid: false,
      },

      // Column dendrogram (top, aligned with heatmap x)
      xaxis2: {
        domain: [heatLeft, heatRight],
        anchor: 'y2',
        showticklabels: false,
        showgrid: false,
        zeroline: false,
        range: [colMin, colMax],
        matches: undefined,
      },
      yaxis2: {
        domain: [dendTop_bottom, dendTop_top],
        anchor: 'x2',
        showticklabels: false,
        showgrid: false,
        zeroline: false,
      },

      // Row dendrogram (right, aligned with heatmap y)
      xaxis3: {
        domain: [dendRight_left, dendRight_right],
        anchor: 'y3',
        showticklabels: false,
        showgrid: false,
        zeroline: false,
      },
      yaxis3: {
        domain: [heatBottom, heatTop],
        anchor: 'x3',
        showticklabels: false,
        showgrid: false,
        zeroline: false,
        range: [rowMax, rowMin],
      },
    }

    return {
      traces: [heatmap, ...colDendTraces, ...rowDendTraces],
      layout,
    }
  }, [data, height])

  if (!data || !plotData) {
    return (
      <div className="flex items-center justify-center h-64 bg-white rounded-lg border border-gray-200">
        <div className="text-gray-400 text-sm">
          {data === null ? 'Not enough data for clustering (need at least 2 experiments)' : 'Loading...'}
        </div>
      </div>
    )
  }

  return (
    <Plot
      data={plotData.traces}
      layout={plotData.layout}
      config={{
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      }}
      useResizeHandler
      style={{ width: '100%' }}
    />
  )
}
