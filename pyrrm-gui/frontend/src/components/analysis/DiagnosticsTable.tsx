import { useMemo, useState } from 'react'
import { ArrowUpDown } from 'lucide-react'
import type { DiagnosticsRow } from '../../types/analysis'

interface DiagnosticsTableProps {
  rows: DiagnosticsRow[]
  metricColumns: string[]
  title?: string
  normalized?: boolean
  highlightKeys?: Set<string>
}

const HIGHER_IS_BETTER = new Set([
  'NSE', 'NSE_sqrt', 'NSE_log', 'NSE_inv',
  'KGE', 'KGE_sqrt', 'KGE_log', 'KGE_inv',
  'KGE_np', 'KGE_np_sqrt', 'KGE_np_log', 'KGE_np_inv',
])

const ERROR_METRICS = new Set(['RMSE', 'MAE', 'SDEB'])

function cellColor(metric: string, value: number | null, normalized: boolean): string {
  if (value == null) return ''

  if (normalized) {
    if (value >= 0.75) return 'bg-green-100 text-green-900'
    if (value >= 0.5) return 'bg-yellow-50 text-yellow-900'
    if (value >= 0.25) return 'bg-orange-50 text-orange-900'
    return 'bg-red-50 text-red-900'
  }

  if (HIGHER_IS_BETTER.has(metric)) {
    if (value >= 0.75) return 'bg-green-100 text-green-900'
    if (value >= 0.5) return 'bg-yellow-50 text-yellow-900'
    return 'bg-red-50 text-red-900'
  }

  if (ERROR_METRICS.has(metric)) {
    return ''
  }

  if (Math.abs(value) <= 10) return 'bg-green-100 text-green-900'
  if (Math.abs(value) <= 25) return 'bg-yellow-50 text-yellow-900'
  return 'bg-red-50 text-red-900'
}

export default function DiagnosticsTable({
  rows,
  metricColumns,
  title,
  normalized = false,
  highlightKeys,
}: DiagnosticsTableProps) {
  const [sortCol, setSortCol] = useState<string | null>(null)
  const [sortAsc, setSortAsc] = useState(true)

  const sortedRows = useMemo(() => {
    if (!sortCol) return rows
    return [...rows].sort((a, b) => {
      const va = a.metrics[sortCol] ?? -Infinity
      const vb = b.metrics[sortCol] ?? -Infinity
      return sortAsc ? (va as number) - (vb as number) : (vb as number) - (va as number)
    })
  }, [rows, sortCol, sortAsc])

  const handleSort = (col: string) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc)
    } else {
      setSortCol(col)
      setSortAsc(false)
    }
  }

  const passCount = highlightKeys ? highlightKeys.size : rows.length

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      {title && (
        <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-700">{title}</h3>
          {highlightKeys && (
            <span className="text-xs text-gray-400">
              {passCount} of {rows.length} pass filters
            </span>
          )}
        </div>
      )}
      <div className="overflow-x-auto">
        <table className="min-w-full text-xs">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-3 py-2 text-left font-medium text-gray-500 sticky left-0 bg-gray-50 z-10 min-w-[320px]">
                Experiment
              </th>
              {metricColumns.map((col) => (
                <th
                  key={col}
                  className="px-2 py-2 text-right font-medium text-gray-500 cursor-pointer hover:bg-gray-100 whitespace-nowrap"
                  onClick={() => handleSort(col)}
                >
                  <div className="flex items-center justify-end gap-1">
                    {col}
                    <ArrowUpDown className="w-3 h-3 text-gray-400" />
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sortedRows.map((row) => {
              const dimmed = highlightKeys && !highlightKeys.has(row.experiment_key)
              return (
                <tr
                  key={row.experiment_key}
                  className={`hover:bg-gray-50 transition-opacity ${dimmed ? 'opacity-25' : ''}`}
                >
                  <td
                    className={`px-3 py-1.5 font-mono sticky left-0 z-10 border-r border-gray-100 whitespace-nowrap ${
                      dimmed ? 'text-gray-400 bg-gray-50' : 'text-gray-700 bg-white'
                    }`}
                    title={row.experiment_key}
                  >
                    {row.experiment_key}
                  </td>
                  {metricColumns.map((col) => {
                    const val = row.metrics[col]
                    const bg = dimmed ? '' : cellColor(col, val, normalized)
                    return (
                      <td key={col} className={`px-2 py-1.5 text-right tabular-nums ${bg}`}>
                        {val != null ? val.toFixed(3) : '—'}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
