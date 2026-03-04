import { useState, useMemo } from 'react'
import { ChevronDown, ChevronUp, Check, X } from 'lucide-react'
import type { ExperimentInfo } from '../../types/analysis'

interface ExperimentSelectorProps {
  experiments: ExperimentInfo[]
  selected: Set<string>
  onToggle: (key: string) => void
  onSelectAll: () => void
  onSelectNone: () => void
}

export default function ExperimentSelector({
  experiments,
  selected,
  onToggle,
  onSelectAll,
  onSelectNone,
}: ExperimentSelectorProps) {
  const [expanded, setExpanded] = useState(true)
  const [filterModel, setFilterModel] = useState<string>('')
  const [filterObjective, setFilterObjective] = useState<string>('')

  const models = useMemo(
    () => [...new Set(experiments.map((e) => e.model))].sort(),
    [experiments]
  )
  const objectives = useMemo(
    () => [...new Set(experiments.map((e) => e.objective))].sort(),
    [experiments]
  )

  const filtered = useMemo(() => {
    return experiments.filter((e) => {
      if (filterModel && e.model !== filterModel) return false
      if (filterObjective && e.objective !== filterObjective) return false
      return true
    })
  }, [experiments, filterModel, filterObjective])

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50"
      >
        <span>
          Experiments ({selected.size}/{experiments.length} selected)
        </span>
        {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {expanded && (
        <div className="border-t border-gray-200 px-4 py-3 space-y-3">
          <div className="flex gap-2">
            <select
              value={filterModel}
              onChange={(e) => setFilterModel(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1"
            >
              <option value="">All models</option>
              {models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <select
              value={filterObjective}
              onChange={(e) => setFilterObjective(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1"
            >
              <option value="">All objectives</option>
              {objectives.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
            <button
              onClick={onSelectAll}
              className="text-xs text-primary-600 hover:text-primary-800 flex items-center gap-1"
            >
              <Check className="w-3 h-3" /> All
            </button>
            <button
              onClick={onSelectNone}
              className="text-xs text-gray-500 hover:text-gray-700 flex items-center gap-1"
            >
              <X className="w-3 h-3" /> None
            </button>
          </div>

          <div className="max-h-60 overflow-y-auto space-y-1">
            {filtered.map((exp) => {
              const isChecked = selected.has(exp.key)
              const label = [exp.model, exp.objective, exp.transformation, exp.algorithm]
                .filter(Boolean)
                .join(' / ')
              return (
                <label
                  key={exp.key}
                  className="flex items-center gap-2 px-2 py-1 rounded hover:bg-gray-50 cursor-pointer text-xs"
                >
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => onToggle(exp.key)}
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className={isChecked ? 'text-gray-900' : 'text-gray-400'}>
                    {label}
                  </span>
                  {exp.best_objective != null && (
                    <span className="ml-auto text-gray-400 tabular-nums">
                      {exp.best_objective.toFixed(4)}
                    </span>
                  )}
                </label>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
