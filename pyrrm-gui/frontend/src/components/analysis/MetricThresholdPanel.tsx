import { useState, useMemo, useCallback, useEffect } from 'react'
import { ChevronDown, ChevronRight, RotateCcw, Filter, Beaker } from 'lucide-react'
import type { DiagnosticsRow } from '../../types/analysis'

// ─── Experiment key parsing ─────────────────────────────────────────────────

interface ParsedExperiment {
  model: string
  objective: string
  transformation: string
  algorithm: string
}

function parseExperimentKey(key: string): ParsedExperiment {
  const parts = key.split('_')
  if (parts.length >= 4) {
    const model = parts[1]
    const objective = parts[2]
    const algorithm = parts[parts.length - 1]
    let transformation = 'none'
    if (parts.length >= 5) {
      const middle = parts[3]
      transformation = middle.includes('-') ? middle.split('-')[0] : middle
    }
    return { model, objective, transformation, algorithm }
  }
  return { model: 'unknown', objective: 'unknown', transformation: 'none', algorithm: 'unknown' }
}

export interface IngredientConfig {
  models: Set<string>
  objectives: Set<string>
  transformations: Set<string>
  algorithms: Set<string>
}

const INGREDIENT_CATEGORIES = [
  { key: 'models' as const, label: 'Model' },
  { key: 'objectives' as const, label: 'Objective' },
  { key: 'transformations' as const, label: 'Transformation' },
  { key: 'algorithms' as const, label: 'Algorithm' },
] as const

function extractIngredients(rows: DiagnosticsRow[]): {
  parsed: Map<string, ParsedExperiment>
  available: Record<keyof IngredientConfig, string[]>
} {
  const parsed = new Map<string, ParsedExperiment>()
  const sets: Record<keyof IngredientConfig, Set<string>> = {
    models: new Set(),
    objectives: new Set(),
    transformations: new Set(),
    algorithms: new Set(),
  }

  for (const row of rows) {
    const p = parseExperimentKey(row.experiment_key)
    parsed.set(row.experiment_key, p)
    sets.models.add(p.model)
    sets.objectives.add(p.objective)
    sets.transformations.add(p.transformation)
    sets.algorithms.add(p.algorithm)
  }

  return {
    parsed,
    available: {
      models: [...sets.models].sort(),
      objectives: [...sets.objectives].sort(),
      transformations: [...sets.transformations].sort(),
      algorithms: [...sets.algorithms].sort(),
    },
  }
}

// ─── Metric category definitions ────────────────────────────────────────────

const SKILL_METRICS = [
  'NSE', 'NSE_sqrt', 'NSE_log', 'NSE_inv',
  'KGE', 'KGE_sqrt', 'KGE_log', 'KGE_inv',
  'KGE_np', 'KGE_np_sqrt', 'KGE_np_log', 'KGE_np_inv',
] as const

const ERROR_METRICS = ['RMSE', 'MAE', 'SDEB'] as const

const VOLUME_METRICS = [
  'PBIAS', 'FHV', 'FMV', 'FLV',
  'Sig_BFI', 'Sig_Flash', 'Sig_Q95', 'Sig_Q5',
] as const

const QUANTILE_OPTIONS = [
  { label: 'Off', value: null },
  { label: 'P5', value: 0.05 },
  { label: 'P10', value: 0.10 },
  { label: 'P15', value: 0.15 },
  { label: 'P25', value: 0.25 },
  { label: 'P50', value: 0.50 },
  { label: 'P75', value: 0.75 },
  { label: 'P90', value: 0.90 },
] as const

type QuantileKey = null | 0.05 | 0.10 | 0.15 | 0.25 | 0.50 | 0.75 | 0.90

// ─── Threshold state types ──────────────────────────────────────────────────

export interface ThresholdConfig {
  skill: Record<string, number | null>
  error: Record<string, QuantileKey>
  volume: Record<string, number | null>
  ingredients: IngredientConfig
}

function defaultThresholds(
  availableCols: string[],
  availableIngredients?: Record<keyof IngredientConfig, string[]>,
): ThresholdConfig {
  const cols = new Set(availableCols)
  const skill: Record<string, number | null> = {}
  const error: Record<string, QuantileKey> = {}
  const volume: Record<string, number | null> = {}

  for (const m of SKILL_METRICS) if (cols.has(m)) skill[m] = null
  for (const m of ERROR_METRICS) if (cols.has(m)) error[m] = null
  for (const m of VOLUME_METRICS) if (cols.has(m)) volume[m] = null

  const ingredients: IngredientConfig = {
    models: new Set(availableIngredients?.models ?? []),
    objectives: new Set(availableIngredients?.objectives ?? []),
    transformations: new Set(availableIngredients?.transformations ?? []),
    algorithms: new Set(availableIngredients?.algorithms ?? []),
  }

  return { skill, error, volume, ingredients }
}

// ─── Quantile computation ───────────────────────────────────────────────────

function computeQuantile(values: number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b)
  const pos = q * (sorted.length - 1)
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  if (lo === hi) return sorted[lo]
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
}

// ─── Pass / fail logic ──────────────────────────────────────────────────────

export function computePassingKeys(
  rows: DiagnosticsRow[],
  config: ThresholdConfig,
  parsedMap?: Map<string, ParsedExperiment>,
): Set<string> {
  const quantileValues: Record<string, Record<number, number>> = {}
  for (const [metric, qKey] of Object.entries(config.error)) {
    if (qKey == null) continue
    const vals = rows
      .map((r) => r.metrics[metric])
      .filter((v): v is number => v != null)
    if (vals.length === 0) continue
    quantileValues[metric] = {}
    for (const opt of QUANTILE_OPTIONS) {
      if (opt.value != null) {
        quantileValues[metric][opt.value] = computeQuantile(vals, opt.value)
      }
    }
  }

  const { ingredients } = config
  const passing = new Set<string>()

  for (const row of rows) {
    let pass = true

    if (parsedMap) {
      const parsed = parsedMap.get(row.experiment_key)
      if (parsed) {
        if (!ingredients.models.has(parsed.model)) pass = false
        if (pass && !ingredients.objectives.has(parsed.objective)) pass = false
        if (pass && !ingredients.transformations.has(parsed.transformation)) pass = false
        if (pass && !ingredients.algorithms.has(parsed.algorithm)) pass = false
      }
    }

    if (pass) {
      for (const [metric, minVal] of Object.entries(config.skill)) {
        if (minVal == null) continue
        const v = row.metrics[metric]
        if (v == null || v < minVal) { pass = false; break }
      }
    }

    if (pass) {
      for (const [metric, qKey] of Object.entries(config.error)) {
        if (qKey == null) continue
        const maxVal = quantileValues[metric]?.[qKey]
        if (maxVal == null) continue
        const v = row.metrics[metric]
        if (v == null || v > maxVal) { pass = false; break }
      }
    }

    if (pass) {
      for (const [metric, tol] of Object.entries(config.volume)) {
        if (tol == null) continue
        const v = row.metrics[metric]
        if (v == null || Math.abs(v) > tol) { pass = false; break }
      }
    }

    if (pass) passing.add(row.experiment_key)
  }

  return passing
}

// ─── Resolved quantile values (for display) ────────────────────────────────

function useResolvedQuantiles(
  rows: DiagnosticsRow[],
  errorConfig: Record<string, QuantileKey>,
): Record<string, Record<number, number>> {
  return useMemo(() => {
    const result: Record<string, Record<number, number>> = {}
    for (const metric of Object.keys(errorConfig)) {
      const vals = rows
        .map((r) => r.metrics[metric])
        .filter((v): v is number => v != null)
      if (vals.length === 0) continue
      result[metric] = {}
      for (const opt of QUANTILE_OPTIONS) {
        if (opt.value != null) {
          result[metric][opt.value] = computeQuantile(vals, opt.value)
        }
      }
    }
    return result
  }, [rows, errorConfig])
}

// ─── Sub-components ─────────────────────────────────────────────────────────

function CollapsibleGroup({
  title,
  activeCount,
  totalCount,
  defaultOpen = true,
  children,
}: {
  title: string
  activeCount: number
  totalCount: number
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-gray-200 rounded-lg">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2 bg-gray-50 rounded-t-lg hover:bg-gray-100 transition-colors"
      >
        <div className="flex items-center gap-2">
          {open ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronRight className="w-4 h-4 text-gray-400" />}
          <span className="text-sm font-medium text-gray-700">{title}</span>
        </div>
        <span className="text-xs text-gray-400">
          {activeCount}/{totalCount} active
        </span>
      </button>
      {open && <div className="p-3 space-y-2">{children}</div>}
    </div>
  )
}

function SkillSlider({
  metric,
  value,
  onChange,
}: {
  metric: string
  value: number | null
  onChange: (metric: string, val: number | null) => void
}) {
  const isInverse = metric.includes('_inv')
  const min = isInverse ? -5.0 : -1.0
  const max = 1.0
  const step = 0.05
  const active = value != null

  return (
    <div className="flex items-center gap-2 text-xs">
      <label className="w-28 font-mono text-gray-600 flex items-center gap-1.5">
        <input
          type="checkbox"
          checked={active}
          onChange={(e) => onChange(metric, e.target.checked ? 0.0 : null)}
          className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
        />
        {metric}
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value ?? 0}
        disabled={!active}
        onChange={(e) => onChange(metric, parseFloat(e.target.value))}
        className="flex-1 h-1.5 accent-primary-600 disabled:opacity-30"
      />
      <span className={`w-12 text-right tabular-nums font-mono ${active ? 'text-gray-700' : 'text-gray-300'}`}>
        {active ? `≥ ${value!.toFixed(2)}` : 'off'}
      </span>
    </div>
  )
}

function ErrorQuantileSelector({
  metric,
  value,
  resolvedValue,
  onChange,
}: {
  metric: string
  value: QuantileKey
  resolvedValue: number | undefined
  onChange: (metric: string, val: QuantileKey) => void
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-28 font-mono text-gray-600">{metric}</span>
      <div className="flex gap-1">
        {QUANTILE_OPTIONS.map((opt) => (
          <button
            key={opt.label}
            onClick={() => onChange(metric, opt.value)}
            className={`px-2 py-0.5 rounded text-xs font-medium transition-colors ${
              value === opt.value
                ? 'bg-primary-100 text-primary-700 ring-1 ring-primary-300'
                : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
      <span className="w-20 text-right tabular-nums font-mono text-gray-500">
        {value != null && resolvedValue != null
          ? `≤ ${resolvedValue.toFixed(2)}`
          : '—'}
      </span>
    </div>
  )
}

function VolumeSlider({
  metric,
  value,
  onChange,
}: {
  metric: string
  value: number | null
  onChange: (metric: string, val: number | null) => void
}) {
  const active = value != null

  return (
    <div className="flex items-center gap-2 text-xs">
      <label className="w-28 font-mono text-gray-600 flex items-center gap-1.5">
        <input
          type="checkbox"
          checked={active}
          onChange={(e) => onChange(metric, e.target.checked ? 25 : null)}
          className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
        />
        {metric}
      </label>
      <input
        type="range"
        min={1}
        max={100}
        step={1}
        value={value ?? 25}
        disabled={!active}
        onChange={(e) => onChange(metric, parseFloat(e.target.value))}
        className="flex-1 h-1.5 accent-primary-600 disabled:opacity-30"
      />
      <span className={`w-16 text-right tabular-nums font-mono ${active ? 'text-gray-700' : 'text-gray-300'}`}>
        {active ? `± ${value!.toFixed(0)}%` : 'off'}
      </span>
    </div>
  )
}

// ─── Ingredient filter sub-component ────────────────────────────────────────

function IngredientCheckboxGroup({
  label,
  options,
  selected,
  onToggle,
  onToggleAll,
}: {
  label: string
  options: string[]
  selected: Set<string>
  onToggle: (value: string) => void
  onToggleAll: () => void
}) {
  const allSelected = options.length > 0 && options.every((o) => selected.has(o))
  const noneSelected = options.every((o) => !selected.has(o))
  const isIndeterminate = !allSelected && !noneSelected

  if (options.length === 0) return null

  return (
    <div>
      <div className="flex items-center gap-1.5 mb-1.5">
        <input
          type="checkbox"
          checked={allSelected}
          ref={(el) => { if (el) el.indeterminate = isIndeterminate }}
          onChange={onToggleAll}
          className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
        />
        <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
          {label}
        </span>
        <span className="text-xs text-gray-400">
          {selected.size}/{options.length}
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {options.map((opt) => {
          const isActive = selected.has(opt)
          return (
            <button
              key={opt}
              onClick={() => onToggle(opt)}
              className={`px-2 py-0.5 rounded text-xs font-medium transition-colors border ${
                isActive
                  ? 'bg-primary-50 text-primary-700 border-primary-300'
                  : 'bg-gray-50 text-gray-400 border-gray-200 hover:bg-gray-100'
              }`}
            >
              {opt}
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ─── Main component ─────────────────────────────────────────────────────────

interface MetricThresholdPanelProps {
  rows: DiagnosticsRow[]
  metricColumns: string[]
  onPassingKeysChange: (keys: Set<string>) => void
}

export default function MetricThresholdPanel({
  rows,
  metricColumns,
  onPassingKeysChange,
}: MetricThresholdPanelProps) {
  const availableSet = useMemo(() => new Set(metricColumns), [metricColumns])

  const { parsed: parsedMap, available: availableIngredients } = useMemo(
    () => extractIngredients(rows),
    [rows],
  )

  const [config, setConfig] = useState<ThresholdConfig>(() =>
    defaultThresholds(metricColumns, availableIngredients),
  )

  const resolvedQuantiles = useResolvedQuantiles(rows, config.error)

  const passingKeys = useMemo(
    () => computePassingKeys(rows, config, parsedMap),
    [rows, config, parsedMap],
  )

  useEffect(() => {
    onPassingKeysChange(passingKeys)
  }, [passingKeys, onPassingKeysChange])

  const handleSkillChange = useCallback((metric: string, val: number | null) => {
    setConfig((prev) => ({
      ...prev,
      skill: { ...prev.skill, [metric]: val },
    }))
  }, [])

  const handleErrorChange = useCallback((metric: string, val: QuantileKey) => {
    setConfig((prev) => ({
      ...prev,
      error: { ...prev.error, [metric]: val },
    }))
  }, [])

  const handleVolumeChange = useCallback((metric: string, val: number | null) => {
    setConfig((prev) => ({
      ...prev,
      volume: { ...prev.volume, [metric]: val },
    }))
  }, [])

  const handleIngredientChange = useCallback(
    (_category: keyof IngredientConfig, next: Set<string>) => {
      setConfig((prev) => ({
        ...prev,
        ingredients: { ...prev.ingredients, [_category]: next },
      }))
    },
    [],
  )

  const handleReset = useCallback(() => {
    setConfig(defaultThresholds(metricColumns, availableIngredients))
  }, [metricColumns, availableIngredients])

  const skillMetrics = useMemo(
    () => SKILL_METRICS.filter((m) => availableSet.has(m)),
    [availableSet],
  )
  const errorMetrics = useMemo(
    () => ERROR_METRICS.filter((m) => availableSet.has(m)),
    [availableSet],
  )
  const volumeMetrics = useMemo(
    () => VOLUME_METRICS.filter((m) => availableSet.has(m)),
    [availableSet],
  )

  const activeSkill = Object.values(config.skill).filter((v) => v != null).length
  const activeError = Object.values(config.error).filter((v) => v != null).length
  const activeVolume = Object.values(config.volume).filter((v) => v != null).length

  const totalIngredients = INGREDIENT_CATEGORIES.reduce(
    (acc, cat) => acc + availableIngredients[cat.key].length,
    0,
  )
  const activeIngredients = INGREDIENT_CATEGORIES.reduce(
    (acc, cat) => acc + config.ingredients[cat.key].size,
    0,
  )
  const ingredientFilterActive = activeIngredients < totalIngredients

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-gray-400" />
          <h3 className="text-sm font-semibold text-gray-700">Experiment Filters</h3>
          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-primary-50 text-primary-700">
            {passingKeys.size} / {rows.length} pass
          </span>
        </div>
        <button
          onClick={handleReset}
          className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset All
        </button>
      </div>

      {/* Ingredient filters row */}
      {totalIngredients > 0 && (
        <div className="px-4 py-3 border-b border-gray-100 bg-gray-50/50">
          <div className="flex items-center gap-2 mb-2">
            <Beaker className="w-3.5 h-3.5 text-gray-400" />
            <span className="text-xs font-semibold text-gray-600">Experiment Ingredients</span>
            {ingredientFilterActive && (
              <span className="text-xs font-medium px-1.5 py-0.5 rounded bg-amber-50 text-amber-700">
                Filtered
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {INGREDIENT_CATEGORIES.map((cat) => (
              <IngredientCheckboxGroup
                key={cat.key}
                label={cat.label}
                options={availableIngredients[cat.key]}
                selected={config.ingredients[cat.key]}
                onToggle={(value) => {
                  const next = new Set(config.ingredients[cat.key])
                  if (next.has(value)) next.delete(value)
                  else next.add(value)
                  handleIngredientChange(cat.key, next)
                }}
                onToggleAll={() => {
                  const current = config.ingredients[cat.key]
                  const all = availableIngredients[cat.key]
                  const allSelected = all.every((o) => current.has(o))
                  handleIngredientChange(cat.key, allSelected ? new Set() : new Set(all))
                }}
              />
            ))}
          </div>
        </div>
      )}

      {/* Metric threshold filters */}
      <div className="p-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        {skillMetrics.length > 0 && (
          <CollapsibleGroup
            title="Skill Metrics (higher = better)"
            activeCount={activeSkill}
            totalCount={skillMetrics.length}
          >
            {skillMetrics.map((m) => (
              <SkillSlider
                key={m}
                metric={m}
                value={config.skill[m] ?? null}
                onChange={handleSkillChange}
              />
            ))}
          </CollapsibleGroup>
        )}

        {errorMetrics.length > 0 && (
          <CollapsibleGroup
            title="Error Metrics (lower = better)"
            activeCount={activeError}
            totalCount={errorMetrics.length}
          >
            {errorMetrics.map((m) => (
              <ErrorQuantileSelector
                key={m}
                metric={m}
                value={config.error[m] ?? null}
                resolvedValue={
                  config.error[m] != null
                    ? resolvedQuantiles[m]?.[config.error[m]!]
                    : undefined
                }
                onChange={handleErrorChange}
              />
            ))}
          </CollapsibleGroup>
        )}

        {volumeMetrics.length > 0 && (
          <CollapsibleGroup
            title="Volume / Signatures (± %)"
            activeCount={activeVolume}
            totalCount={volumeMetrics.length}
          >
            {volumeMetrics.map((m) => (
              <VolumeSlider
                key={m}
                metric={m}
                value={config.volume[m] ?? null}
                onChange={handleVolumeChange}
              />
            ))}
          </CollapsibleGroup>
        )}
      </div>
    </div>
  )
}
