import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useParams, Link, useNavigate } from 'react-router-dom'
import {
  ArrowRight,
  FlaskConical,
  AlertTriangle,
  Table2,
  LayoutGrid,
  MapPin,
  Droplets,
  Sun,
  Ruler,
  Calendar,
} from 'lucide-react'
import { getSession, getSessionSummary } from '../../services/analysisApi'
import type { GaugeSummary, SummaryTableRow } from '../../types/analysis'
import CatchmentMap from '../../components/analysis/CatchmentMap'
import GeoLayerPanel from '../../components/analysis/GeoLayerPanel'

function fmt(v: number | null | undefined, decimals = 1, suffix = ''): string {
  if (v == null) return '—'
  return `${v.toFixed(decimals)}${suffix}`
}

export default function SessionOverview() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const navigate = useNavigate()
  const [view, setView] = useState<'cards' | 'table'>('cards')

  const { data: session, isLoading } = useQuery({
    queryKey: ['analysis-session', sessionId],
    queryFn: () => getSession(sessionId!),
    enabled: !!sessionId,
  })

  const { data: summary } = useQuery({
    queryKey: ['analysis-summary', sessionId],
    queryFn: () => getSessionSummary(sessionId!),
    enabled: !!sessionId && view === 'table',
  })

  if (isLoading) {
    return <div className="text-center py-12 text-gray-500">Loading session…</div>
  }

  if (!session) {
    return <div className="text-center py-12 text-red-500">Session not found</div>
  }

  return (
    <div>
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">{session.name}</h1>
        <div className="flex gap-6 mt-2 text-sm text-gray-500">
          <span>{session.gauge_ids.length} gauges</span>
          <span>{session.total_experiments} experiments</span>
          {session.total_failures > 0 && (
            <span className="text-amber-600">{session.total_failures} failures</span>
          )}
        </div>
      </div>

      {/* Map + layer panel row */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-6">
        <div className="lg:col-span-3">
          <CatchmentMap
            onGaugeClick={(gid) =>
              navigate(`/analysis/sessions/${sessionId}/gauges/${gid}`)
            }
            className="h-[900px]"
          />
        </div>
        <div className="lg:col-span-1">
          <GeoLayerPanel />
        </div>
      </div>

      {/* View toggle */}
      <div className="flex items-center gap-2 mb-5">
        <button
          onClick={() => setView('cards')}
          className={`inline-flex items-center px-3 py-1.5 rounded-lg text-sm font-medium ${
            view === 'cards'
              ? 'bg-primary-50 text-primary-700'
              : 'text-gray-500 hover:bg-gray-100'
          }`}
        >
          <LayoutGrid className="w-4 h-4 mr-1.5" /> Gauges
        </button>
        <button
          onClick={() => setView('table')}
          className={`inline-flex items-center px-3 py-1.5 rounded-lg text-sm font-medium ${
            view === 'table'
              ? 'bg-primary-50 text-primary-700'
              : 'text-gray-500 hover:bg-gray-100'
          }`}
        >
          <Table2 className="w-4 h-4 mr-1.5" /> Summary Table
        </button>
      </div>

      {/* Gauge cards */}
      {view === 'cards' && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {session.gauges.map((g) => (
            <GaugeCard
              key={g.gauge_id}
              gauge={g}
              sessionId={sessionId!}
            />
          ))}
        </div>
      )}

      {/* Summary table */}
      {view === 'table' && <SummaryTable data={summary} />}
    </div>
  )
}

// ── Gauge Card ──────────────────────────────────────────────────────────────

function GaugeCard({
  gauge,
  sessionId,
}: {
  gauge: GaugeSummary
  sessionId: string
}) {
  const m = gauge.metadata

  return (
    <Link
      to={`/analysis/sessions/${sessionId}/gauges/${gauge.gauge_id}`}
      className="bg-white rounded-lg border border-gray-200 p-5 hover:border-primary-300 hover:shadow-md transition-all group"
    >
      {/* Title row */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-lg font-bold text-gray-900">{gauge.gauge_id}</h3>
          {m?.gauge_name && (
            <p className="text-xs text-gray-400">{m.gauge_name}</p>
          )}
        </div>
        <ArrowRight className="w-4 h-4 text-gray-300 group-hover:text-primary-500 transition-colors" />
      </div>

      {/* Experiment counts */}
      <div className="flex gap-4 text-sm text-gray-500 mb-3">
        <span className="flex items-center gap-1">
          <FlaskConical className="w-3.5 h-3.5" /> {gauge.n_experiments} experiments
        </span>
        {gauge.n_failures > 0 && (
          <span className="flex items-center gap-1 text-amber-600">
            <AlertTriangle className="w-3.5 h-3.5" /> {gauge.n_failures}
          </span>
        )}
      </div>

      {/* Metadata grid */}
      {m && (
        <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs border-t border-gray-100 pt-3">
          <MetaItem
            icon={<Ruler className="w-3 h-3" />}
            label="Area"
            value={fmt(m.area_km2, 1, ' km²')}
          />
          <MetaItem
            icon={<Calendar className="w-3 h-3" />}
            label="Record"
            value={
              m.record_start && m.record_end
                ? `${m.record_start.slice(0, 4)}–${m.record_end.slice(0, 4)}`
                : '—'
            }
            sub={m.record_years != null ? `Length: ${m.record_years.toFixed(1)} yrs` : undefined}
          />
          <MetaItem
            icon={<Droplets className="w-3 h-3" />}
            label="Mean P"
            value={fmt(m.mean_precip_mm_day, 2, ' mm/d')}
            sub={m.total_precip_mm_yr != null ? `Annual: ${m.total_precip_mm_yr.toFixed(0)} mm/yr` : undefined}
          />
          <MetaItem
            icon={<Sun className="w-3 h-3" />}
            label="Mean PET"
            value={fmt(m.mean_pet_mm_day, 2, ' mm/d')}
            sub={m.total_pet_mm_yr != null ? `Annual: ${m.total_pet_mm_yr.toFixed(0)} mm/yr` : undefined}
          />
          <MetaItem
            icon={<MapPin className="w-3 h-3" />}
            label="Mean Q"
            value={fmt(m.mean_flow, 2, ' m³/s')}
            sub={m.median_flow != null ? `Median: ${m.median_flow.toFixed(2)} m³/s` : undefined}
          />
          <MetaItem
            label="Aridity Index"
            value={fmt(m.aridity_index, 2)}
          />
          <MetaItem
            label="Runoff Ratio"
            value={fmt(m.runoff_ratio, 2)}
          />
        </div>
      )}
    </Link>
  )
}

function MetaItem({
  icon,
  label,
  value,
  sub,
}: {
  icon?: React.ReactNode
  label: string
  value: string
  sub?: string
}) {
  return (
    <div className="flex items-start gap-1.5">
      {icon && <span className="text-gray-400 mt-px">{icon}</span>}
      <div>
        <span className="text-gray-400">{label}:</span>{' '}
        <span className="font-medium text-gray-700">{value}</span>
        {sub && <span className="block text-gray-400">{sub}</span>}
      </div>
    </div>
  )
}

// ── Summary Table ───────────────────────────────────────────────────────────

function SummaryTable({ data }: { data: SummaryTableRow[] | undefined }) {
  const [sortField, setSortField] = useState<keyof SummaryTableRow>('gauge_id')
  const [sortAsc, setSortAsc] = useState(true)

  if (!data) {
    return <div className="text-center py-12 text-gray-400">Loading summary…</div>
  }

  const sorted = [...data].sort((a, b) => {
    const va = a[sortField]
    const vb = b[sortField]
    if (va == null && vb == null) return 0
    if (va == null) return 1
    if (vb == null) return -1
    if (typeof va === 'string' && typeof vb === 'string') {
      return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va)
    }
    return sortAsc ? Number(va) - Number(vb) : Number(vb) - Number(va)
  })

  const handleSort = (field: keyof SummaryTableRow) => {
    if (sortField === field) {
      setSortAsc(!sortAsc)
    } else {
      setSortField(field)
      setSortAsc(true)
    }
  }

  const cols: { key: keyof SummaryTableRow; label: string }[] = [
    { key: 'gauge_id', label: 'Gauge' },
    { key: 'model', label: 'Model' },
    { key: 'objective', label: 'Objective' },
    { key: 'transformation', label: 'Transform' },
    { key: 'algorithm', label: 'Algorithm' },
    { key: 'best_objective', label: 'Best Obj.' },
    { key: 'runtime_seconds', label: 'Runtime (s)' },
    { key: 'success', label: 'OK' },
  ]

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              {cols.map((c) => (
                <th
                  key={c.key}
                  className="px-4 py-2 text-left font-medium text-gray-500 cursor-pointer hover:bg-gray-100 whitespace-nowrap"
                  onClick={() => handleSort(c.key)}
                >
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sorted.map((row, i) => (
              <tr key={i} className="hover:bg-gray-50">
                <td className="px-4 py-1.5 font-medium text-gray-900">{row.gauge_id}</td>
                <td className="px-4 py-1.5 text-gray-600">{row.model}</td>
                <td className="px-4 py-1.5 text-gray-600">{row.objective}</td>
                <td className="px-4 py-1.5 text-gray-600">{row.transformation || '—'}</td>
                <td className="px-4 py-1.5 text-gray-600">{row.algorithm}</td>
                <td className="px-4 py-1.5 font-mono text-gray-700">
                  {row.best_objective != null ? row.best_objective.toFixed(4) : '—'}
                </td>
                <td className="px-4 py-1.5 font-mono text-gray-500">
                  {row.runtime_seconds != null ? row.runtime_seconds.toFixed(1) : '—'}
                </td>
                <td className="px-4 py-1.5">
                  {row.success ? (
                    <span className="inline-block w-2 h-2 rounded-full bg-green-500" />
                  ) : (
                    <span className="inline-block w-2 h-2 rounded-full bg-red-500" />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
