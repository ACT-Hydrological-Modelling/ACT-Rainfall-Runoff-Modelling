import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Activity, BookOpen } from 'lucide-react'
import { getComparisonSignatures, getSignatureReference } from '../../services/analysisApi'
import PlotlyChart from './PlotlyChart'
import SignatureInfoPopover from './SignatureInfoPopover'
import SignatureReferenceModal from './SignatureReferenceModal'
import type { SignatureComparisonResponse, SignatureInfo } from '../../types/analysis'

const CATEGORY_LABELS: Record<string, string> = {
  'Magnitude': 'Magnitude',
  'Variability': 'Variability',
  'Timing': 'Timing',
  'Flow Duration Curve': 'FDC',
  'Frequency': 'Frequency',
  'Recession': 'Recession',
  'Baseflow': 'Baseflow',
  'Event': 'Event',
  'Seasonality': 'Seasonality',
}

interface SignatureComparisonChartsProps {
  sessionId: string
  gaugeId: string
  experimentKeys: string[]
}

export default function SignatureComparisonCharts({
  sessionId,
  gaugeId,
  experimentKeys,
}: SignatureComparisonChartsProps) {
  const [activeCategory, setActiveCategory] = useState<string>('Magnitude')
  const [referenceModalOpen, setReferenceModalOpen] = useState(false)

  const { data, isLoading, error } = useQuery<SignatureComparisonResponse>({
    queryKey: ['signature-comparison', sessionId, gaugeId, experimentKeys],
    queryFn: () => getComparisonSignatures(sessionId, gaugeId, experimentKeys),
    enabled: experimentKeys.length > 0,
  })

  const { data: referenceData } = useQuery({
    queryKey: ['signature-reference'],
    queryFn: getSignatureReference,
    staleTime: Infinity,
  })

  const getSignatureInfo = (sigId: string): SignatureInfo | undefined => {
    if (!referenceData) return undefined
    for (const sigs of Object.values(referenceData.categories)) {
      const found = sigs.find((s) => s.id === sigId)
      if (found) return found
    }
    return undefined
  }

  if (experimentKeys.length === 0) {
    return null
  }

  if (isLoading) {
    return (
      <div className="mt-6 border border-gray-100 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary-500" />
          Hydrologic Signatures
        </h4>
        <div className="text-center py-8 text-gray-400 text-sm">
          Computing signatures...
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="mt-6 border border-gray-100 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary-500" />
          Hydrologic Signatures
        </h4>
        <div className="text-center py-8 text-red-500 text-sm">
          Failed to load signature comparison
        </div>
      </div>
    )
  }

  const categories = Object.keys(data.categories)

  return (
    <div className="mt-6 border border-gray-100 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary-500" />
          Hydrologic Signatures
        </h4>
        <button
          onClick={() => setReferenceModalOpen(true)}
          className="flex items-center gap-1.5 px-2 py-1 text-xs text-gray-600 hover:text-primary-700 hover:bg-primary-50 rounded transition-colors"
        >
          <BookOpen className="w-3.5 h-3.5" />
          View Reference
        </button>
      </div>

      {/* Signature reference modal */}
      <SignatureReferenceModal
        isOpen={referenceModalOpen}
        onClose={() => setReferenceModalOpen(false)}
        initialCategory={activeCategory}
      />

      {/* Category tabs */}
      <div className="flex flex-wrap gap-1 mb-4 p-1 bg-gray-50 rounded-lg">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setActiveCategory(cat)}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
              activeCategory === cat
                ? 'bg-white text-primary-700 shadow-sm'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
            }`}
          >
            {CATEGORY_LABELS[cat] || cat}
          </button>
        ))}
      </div>

      {/* Category bar chart */}
      <div className="mb-6">
        <PlotlyChart
          figure={data.bar_figures[activeCategory]}
          loading={false}
          autoHeight
        />
      </div>

      {/* Signature values table */}
      <div className="mb-6">
        <SignatureValuesTable
          categoryData={data.categories[activeCategory]}
          category={activeCategory}
          getSignatureInfo={getSignatureInfo}
        />
      </div>

      {/* Overall heatmap - centered */}
      <div className="mt-8 flex flex-col items-center w-full">
        <h5 className="text-xs font-semibold text-gray-600 mb-2 text-center">
          Signature Percent Errors (all categories)
        </h5>
        <PlotlyChart
          figure={data.heatmap_figure}
          loading={false}
          className="min-h-[300px]"
        />
      </div>
    </div>
  )
}

interface SignatureValuesTableProps {
  categoryData: SignatureComparisonResponse['categories'][string]
  category: string
  getSignatureInfo: (sigId: string) => SignatureInfo | undefined
}

function SignatureValuesTable({
  categoryData,
  category: _category,
  getSignatureInfo,
}: SignatureValuesTableProps) {
  const { signatures, observed, experiments, percent_errors } = categoryData
  const experimentKeys = Object.keys(experiments)

  if (experimentKeys.length === 0) {
    return null
  }

  const formatValue = (val: number | null) => {
    if (val === null || val === undefined) return '—'
    if (Math.abs(val) < 0.01) return val.toExponential(2)
    if (Math.abs(val) > 10000) return val.toExponential(2)
    return val.toFixed(3)
  }

  const formatError = (val: number | null) => {
    if (val === null || val === undefined) return '—'
    const sign = val >= 0 ? '+' : ''
    return `${sign}${val.toFixed(1)}%`
  }

  const getErrorClass = (val: number | null) => {
    if (val === null || val === undefined) return ''
    const absVal = Math.abs(val)
    if (absVal < 10) return 'text-green-600'
    if (absVal < 25) return 'text-yellow-600'
    return 'text-red-600'
  }

  const formatExperimentLabel = (key: string): string => {
    const parts = key.split('__').slice(1)
    if (parts.length === 0) return key
    return parts.join(' / ')
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-xs table-fixed">
        <thead>
          <tr className="border-b border-gray-200">
            <th className="text-left py-2 px-2 font-semibold text-gray-600 w-32">
              Signature
            </th>
            <th className="text-right py-2 px-2 font-semibold text-gray-600 w-24">
              Observed
            </th>
            {experimentKeys.map((key) => {
              const label = formatExperimentLabel(key)
              return (
                <th
                  key={key}
                  className="text-right py-2 px-2 font-semibold text-gray-600 min-w-[120px] max-w-[180px] whitespace-normal break-words"
                  title={key}
                >
                  {label}
                </th>
              )
            })}
          </tr>
        </thead>
        <tbody>
          {signatures.map((sig) => (
            <tr key={sig} className="border-b border-gray-100 hover:bg-gray-50">
              <td className="py-1.5 px-2 text-gray-700">
                <span className="inline-flex items-center gap-1.5">
                  <span className="font-mono">{sig}</span>
                  <SignatureInfoPopover
                    signatureId={sig}
                    signatureInfo={getSignatureInfo(sig)}
                  />
                </span>
              </td>
              <td className="py-1.5 px-2 text-right font-mono text-gray-900">
                {formatValue(observed[sig])}
              </td>
              {experimentKeys.map((expKey) => {
                const simVal = experiments[expKey]?.[sig]
                const errVal = percent_errors[expKey]?.[sig]
                return (
                  <td key={expKey} className="py-1.5 px-2 text-right font-mono">
                    <span className="text-gray-700">{formatValue(simVal)}</span>
                    <span className={`ml-1 ${getErrorClass(errVal)}`}>
                      ({formatError(errVal)})
                    </span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
