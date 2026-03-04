import { useState, useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { X, Search, ChevronDown, ChevronRight, BookOpen } from 'lucide-react'
import { getSignatureReference } from '../../services/analysisApi'
import type { SignatureInfo } from '../../types/analysis'

interface SignatureReferenceModalProps {
  isOpen: boolean
  onClose: () => void
  initialCategory?: string
  initialSignature?: string
}

export default function SignatureReferenceModal({
  isOpen,
  onClose,
  initialCategory,
  initialSignature,
}: SignatureReferenceModalProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set()
  )
  const [selectedSignature, setSelectedSignature] = useState<SignatureInfo | null>(
    null
  )
  const modalRef = useRef<HTMLDivElement>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['signature-reference'],
    queryFn: getSignatureReference,
    enabled: isOpen,
    staleTime: Infinity,
  })

  useEffect(() => {
    if (isOpen && data) {
      if (initialCategory) {
        setExpandedCategories(new Set([initialCategory]))
      } else {
        setExpandedCategories(new Set(data.category_order))
      }

      if (initialSignature && data.categories) {
        for (const sigs of Object.values(data.categories)) {
          const found = sigs.find((s) => s.id === initialSignature)
          if (found) {
            setSelectedSignature(found)
            break
          }
        }
      }
    }
  }, [isOpen, data, initialCategory, initialSignature])

  useEffect(() => {
    function handleEscape(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      return () => document.removeEventListener('keydown', handleEscape)
    }
  }, [isOpen, onClose])

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
        onClose()
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen, onClose])

  if (!isOpen) return null

  const toggleCategory = (cat: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev)
      if (next.has(cat)) {
        next.delete(cat)
      } else {
        next.add(cat)
      }
      return next
    })
  }

  const filterSignatures = (sigs: SignatureInfo[]): SignatureInfo[] => {
    if (!searchQuery.trim()) return sigs
    const q = searchQuery.toLowerCase()
    return sigs.filter(
      (s) =>
        s.id.toLowerCase().includes(q) ||
        s.name.toLowerCase().includes(q) ||
        s.description.toLowerCase().includes(q) ||
        s.interpretation.toLowerCase().includes(q)
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div
        ref={modalRef}
        className="bg-white rounded-xl shadow-2xl w-[90vw] max-w-5xl h-[85vh] flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <BookOpen className="w-5 h-5 text-primary-600" />
            <h2 className="text-lg font-semibold text-gray-900">
              Hydrologic Signature Reference
            </h2>
            {data && (
              <span className="text-sm text-gray-500">
                ({data.total_signatures} signatures)
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 p-1 rounded-lg hover:bg-gray-100"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search */}
        <div className="px-6 py-3 border-b border-gray-100">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search signatures by name, ID, or description..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Left panel - categories & signatures list */}
          <div className="w-1/3 border-r border-gray-200 overflow-y-auto">
            {isLoading && (
              <div className="p-6 text-center text-gray-500 text-sm">
                Loading signature reference...
              </div>
            )}
            {error && (
              <div className="p-6 text-center text-red-500 text-sm">
                Failed to load signature reference
              </div>
            )}
            {data && (
              <div className="py-2">
                {data.category_order.map((cat) => {
                  const sigs = data.categories[cat] || []
                  const filtered = filterSignatures(sigs)
                  if (searchQuery && filtered.length === 0) return null

                  return (
                    <div key={cat} className="mb-1">
                      <button
                        onClick={() => toggleCategory(cat)}
                        className="w-full flex items-center justify-between px-4 py-2 text-left hover:bg-gray-50"
                      >
                        <span className="font-medium text-sm text-gray-700">
                          {cat}
                        </span>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-gray-400">
                            {filtered.length}
                          </span>
                          {expandedCategories.has(cat) ? (
                            <ChevronDown className="w-4 h-4 text-gray-400" />
                          ) : (
                            <ChevronRight className="w-4 h-4 text-gray-400" />
                          )}
                        </div>
                      </button>

                      {expandedCategories.has(cat) && (
                        <div className="ml-4 border-l border-gray-100">
                          {filtered.map((sig) => (
                            <button
                              key={sig.id}
                              onClick={() => setSelectedSignature(sig)}
                              className={`w-full text-left px-3 py-1.5 text-sm hover:bg-gray-50 ${
                                selectedSignature?.id === sig.id
                                  ? 'bg-primary-50 text-primary-700 border-l-2 border-primary-500 -ml-px'
                                  : 'text-gray-600'
                              }`}
                            >
                              <span className="font-mono text-xs">{sig.id}</span>
                              <span className="ml-2 text-gray-400">
                                {sig.name}
                              </span>
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* Right panel - signature detail */}
          <div className="flex-1 overflow-y-auto p-6">
            {!selectedSignature ? (
              <div className="h-full flex items-center justify-center text-gray-400 text-sm">
                Select a signature from the list to view details
              </div>
            ) : (
              <SignatureDetailView signature={selectedSignature} />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function SignatureDetailView({ signature }: { signature: SignatureInfo }) {
  return (
    <div>
      {/* Title */}
      <div className="mb-6">
        <h3 className="text-xl font-semibold text-gray-900">{signature.name}</h3>
        <p className="text-sm font-mono text-gray-500">{signature.id}</p>
      </div>

      {/* Category & Units */}
      <div className="flex gap-3 mb-6">
        <div>
          <p className="text-xs font-medium text-gray-500 mb-1">Category</p>
          <span className="inline-block px-2 py-1 bg-primary-50 text-primary-700 text-sm rounded">
            {signature.category}
          </span>
        </div>
        {signature.units && (
          <div>
            <p className="text-xs font-medium text-gray-500 mb-1">Units</p>
            <span className="inline-block px-2 py-1 bg-gray-100 text-gray-700 text-sm rounded">
              {signature.units}
            </span>
          </div>
        )}
        {signature.range && (
          <div>
            <p className="text-xs font-medium text-gray-500 mb-1">Range</p>
            <span className="inline-block px-2 py-1 bg-gray-100 text-gray-700 text-sm font-mono rounded">
              [{signature.range[0]}, {signature.range[1]}]
            </span>
          </div>
        )}
      </div>

      {/* Description */}
      <div className="mb-6">
        <p className="text-xs font-medium text-gray-500 mb-2">Description</p>
        <p className="text-sm text-gray-700 leading-relaxed">
          {signature.description}
        </p>
      </div>

      {/* Formula */}
      {signature.formula && (
        <div className="mb-6">
          <p className="text-xs font-medium text-gray-500 mb-2">Formula</p>
          <div className="bg-gray-50 border border-gray-200 rounded-lg px-4 py-3">
            <code className="text-sm font-mono text-gray-800">
              {signature.formula}
            </code>
          </div>
        </div>
      )}

      {/* Interpretation */}
      {signature.interpretation && (
        <div className="mb-6">
          <p className="text-xs font-medium text-gray-500 mb-2">Interpretation</p>
          <p className="text-sm text-gray-700 leading-relaxed">
            {signature.interpretation}
          </p>
        </div>
      )}

      {/* Related signatures */}
      {signature.related && signature.related.length > 0 && (
        <div className="mb-6">
          <p className="text-xs font-medium text-gray-500 mb-2">
            Related Signatures
          </p>
          <div className="flex flex-wrap gap-2">
            {signature.related.map((rel) => (
              <span
                key={rel}
                className="inline-block px-2 py-1 bg-gray-100 text-gray-700 text-sm font-mono rounded"
              >
                {rel}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* References */}
      {signature.references && signature.references.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-500 mb-2">References</p>
          <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
            {signature.references.map((ref, idx) => (
              <li key={idx} className="italic">
                {ref}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
