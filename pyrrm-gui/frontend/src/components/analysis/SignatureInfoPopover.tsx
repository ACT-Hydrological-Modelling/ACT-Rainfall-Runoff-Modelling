import { useState, useRef, useEffect, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { HelpCircle, X } from 'lucide-react'
import type { SignatureInfo } from '../../types/analysis'

interface SignatureInfoPopoverProps {
  signatureId: string
  signatureInfo: SignatureInfo | undefined
}

export default function SignatureInfoPopover({
  signatureId,
  signatureInfo,
}: SignatureInfoPopoverProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [popoverStyle, setPopoverStyle] = useState<React.CSSProperties>({})
  const popoverRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)

  const updatePosition = useCallback(() => {
    if (buttonRef.current && isOpen) {
      const rect = buttonRef.current.getBoundingClientRect()
      const viewportHeight = window.innerHeight
      const viewportWidth = window.innerWidth
      const popoverHeight = 380
      const popoverWidth = 320
      
      let top: number
      let left: number
      
      // Vertical positioning: prefer below, use above if not enough space
      if (viewportHeight - rect.bottom >= popoverHeight) {
        top = rect.bottom + 4
      } else if (rect.top >= popoverHeight) {
        top = rect.top - popoverHeight - 4
      } else {
        top = Math.max(8, viewportHeight - popoverHeight - 8)
      }
      
      // Horizontal positioning: prefer right of button, use left if not enough space
      if (viewportWidth - rect.right >= popoverWidth + 8) {
        left = rect.right + 8
      } else if (rect.left >= popoverWidth + 8) {
        left = rect.left - popoverWidth - 8
      } else {
        left = Math.max(8, (viewportWidth - popoverWidth) / 2)
      }
      
      setPopoverStyle({
        position: 'fixed',
        top: `${top}px`,
        left: `${left}px`,
        zIndex: 9999,
      })
    }
  }, [isOpen])

  useEffect(() => {
    if (isOpen) {
      updatePosition()
      window.addEventListener('scroll', updatePosition, true)
      window.addEventListener('resize', updatePosition)
      return () => {
        window.removeEventListener('scroll', updatePosition, true)
        window.removeEventListener('resize', updatePosition)
      }
    }
  }, [isOpen, updatePosition])

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        popoverRef.current &&
        !popoverRef.current.contains(event.target as Node) &&
        buttonRef.current &&
        !buttonRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  if (!signatureInfo) {
    return (
      <span
        className="inline-flex items-center text-gray-400 cursor-help"
        title={`No documentation available for ${signatureId}`}
      >
        <HelpCircle className="w-3 h-3" />
      </span>
    )
  }

  const popoverContent = isOpen && (
    <div
      ref={popoverRef}
      style={popoverStyle}
      className="w-80 bg-white border border-gray-200 rounded-lg shadow-xl"
    >
      <div className="p-3 max-h-[70vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-start justify-between mb-2">
          <div>
            <h4 className="font-semibold text-sm text-gray-900">
              {signatureInfo.name}
            </h4>
            <p className="text-xs text-gray-500 font-mono">{signatureId}</p>
          </div>
          <button
            onClick={() => setIsOpen(false)}
            className="text-gray-400 hover:text-gray-600 p-0.5"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Category & Units */}
        <div className="flex gap-2 mb-2">
          <span className="inline-block px-1.5 py-0.5 bg-primary-50 text-primary-700 text-xs rounded">
            {signatureInfo.category}
          </span>
          {signatureInfo.units && (
            <span className="inline-block px-1.5 py-0.5 bg-gray-100 text-gray-600 text-xs rounded">
              {signatureInfo.units}
            </span>
          )}
        </div>

        {/* Description */}
        <p className="text-xs text-gray-700 mb-2">
          {signatureInfo.description}
        </p>

        {/* Formula */}
        {signatureInfo.formula && (
          <div className="mb-2">
            <p className="text-xs font-medium text-gray-500 mb-0.5">Formula</p>
            <p className="text-xs font-mono bg-gray-50 px-2 py-1 rounded text-gray-800">
              {signatureInfo.formula}
            </p>
          </div>
        )}

        {/* Interpretation */}
        {signatureInfo.interpretation && (
          <div className="mb-2">
            <p className="text-xs font-medium text-gray-500 mb-0.5">
              Interpretation
            </p>
            <p className="text-xs text-gray-600 leading-relaxed">
              {signatureInfo.interpretation}
            </p>
          </div>
        )}

        {/* Related signatures */}
        {signatureInfo.related && signatureInfo.related.length > 0 && (
          <div className="mb-2">
            <p className="text-xs font-medium text-gray-500 mb-0.5">Related</p>
            <div className="flex flex-wrap gap-1">
              {signatureInfo.related.map((rel) => (
                <span
                  key={rel}
                  className="inline-block px-1.5 py-0.5 bg-gray-100 text-gray-600 text-xs font-mono rounded"
                >
                  {rel}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* References */}
        {signatureInfo.references && signatureInfo.references.length > 0 && (
          <div>
            <p className="text-xs font-medium text-gray-500 mb-0.5">
              References
            </p>
            <p className="text-xs text-gray-500 italic">
              {signatureInfo.references.join('; ')}
            </p>
          </div>
        )}
      </div>
    </div>
  )

  return (
    <span className="inline-flex items-center">
      <button
        ref={buttonRef}
        onClick={() => setIsOpen(!isOpen)}
        className="text-gray-400 hover:text-primary-600 transition-colors p-0.5 -m-0.5 rounded"
        title={`Click for info about ${signatureInfo.name}`}
      >
        <HelpCircle className="w-3 h-3" />
      </button>

      {popoverContent && createPortal(popoverContent, document.body)}
    </span>
  )
}
