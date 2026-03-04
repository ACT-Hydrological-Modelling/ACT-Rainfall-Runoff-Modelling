import { useRef, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Upload,
  Trash2,
  Layers,
  CheckCircle2,
  XCircle,
  Loader2,
  MapPin,
  Hexagon,
} from 'lucide-react'
import {
  listGeoLayers,
  uploadGeoLayer,
  deleteGeoLayer,
} from '../../services/analysisApi'

const LAYER_META: Record<string, { label: string; description: string; icon: typeof MapPin }> = {
  catchments: {
    label: 'Catchment Boundaries',
    description: 'Polygons defining catchment areas',
    icon: Hexagon,
  },
  gauges: {
    label: 'Gauge Locations',
    description: 'Point features with gauge_id property',
    icon: MapPin,
  },
}

export default function GeoLayerPanel() {
  const [expanded, setExpanded] = useState(false)

  const { data: layersData, isLoading } = useQuery({
    queryKey: ['geo-layers'],
    queryFn: listGeoLayers,
    staleTime: 30_000,
  })

  const layers = layersData?.layers ?? []

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 rounded-lg"
      >
        <span className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-gray-400" />
          Map Layers
        </span>
        <span className="text-xs text-gray-400">
          {layers.filter((l) => l.available).length}/{layers.length} loaded
        </span>
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-3 border-t border-gray-100 pt-3">
          {isLoading ? (
            <div className="text-sm text-gray-400 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading…
            </div>
          ) : (
            layers.map((layer) => (
              <LayerRow
                key={layer.name}
                name={layer.name}
                available={layer.available}
                nFeatures={layer.n_features}
              />
            ))
          )}
        </div>
      )}
    </div>
  )
}

function LayerRow({
  name,
  available,
  nFeatures,
}: {
  name: string
  available: boolean
  nFeatures: number
}) {
  const queryClient = useQueryClient()
  const fileRef = useRef<HTMLInputElement>(null)
  const [error, setError] = useState<string | null>(null)

  const meta = LAYER_META[name] || {
    label: name,
    description: '',
    icon: Layers,
  }
  const Icon = meta.icon

  const uploadMut = useMutation({
    mutationFn: (files: File[]) => uploadGeoLayer(name, files),
    onSuccess: () => {
      setError(null)
      queryClient.invalidateQueries({ queryKey: ['geo-layers'] })
      queryClient.invalidateQueries({ queryKey: ['geo-layer', name] })
    },
    onError: (err: any) => {
      setError(
        err?.response?.data?.detail || err.message || 'Upload failed'
      )
    },
  })

  const deleteMut = useMutation({
    mutationFn: () => deleteGeoLayer(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['geo-layers'] })
      queryClient.invalidateQueries({ queryKey: ['geo-layer', name] })
    },
  })

  const handleFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length > 0) uploadMut.mutate(files)
    e.target.value = ''
  }

  return (
    <div className="flex items-start gap-3 text-sm">
      <Icon className="w-4 h-4 mt-0.5 text-gray-400 flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-700">{meta.label}</span>
          {available ? (
            <span className="flex items-center gap-1 text-xs text-emerald-600">
              <CheckCircle2 className="w-3 h-3" /> {nFeatures} features
            </span>
          ) : (
            <span className="flex items-center gap-1 text-xs text-gray-400">
              <XCircle className="w-3 h-3" /> Not loaded
            </span>
          )}
        </div>
        <p className="text-xs text-gray-400">{meta.description}</p>
        {error && (
          <p className="text-xs text-red-600 mt-1">{error}</p>
        )}
      </div>
      <div className="flex items-center gap-1 flex-shrink-0">
        <input
          ref={fileRef}
          type="file"
          multiple
          accept=".shp,.dbf,.shx,.prj,.cpg,.geojson"
          className="hidden"
          onChange={handleFiles}
        />
        <button
          onClick={() => fileRef.current?.click()}
          disabled={uploadMut.isPending}
          className="p-1.5 text-gray-400 hover:text-primary-600 hover:bg-primary-50 rounded"
          title="Upload shapefile or GeoJSON"
        >
          {uploadMut.isPending ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Upload className="w-4 h-4" />
          )}
        </button>
        {available && (
          <button
            onClick={() => deleteMut.mutate()}
            className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded"
            title="Remove layer"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  )
}
