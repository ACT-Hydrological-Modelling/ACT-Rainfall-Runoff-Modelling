import { useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { MapContainer, TileLayer, GeoJSON, Marker, Popup, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { getGeoLayerGeoJSON, listGeoLayers } from '../../services/analysisApi'

const ACT_CENTER: [number, number] = [-35.45, 149.1]
const DEFAULT_ZOOM = 10

const gaugeIcon = new L.DivIcon({
  html: `<div style="
    width: 12px; height: 12px;
    background: #2563eb;
    border: 2px solid white;
    border-radius: 50%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.4);
  "></div>`,
  className: '',
  iconSize: [12, 12],
  iconAnchor: [6, 6],
})

const catchmentStyle: L.PathOptions = {
  color: '#6366f1',
  weight: 2,
  fillColor: '#818cf8',
  fillOpacity: 0.12,
  dashArray: '4 4',
}

function FitBoundsToLayers({ geojson }: { geojson: any | null }) {
  const map = useMap()
  useEffect(() => {
    if (!geojson) return
    try {
      const layer = L.geoJSON(geojson)
      const bounds = layer.getBounds()
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [30, 30], maxZoom: 13 })
      }
    } catch {
      // ignore invalid geojson
    }
  }, [geojson, map])
  return null
}

interface CatchmentMapProps {
  onGaugeClick?: (gaugeId: string) => void
  className?: string
}

export default function CatchmentMap({
  onGaugeClick,
  className = '',
}: CatchmentMapProps) {
  const mapRef = useRef<L.Map | null>(null)

  const { data: layersInfo } = useQuery({
    queryKey: ['geo-layers'],
    queryFn: listGeoLayers,
    staleTime: 60_000,
  })

  const catchmentsAvailable = layersInfo?.layers.find(
    (l) => l.name === 'catchments' && l.available
  )
  const gaugesLayerAvailable = layersInfo?.layers.find(
    (l) => l.name === 'gauges' && l.available
  )

  const { data: catchmentsGeoJSON } = useQuery({
    queryKey: ['geo-layer', 'catchments'],
    queryFn: () => getGeoLayerGeoJSON('catchments'),
    enabled: !!catchmentsAvailable,
    staleTime: 300_000,
  })

  const { data: gaugesGeoJSON } = useQuery({
    queryKey: ['geo-layer', 'gauges'],
    queryFn: () => getGeoLayerGeoJSON('gauges'),
    enabled: !!gaugesLayerAvailable,
    staleTime: 300_000,
  })

  const gaugeMarkers = gaugesGeoJSON?.features
    ?.filter((f: any) => f.geometry?.type === 'Point')
    ?.map((f: any) => ({
      id: String(f.properties?.gauge_id || f.properties?.id || f.properties?.name || ''),
      lat: f.geometry.coordinates[1],
      lng: f.geometry.coordinates[0],
      name: f.properties?.name || f.properties?.gauge_id || '',
    }))

  const fitTarget = catchmentsGeoJSON || gaugesGeoJSON || null

  return (
    <div className={`relative rounded-lg overflow-hidden border border-gray-200 ${className}`}>
      <MapContainer
        center={ACT_CENTER}
        zoom={DEFAULT_ZOOM}
        className="w-full h-full"
        ref={mapRef}
        zoomControl={true}
        style={{ minHeight: 300 }}
      >
        <TileLayer
          attribution='&copy; Esri &mdash; Esri, DeLorme, NAVTEQ'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        />
        <TileLayer
          attribution='&copy; Esri'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
          opacity={0.6}
        />

        {catchmentsGeoJSON && (
          <GeoJSON data={catchmentsGeoJSON} style={() => catchmentStyle} />
        )}

        {gaugeMarkers?.map((g: any) => (
          <Marker
            key={g.id}
            position={[g.lat, g.lng]}
            icon={gaugeIcon}
            eventHandlers={{
              click: () => onGaugeClick?.(g.id),
            }}
          >
            <Popup>
              <span className="font-semibold">{g.id}</span>
              {g.name && g.name !== g.id && (
                <span className="block text-xs text-gray-500">{g.name}</span>
              )}
            </Popup>
          </Marker>
        ))}

        <FitBoundsToLayers geojson={fitTarget} />
      </MapContainer>

      {/* Overlay badge when no layers loaded */}
      {!catchmentsAvailable && !gaugesLayerAvailable && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/20 pointer-events-none">
          <div className="bg-white/90 backdrop-blur px-4 py-3 rounded-lg text-center shadow">
            <p className="text-sm font-medium text-gray-700">
              No geospatial layers loaded
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Upload catchment boundaries or gauge locations via the Layers panel
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
