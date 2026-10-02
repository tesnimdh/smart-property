import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import {
  CircleMarker,
  MapContainer,
  Marker,
  Popup,
  TileLayer,
  useMap,
  useMapEvents,
} from 'react-leaflet'
import type { LatLngBoundsExpression, LatLngTuple } from 'leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

import FormField from '../components/FormField'
import {
  AMENITIES,
  API_BASE,
  defaultAmenityState,
  type AmenityKey,
  type AmenityState,
} from '../constants'

import s from '../components/shared.module.css'
import styles from './SellerPage.module.css'

type SellerFormData = {
  area: string
  pieces: string
  room: string
  bathroom: string
  distance_to_capital: string
  age_num: string
  governorate: string
  city: string
  location: string
  state: string
  latt: string
  long: string
}

type Place = {
  location: string
  city: string
  governorate: string
  latt: number
  long: number
}

type ModelMetrics = {
  r2: number | null
  mae: number | null
  rmse: number | null
  mape: number | null
}

type MapClickHandlerProps = {
  onMapClick: (position: { lat: number; lng: number }) => void
}

type MapRecenterProps = {
  position: LatLngTuple | null
}

const DEFAULT_FORM: SellerFormData = {
  area: '',
  pieces: '',
  room: '',
  bathroom: '',
  distance_to_capital: '',
  age_num: '',
  governorate: '',
  city: '',
  location: '',
  state: '',
  latt: '',
  long: '',
}

const CAPITAL = {
  lat: 36.8065,
  lng: 10.1815,
}

const TUNISIA_CENTER: LatLngTuple = [34.0, 9.5]

const TUNISIA_BOUNDS: LatLngBoundsExpression = [
  [30.0, 7.0],
  [37.6, 12.2],
]

const PROPERTY_STATE_OPTIONS = ['neuf', 'bon état', 'à rénover']

type AmenityDefinition = (typeof AMENITIES)[number]

const HIDDEN_AMENITY_KEYS = new Set<string>(['garden'])

const getAmenityLabel = (amenity: AmenityDefinition) => {
  const record = amenity as AmenityDefinition & {
    label?: unknown
    name?: unknown
  }

  if (typeof record.label === 'string' && record.label.trim()) {
    return record.label
  }

  if (typeof record.name === 'string' && record.name.trim()) {
    return record.name
  }

  return String(amenity.key).replace(/_/g, ' ')
}

const MISSING_TEXT_VALUES = new Set([
  '',
  'none',
  'nan',
  'null',
  'undefined',
  'missing',
  'non renseigné',
  'non renseigne',
])

const markerIcon = new L.Icon({
  iconRetinaUrl:
    'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
})

const normalizeText = (value: unknown): string => {
  if (value === null || value === undefined) {
    return ''
  }

  return String(value).trim().toLowerCase()
}

const isMissingText = (value: unknown): boolean => {
  const normalized = normalizeText(value)

  return MISSING_TEXT_VALUES.has(normalized)
}

const textOrNull = (value: unknown): string | null => {
  if (isMissingText(value)) {
    return null
  }

  return normalizeText(value)
}

const numOrNull = (value: unknown): number | null => {
  if (isMissingText(value)) {
    return null
  }

  const cleaned = String(value).trim().replace(',', '.')
  const n = Number(cleaned)

  return Number.isFinite(n) ? n : null
}

const formatNumber = (value: number, digits = 6) => {
  return Number(value.toFixed(digits)).toString()
}

const unique = (values: string[]) => {
  return Array.from(
    new Set(
      values
        .map((value) => normalizeText(value))
        .filter((value) => value && !MISSING_TEXT_VALUES.has(value))
    )
  ).sort((a, b) => a.localeCompare(b, 'fr'))
}

const parsePlaces = (data: unknown): Place[] => {
  if (!Array.isArray(data)) {
    return []
  }

  return data
    .map((item) => {
      const record = item as Record<string, unknown>

      const latt = Number(record.latt)
      const long = Number(record.long)

      return {
        location: normalizeText(record.location),
        city: normalizeText(record.city),
        governorate: normalizeText(record.governorate),
        latt,
        long,
      }
    })
    .filter(
      (place) =>
        Boolean(place.location) &&
        Boolean(place.city) &&
        Boolean(place.governorate) &&
        Number.isFinite(place.latt) &&
        Number.isFinite(place.long)
    )
}

const formatBackendError = (err: unknown) => {
  if (typeof err === 'string') {
    return err
  }

  if (Array.isArray(err)) {
    return err
      .map((item) => {
        if (typeof item?.msg === 'string') {
          return item.msg
        }

        return 'Erreur de validation'
      })
      .join(' | ')
  }

  if (err && typeof err === 'object') {
    const detail = err as {
      message?: string
      detail?: string
    }

    if (typeof detail.message === 'string') {
      return detail.message
    }

    if (typeof detail.detail === 'string') {
      return detail.detail
    }

    return JSON.stringify(err)
  }

  return 'Erreur serveur'
}

const metricNumberOrNull = (value: unknown): number | null => {
  if (value === null || value === undefined || value === '') {
    return null
  }

  const numericValue = Number(String(value).replace(',', '.'))

  return Number.isFinite(numericValue) ? numericValue : null
}

const readMetricValue = (
  source: Record<string, unknown>,
  keys: string[]
): number | null => {
  for (const key of keys) {
    const value = metricNumberOrNull(source[key])

    if (value !== null) {
      return value
    }
  }

  return null
}

const parseModelMetrics = (data: unknown): ModelMetrics | null => {
  if (!data || typeof data !== 'object') {
    return null
  }

  const record = data as Record<string, unknown>
  const metricsSource =
    record.metrics ?? record.model_metrics ?? record.performance_metrics ?? record

  if (!metricsSource || typeof metricsSource !== 'object') {
    return null
  }

  const metricsRecord = metricsSource as Record<string, unknown>

  const metrics: ModelMetrics = {
    r2: readMetricValue(metricsRecord, ['r2', 'r2_score', 'R2', 'R²', 'score']),
    mae: readMetricValue(metricsRecord, ['mae', 'MAE', 'mean_absolute_error']),
    rmse: readMetricValue(metricsRecord, [
      'rmse',
      'RMSE',
      'root_mean_squared_error',
    ]),
    mape: readMetricValue(metricsRecord, [
      'mape',
      'MAPE',
      'mean_absolute_percentage_error',
    ]),
  }

  return Object.values(metrics).some((value) => value !== null) ? metrics : null
}

const formatMetricValue = (
  key: keyof ModelMetrics,
  value: number | null
): string => {
  if (value === null) {
    return 'Non disponible'
  }

  if (key === 'r2') {
    return value.toFixed(3)
  }

  if (key === 'mape') {
    const percentValue = value <= 1 ? value * 100 : value

    return `${percentValue.toLocaleString('fr-TN', {
      maximumFractionDigits: 2,
    })} %`
  }

  return `${value.toLocaleString('fr-TN', {
    maximumFractionDigits: 0,
  })} TND`
}

const haversineDistance = (
  lat1: number,
  lng1: number,
  lat2: number,
  lng2: number
) => {
  const toRad = (deg: number) => (deg * Math.PI) / 180
  const earthRadiusKm = 6371
  const dLat = toRad(lat2 - lat1)
  const dLng = toRad(lng2 - lng1)

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) *
      Math.cos(toRad(lat2)) *
      Math.sin(dLng / 2) *
      Math.sin(dLng / 2)

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))

  return earthRadiusKm * c
}

const simpleDistance = (lat1: number, lng1: number, lat2: number, lng2: number) => {
  return Math.sqrt((lat1 - lat2) ** 2 + (lng1 - lng2) ** 2)
}

const findNearestPlace = (lat: number, lng: number, places: Place[]) => {
  let nearestPlace: Place | null = null
  let nearestDistance = Infinity

  for (const place of places) {
    const currentDistance = simpleDistance(lat, lng, place.latt, place.long)

    if (currentDistance < nearestDistance) {
      nearestDistance = currentDistance
      nearestPlace = place
    }
  }

  return nearestPlace
}

const filterPlacesByForm = (places: Place[], form: SellerFormData) => {
  const gov = normalizeText(form.governorate)
  const city = normalizeText(form.city)
  const location = normalizeText(form.location)

  return places
    .filter((p) => !gov || normalizeText(p.governorate) === gov)
    .filter((p) => !city || normalizeText(p.city) === city)
    .filter((p) => !location || normalizeText(p.location) === location)
}

const resolveBestPlace = (places: Place[], form: SellerFormData) => {
  const candidates = filterPlacesByForm(places, form)

  return candidates[0]
}

function MapClickHandler({ onMapClick }: MapClickHandlerProps) {
  useMapEvents({
    click(e) {
      onMapClick({ lat: e.latlng.lat, lng: e.latlng.lng })
    },
  })

  return null
}

function MapRecenter({ position }: MapRecenterProps) {
  const map = useMap()

  useEffect(() => {
    if (position) {
      map.setView(position, 12)
    }
  }, [map, position])

  return null
}

export default function SellerPage() {
  const [form, setForm] = useState<SellerFormData>(DEFAULT_FORM)
  const [amenities, setAmenities] = useState<AmenityState>(defaultAmenityState())
  const [places, setPlaces] = useState<Place[]>([])
  const [markerPosition, setMarkerPosition] = useState<LatLngTuple | null>(null)
  const [showMap, setShowMap] = useState(true)

  const [prediction, setPrediction] = useState<number | null>(null)
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true

    const loadPlaces = async () => {
      try {
        const res = await fetch(`${API_BASE}/places`)

        if (!res.ok) {
          throw new Error(`Erreur HTTP ${res.status}`)
        }

        const data = await res.json()

        if (active) {
          setPlaces(parsePlaces(data))
        }
      } catch {
        if (active) {
          setPlaces([])
        }
      }
    }

    loadPlaces()

    return () => {
      active = false
    }
  }, [])

  const governorateOptions = useMemo(() => {
    return unique(places.map((p) => p.governorate))
  }, [places])

  const cityOptions = useMemo(() => {
    const gov = normalizeText(form.governorate)

    return unique(
      places
        .filter((p) => !gov || normalizeText(p.governorate) === gov)
        .map((p) => p.city)
    )
  }, [places, form.governorate])

  const locationOptions = useMemo(() => {
    const gov = normalizeText(form.governorate)
    const city = normalizeText(form.city)

    return unique(
      places
        .filter((p) => !gov || normalizeText(p.governorate) === gov)
        .filter((p) => !city || normalizeText(p.city) === city)
        .map((p) => p.location)
    )
  }, [places, form.governorate, form.city])

  const stateOptions = useMemo(() => PROPERTY_STATE_OPTIONS, [])

  const selectedPlace = useMemo(() => {
    if (!form.governorate || !form.city || !form.location) {
      return undefined
    }

    return resolveBestPlace(places, form)
  }, [places, form])

  const applyCoordinates = useCallback(
    (lat: number, lng: number) => {
      const nearestPlace = findNearestPlace(lat, lng, places)
      const distanceToCapital = haversineDistance(
        lat,
        lng,
        CAPITAL.lat,
        CAPITAL.lng
      )

      setMarkerPosition([lat, lng])

      setForm((prev) => ({
        ...prev,
        ...(nearestPlace
          ? {
              governorate: nearestPlace.governorate,
              city: nearestPlace.city,
              location: nearestPlace.location,
            }
          : {}),
        latt: formatNumber(lat),
        long: formatNumber(lng),
        distance_to_capital: formatNumber(distanceToCapital, 2),
      }))
    },
    [places]
  )

  const handleChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const fieldName = e.target.name as keyof SellerFormData
    const value = e.target.value

    setForm((prev) => {
      const next: SellerFormData = {
        ...prev,
        [fieldName]: value,
      }

      if (fieldName === 'governorate') {
        next.city = ''
        next.location = ''
        next.latt = ''
        next.long = ''
        next.distance_to_capital = ''
      }

      if (fieldName === 'city') {
        next.location = ''
        next.latt = ''
        next.long = ''
        next.distance_to_capital = ''
      }

      if (fieldName === 'location') {
        next.latt = ''
        next.long = ''
        next.distance_to_capital = ''
      }

      const bestPlace = resolveBestPlace(places, next)

      if (
        bestPlace &&
        fieldName === 'location' &&
        next.governorate &&
        next.location
      ) {
        const distanceToCapital = haversineDistance(
          bestPlace.latt,
          bestPlace.long,
          CAPITAL.lat,
          CAPITAL.lng
        )

        next.governorate = bestPlace.governorate
        next.city = bestPlace.city
        next.location = bestPlace.location
        next.latt = formatNumber(bestPlace.latt)
        next.long = formatNumber(bestPlace.long)
        next.distance_to_capital = formatNumber(distanceToCapital, 2)
      }

      return next
    })
  }

  useEffect(() => {
    if (!form.latt || !form.long) {
      setMarkerPosition(null)
      return
    }

    const lat = numOrNull(form.latt)
    const lng = numOrNull(form.long)

    if (lat !== null && lng !== null) {
      setMarkerPosition([lat, lng])
    }
  }, [form.latt, form.long])

  const handleAmenity = (key: AmenityKey, value: boolean) => {
    setAmenities((prev) => ({
      ...prev,
      [key]: value,
    }))
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()

    setLoading(true)
    setError('')
    setPrediction(null)
    setModelMetrics(null)

    const amenityPayload = Object.fromEntries(
      AMENITIES.map((a) => [a.key, amenities[a.key] ? 1.0 : 0.0])
    ) as Record<AmenityKey, number>

    const payload = {
      area: numOrNull(form.area),
      pieces: numOrNull(form.pieces),
      room: numOrNull(form.room),
      bathroom: numOrNull(form.bathroom),
      distance_to_capital: numOrNull(form.distance_to_capital),
      age_num: numOrNull(form.age_num),

      governorate: textOrNull(form.governorate),
      city: textOrNull(form.city),
      location: textOrNull(form.location),
      state: textOrNull(form.state),

      latt: numOrNull(form.latt) ?? selectedPlace?.latt ?? null,
      long: numOrNull(form.long) ?? selectedPlace?.long ?? null,

      ...amenityPayload,
    }

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        let message = 'Erreur serveur'

        try {
          const err = await res.json()
          message = formatBackendError(err.detail)
        } catch {
          message = `Erreur HTTP ${res.status}`
        }

        throw new Error(message)
      }

      const data = await res.json()
      const value = Number(data.predicted_price_tnd)

      if (!Number.isFinite(value)) {
        throw new Error(
          "La réponse du backend ne contient pas une prédiction valide."
        )
      }

      setPrediction(value)
      setModelMetrics(parseModelMetrics(data))
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : `Impossible de contacter le serveur. Vérifiez que le backend tourne sur ${API_BASE}`
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className={s.hero}>
        <h1 className={s.heroTitle}>Estimez le prix de votre bien</h1>
        <p className={s.heroSub}>
          Renseignez les caractéristiques de votre propriété pour obtenir une
          estimation.
        </p>
      </div>

      <form onSubmit={handleSubmit} className={s.formCard}>
        <div className={s.sectionTitle}>Informations générales</div>

        <div className={s.grid3}>
          <FormField label="Superficie (m²)">
            <input
              name="area"
              type="number"
              step="any"
              min="0"
               onWheel={(e) => e.currentTarget.blur()}
               required
              placeholder="ex: 120"
              value={form.area}
              onChange={handleChange}
              className={s.input}
            />
          </FormField>

          <FormField label="Nombre de pièces">
            <input
              name="pieces"
              type="number"
              step="any"
              min="0"
               onWheel={(e) => e.currentTarget.blur()}
              placeholder="ex: 4"
              value={form.pieces}
              onChange={handleChange}
              className={s.input}
            />
          </FormField>

          <FormField label="Chambres">
            <input
              name="room"
              type="number"
              step="any"
              min="0"
               onWheel={(e) => e.currentTarget.blur()}
              placeholder="ex: 3"
              value={form.room}
              onChange={handleChange}
              className={s.input}
            />
          </FormField>

          <FormField label="Salles de bain">
            <input
              name="bathroom"
              type="number"
              step="any"
              min="0"
               onWheel={(e) => e.currentTarget.blur()}
              placeholder="ex: 2"
              value={form.bathroom}
              onChange={handleChange}
              className={s.input}
            />
          </FormField>

          

         

            <FormField label="État du bien">
              <select
                name="state"
                value={form.state}
                onChange={handleChange}
                className={s.select}
              >
                <option value="">Sélectionner </option>
                {stateOptions.map((state) => (
                  <option key={state} value={state}>
                    {state}
                  </option>
                ))}
              </select>
            </FormField>
        </div>

        <div className={s.sectionTitle}>Localisation</div>

        <div
          style={{
            display: 'grid',
            gap: '1.25rem',
          }}
        >
          <button
            type="button"
            onClick={() => setShowMap((current) => !current)}
            className={s.submitBtn}
            style={{ justifySelf: 'start', paddingInline: '1rem' }}
          >
            {showMap ? 'Masquer la carte' : 'Utiliser la carte'}
          </button>

          {showMap && (
            <div
              style={{
                height: 390,
                overflow: 'hidden',
                borderRadius: 18,
                border: '1px solid rgba(148, 163, 184, 0.35)',
              }}
            >
              <MapContainer
                center={TUNISIA_CENTER}
                zoom={7}
                minZoom={6}
                maxZoom={18}
                maxBounds={TUNISIA_BOUNDS}
                maxBoundsViscosity={1.0}
                scrollWheelZoom
                style={{ height: '100%', width: '100%' }}
              >
                <TileLayer
                  attribution="&copy; OpenStreetMap contributors"
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                <MapClickHandler onMapClick={({ lat, lng }) => applyCoordinates(lat, lng)} />
                <MapRecenter position={markerPosition} />

                {markerPosition && (
                  <Marker position={markerPosition} icon={markerIcon}>
                    <Popup>
                      Point sélectionné
                      <br />
                      {form.location || 'Localisation non précisée'}
                    </Popup>
                  </Marker>
                )}

                {places.map((place, index) => (
                  <CircleMarker
                    key={`${place.governorate}-${place.city}-${place.location}-${index}`}
                    center={[place.latt, place.long]}
                    radius={5}
                    pathOptions={{
                      color: '#E91E63',
                      fillColor: '#E91E63',
                      fillOpacity: 0.9,
                    }}
                  >
                    <Popup>
                      <strong>{place.location}</strong>
                      <br />
                      {place.city}
                      <br />
                      {place.governorate}
                    </Popup>
                  </CircleMarker>
                ))}
              </MapContainer>
            </div>
          )}

          <div className={s.grid3}>
            <FormField label="Gouvernorat">
              <select
                name="governorate"
                required
                value={form.governorate}
                onChange={handleChange}
                className={s.select}
              >
                <option value="">Sélectionner un gouvernorat</option>
                {governorateOptions.map((gov) => (
                  <option key={gov} value={gov}>
                    {gov}
                  </option>
                ))}
              </select>
            </FormField>

            <FormField label="Ville">
              <select
                name="city"
                value={form.city}
                onChange={handleChange}
                className={s.select}
                disabled={!form.governorate}
              >
                <option value="">
                  {form.governorate
                    ? 'Sélectionner une ville'
                    : "Sélectionnez d'abord un gouvernorat"}
                </option>
                {cityOptions.map((city) => (
                  <option key={city} value={city}>
                    {city}
                  </option>
                ))}
              </select>
            </FormField>

            <FormField label="Localisation">
              <select
                name="location"
                value={form.location}
                onChange={handleChange}
                className={s.select}
                disabled={!form.governorate}
              >
                <option value="">
                  {form.governorate
                    ? 'Sélectionner une localisation'
                    : "Sélectionnez d'abord un gouvernorat"}
                </option>
                {locationOptions.map((location) => (
                  <option key={location} value={location}>
                    {location}
                  </option>
                ))}
              </select>
            </FormField>
<FormField label="Distance à la capitale (km)">
            <input
              name="distance_to_capital"
              type="number"
              step="any"
              placeholder="calculée ou saisie manuellement"
              value={form.distance_to_capital}
              onChange={handleChange}
              className={s.input}
            />
          </FormField>
            <FormField label="Latitude">
              <input
                name="latt"
                type="number"
                step="any"
                placeholder="calculée ou saisie"
                value={form.latt}
                onChange={handleChange}
                className={s.input}
              />
            </FormField>

            <FormField label="Longitude">
              <input
                name="long"
                type="number"
                step="any"
                placeholder="calculée ou saisie"
                value={form.long}
                onChange={handleChange}
                className={s.input}
              />
            </FormField>
          </div>
        </div>

        <div className={s.sectionTitle}>Équipements</div>
       

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '0.85rem',
          }}
        >
          {AMENITIES.filter(
            (amenity) => !HIDDEN_AMENITY_KEYS.has(String(amenity.key))
          ).map((amenity) => (
            <label
              key={amenity.key}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.65rem',
                padding: '0.85rem 0.95rem',
                borderRadius: 14,
                border: '1px solid rgba(148, 163, 184, 0.35)',
                cursor: 'pointer',
              }}
            >
              <input
                type="checkbox"
                checked={Boolean(amenities[amenity.key])}
                onChange={(event) =>
                  handleAmenity(amenity.key, event.target.checked)
                }
              />
              <span>{getAmenityLabel(amenity)}</span>
            </label>
          ))}
        </div>

        {error && <div className={s.errorMsg}>{error}</div>}
         <br />

        <button
          type="submit"
          disabled={loading}
          className={`${s.submitBtn} ${s.submitBtnSeller}`}
        >
          {loading ? (
            <>
              <span className={s.spinner} />
              Estimation en cours...
            </>
          ) : (
            'Estimer le prix'
          )}
        </button>
      </form>

      {prediction !== null && (
        <div className={styles.predResult}>
          <div className={styles.predLabel}>
            ESTIMATION IA DU PRIX DE MARCHÉ
          </div>

          <div className={styles.predPrice}>
            {Number(prediction).toLocaleString('fr-TN', {
              maximumFractionDigits: 0,
            })}{' '}
            TND
          </div>

          <div className={styles.predSub}>
            Estimation basée sur votre modèle XGBoost entraîné sur les données
            immobilières tunisiennes.
          </div>

          <div
            style={{
              marginTop: '1.25rem',
              paddingTop: '1.25rem',
              borderTop: '1px solid rgba(148, 163, 184, 0.35)',
            }}
          >
            <div
              style={{
                marginBottom: '0.85rem',
                fontSize: '0.78rem',
                fontWeight: 800,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                opacity: 0.78,
              }}
            >
              Métriques de performance du modèle
            </div>

            {modelMetrics ? (
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                  gap: '0.75rem',
                }}
              >
                {([
                  {
                    key: 'r2',
                    label: 'R²',
                    help: 'Pouvoir explicatif du modèle',
                  },
                  
                
                  {
                    key: 'mape',
                    label: 'MAPE',
                    help: 'Erreur moyenne en pourcentage',
                  },
                ] as const).map((metric) => (
                  <div
                    key={metric.key}
                    style={{
                      padding: '0.9rem',
                      borderRadius: 14,
                      border: '1px solid rgba(148, 163, 184, 0.35)',
                      background: 'rgba(255, 255, 255, 0.58)',
                    }}
                  >
                    <div
                      style={{
                        fontSize: '0.76rem',
                        fontWeight: 800,
                        opacity: 0.7,
                      }}
                    >
                      {metric.label}
                    </div>
                    <div
                      style={{
                        marginTop: '0.25rem',
                        fontSize: '1.05rem',
                        fontWeight: 900,
                      }}
                    >
                      {formatMetricValue(metric.key, modelMetrics[metric.key])}
                    </div>
                    <div
                      style={{
                        marginTop: '0.25rem',
                        fontSize: '0.72rem',
                        opacity: 0.68,
                      }}
                    >
                      {metric.help}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div
                style={{
                  padding: '0.95rem',
                  borderRadius: 14,
                  border: '1px dashed rgba(148, 163, 184, 0.6)',
                  fontSize: '0.9rem',
                  opacity: 0.78,
                }}
              >
                Les métriques ne sont pas encore renvoyées par le backend. Pour
                les afficher ici, ajoutez-les dans la réponse de l’API
                <strong> /predict</strong>, par exemple dans un objet
                <strong> metrics</strong>.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
