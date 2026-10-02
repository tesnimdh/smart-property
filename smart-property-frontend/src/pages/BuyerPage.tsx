import { useEffect, useMemo, useState } from 'react'
import type { ChangeEvent, FormEvent } from 'react'

import FormField from '../components/FormField'
import {
  AMENITIES,
  API_BASE,
  defaultAmenityState,
  type AmenityKey,
  type AmenityState,
} from '../constants'

import s from '../components/shared.module.css'
import styles from './BuyerPage.module.css'

type FormData = {
  budget_max: string
  governorate: string
  city: string
  location: string
  min_area: string
  min_pieces: string
  min_room: string
  min_bathroom: string
  state: string
}

type Place = {
  location: string
  city: string
  governorate: string
  latt: number
  long: number
}

type ResultItem = {
  price_tnd: number | null
  location: string | null
  city: string | null
  governorate: string | null
  Area: number | null
  pieces: number | null
  room: number | null
  bathroom: number | null
  state: string | null
  garage: number | null
  garden: number | null
  concierge?: number | null
  beach_view?: number | null
  mountain_view?: number | null
  pool: number | null
  elevator?: number | null
  furnished: number | null
  equipped_kitchen?: number | null
  central_heating?: number | null
  air_conditioning?: number | null
  score: number | null
}

const DEFAULT_FORM: FormData = {
  budget_max: '',
  governorate: '',
  city: '',
  location: '',
  min_area: '',
  min_pieces: '',
  min_room: '',
  min_bathroom: '',
  state: '',
}

const PROPERTY_STATE_OPTIONS = ['neuf', 'bon état', 'à rénover']

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

type AmenityDefinition = (typeof AMENITIES)[number]


const HIDDEN_AMENITY_KEYS = new Set<string>(['garden'])

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

const formatBackendError = (detail: unknown) => {
  if (typeof detail === 'string') {
    return detail
  }

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === 'object' && 'msg' in item) {
          return String((item as { msg?: string }).msg)
        }

        return 'Erreur de validation'
      })
      .join(' | ')
  }

  if (detail && typeof detail === 'object') {
    const record = detail as {
      message?: string
      detail?: string
    }

    if (typeof record.message === 'string') {
      return record.message
    }

    if (typeof record.detail === 'string') {
      return record.detail
    }

    return JSON.stringify(detail)
  }

  return 'Erreur serveur'
}

export default function BuyerPage() {
  const [form, setForm] = useState<FormData>(DEFAULT_FORM)
  const [amenities, setAmenities] = useState<AmenityState>(defaultAmenityState())
  const [places, setPlaces] = useState<Place[]>([])
  const [results, setResults] = useState<ResultItem[] | null>(null)
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

  const handleChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const fieldName = e.target.name as keyof FormData
    const value = e.target.value

    setForm((prev) => {
      const next: FormData = {
        ...prev,
        [fieldName]: value,
      }

      if (fieldName === 'governorate') {
        next.city = ''
        next.location = ''
      }

      if (fieldName === 'city') {
        next.location = ''
      }

      return next
    })
  }

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
    setResults(null)

    const amenityPayload = Object.fromEntries(
      AMENITIES.map((a) => [a.key, amenities[a.key] ? 1.0 : 0.0])
    ) as Record<AmenityKey, number>

    const payload = {
      budget_max: numOrNull(form.budget_max),
      governorate: textOrNull(form.governorate),
      city: textOrNull(form.city),
      location: textOrNull(form.location),
      min_area: numOrNull(form.min_area),
      min_pieces: numOrNull(form.min_pieces),
      min_room: numOrNull(form.min_room),
      min_bathroom: numOrNull(form.min_bathroom),
      state: textOrNull(form.state),
      ...amenityPayload,
    }

    try {
      const res = await fetch(`${API_BASE}/recommendation/recommend`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        let message = `Erreur HTTP ${res.status}`

        try {
          const err = await res.json()
          message = formatBackendError(err.detail)
        } catch {
          message = `Erreur HTTP ${res.status}`
        }

        throw new Error(message)
      }

      const data = await res.json()

      if (!Array.isArray(data)) {
        throw new Error("Le backend n'a pas retourné une liste de recommandations.")
      }

      setResults(data as ResultItem[])
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
        <h1 className={`${s.heroTitle} ${s.heroTitleSeller}`}>
          Trouvez votre maison idéale
        </h1>

        <p className={s.heroSub}>
          Entrez vos critères et nous vous recommandons les meilleures propriétés disponibles.
        </p>
      </div>

      <form onSubmit={handleSubmit} className={s.formCard} >
        <div className={s.sectionTitle}>Critères de recherche</div>

        <div className={s.grid3}>
          <FormField label="Budget maximum (TND)">
            <input
              name="budget_max"
              type="number"
              step="any"
              min="0"
              required
               onWheel={(e) => e.currentTarget.blur()}
              placeholder="ex: 350000"
              value={form.budget_max}
              onChange={handleChange}
              className={s.input}
            />
          </FormField>

          <FormField label="Surface minimale (m²)">
            <input
              name="min_area"
              type="number"
              step="any"
              min="0"
               onWheel={(e) => e.currentTarget.blur()}
              placeholder="ex: 80"
              value={form.min_area}
              onChange={handleChange}
              className={s.input}
            />
          </FormField>

          <FormField label="Nombre de pièces minimum">
            <input
              name="min_pieces"
              type="number"
              min="0"
               onWheel={(e) => e.currentTarget.blur()}
              step="any"
              placeholder="ex: 3"
              value={form.min_pieces}
              onChange={handleChange}
              className={s.input}
            />
          </FormField>

          <FormField label="Chambres minimum">
            <input
              name="min_room"
              type="number"
              step="any"
              min="0"
               onWheel={(e) => e.currentTarget.blur()}
              placeholder="ex: 2"
              value={form.min_room}
              onChange={handleChange}
              className={s.input}
            />
          </FormField>

          <FormField label="Salles de bain minimum">
            <input
              name="min_bathroom"
              type="number"
              step="any"
              min="0"
               onWheel={(e) => e.currentTarget.blur()}
              placeholder="ex: 1"
              value={form.min_bathroom}
              onChange={handleChange}
              className={s.input}
            />
          </FormField>

    
        </div>

        <div className={s.sectionTitle}>Localisation</div>

        <div className={s.grid3}>
          <FormField label="Gouvernorat">
            <select
              name="governorate"
              value={form.governorate}
              onChange={handleChange}
              required
              className={s.select}
            >
              <option value=""> Sélectionner un gouvernorat</option>
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
                  ? ' Sélectionner une ville'
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
                  ? ' Sélectionner une localisation'
                  : "Sélectionnez d'abord un gouvernorat"}
              </option>
              {locationOptions.map((location) => (
                <option key={location} value={location}>
                  {location}
                </option>
              ))}
            </select>
          </FormField>
        </div>

        <div className={s.sectionTitle}>Équipements souhaités</div>

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
              Recherche en cours...
            </>
          ) : (
            'Rechercher des propriétés'
          )}
        </button>
      </form>

      {results !== null && (
        <div>
          <div className={styles.resultsHeader}>
            {results.length === 0
              ? 'Aucune propriété trouvée'
              : `${results.length} propriété${results.length > 1 ? 's' : ''} trouvée${results.length > 1 ? 's' : ''}`}
          </div>

          {results.length === 0 ? (
            <div className={styles.emptyState}>
              <div className={styles.emptyIcon}></div>
              <p>
                Aucune propriété ne correspond à vos critères.
                <br />
                Essayez d'élargir votre recherche.
              </p>
            </div>
          ) : (
            <div className={styles.resultsGrid}>
              {results.map((item, i) => (
                <ResultCard key={i} item={item} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ResultCard({ item }: { item: ResultItem }) {
  const tags: string[] = []

  if (item.Area) tags.push(`${item.Area} m²`)
  if (item.pieces) tags.push(`${item.pieces} pièces`)
  if (item.room) tags.push(`${item.room} ch.`)
  if (item.bathroom) tags.push(`${item.bathroom} sdb`)
  if (item.pool) tags.push('Piscine')
  if (item.garden) tags.push('Jardin')
  if (item.furnished) tags.push('Meublé')
  if (item.garage) tags.push('Garage')

  const score = Math.min(Math.round(Number(item.score || 0)), 100)

  return (
    <div className={styles.card}>
      <div className={styles.cardCity}>{item.city || '—'}</div>

      <div className={styles.cardLoc}>
        {item.location || ''}
        {item.governorate ? ` · ${item.governorate}` : ''}
      </div>

      <div className={styles.cardPrice}>
        {item.price_tnd
          ? Number(item.price_tnd).toLocaleString('fr-TN') + ' TND'
          : 'Prix non disponible'}
      </div>

      <div className={styles.tags}>
        {tags.map((tag) => (
          <span key={tag} className={styles.tag}>
            {tag}
          </span>
        ))}
      </div>

      {item.state && (
        <div className={styles.cardState}>
          État : {item.state}
        </div>
      )}

      <div className={styles.scoreWrap}>
        <div className={styles.scoreRow}>
          <span>Correspondance</span>
          <span>{score}%</span>
        </div>

        <div className={styles.scoreBar}>
          <div
            className={styles.scoreFill}
            style={{ width: `${score}%` }}
          />
        </div>
      </div>
    </div>
  )
}
