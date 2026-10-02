export const AMENITIES = [
  { key: 'garage', label: 'Garage' },
  { key: 'concierge', label: 'Concierge' },
  { key: 'beach_view', label: 'Vue sur mer' },
  { key: 'pool', label: 'Piscine' },
  { key: 'elevator', label: 'Ascenseur' },
  { key: 'furnished', label: 'Meublé' },
  { key: 'equipped_kitchen', label: 'Cuisine équipée' },
  { key: 'central_heating', label: 'Chauffage central' },
  { key: 'air_conditioning', label: 'Climatisation' },
] as const

export type AmenityKey = (typeof AMENITIES)[number]['key']
export type AmenityState = Record<AmenityKey, boolean>

export function defaultAmenityState(): AmenityState {
  return Object.fromEntries(AMENITIES.map((a) => [a.key, false])) as AmenityState
}

export const API_BASE = 'http://127.0.0.1:8000'
