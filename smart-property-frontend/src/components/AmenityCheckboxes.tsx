import { AMENITIES, type AmenityKey, type AmenityState } from '../constants'
import styles from './AmenityCheckboxes.module.css'

type Props = {
  state: AmenityState
  onChange: (key: AmenityKey, value: boolean) => void
}

export default function AmenityCheckboxes({ state, onChange }: Props) {
  return (
    <div className={styles.grid}>
      {AMENITIES.map((a) => (
        <label
          key={a.key}
          className={`${styles.item} ${state[a.key] ? styles.checked : ''}`}
          onClick={() => onChange(a.key, !state[a.key])}
        >
          <span className={styles.checkIcon}>{state[a.key] ? '✓' : ''}</span>
          <span>{a.label}</span>
        </label>
      ))}
    </div>
  )
}
