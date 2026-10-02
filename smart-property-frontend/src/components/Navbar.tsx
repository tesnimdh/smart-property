import type { Mode } from '../App'
import styles from './Navbar.module.css'

type Props = {
  mode: Mode
  onModeChange: (m: Mode) => void
}

export default function Navbar({ mode, onModeChange }: Props) {
  return (
    <nav className={styles.nav}>
      <div className={styles.logo}>
        Immo<span>Tunisia</span>
      </div>

      <div className={styles.tabs}>
       
        <button
          className={`${styles.tab} ${mode === 'seller' ? styles.activeSeller : ''}`}
          onClick={() => onModeChange('seller')}
        >
           Vendeur
        </button>

         <button
          className={`${styles.tab} ${mode === 'buyer' ? styles.activeBuyer : ''}`}
          onClick={() => onModeChange('buyer')}
        >
           Acheteur
        </button>
        
      </div>
    </nav>
  )
}