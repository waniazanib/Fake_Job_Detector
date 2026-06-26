import { ShieldCheck } from 'lucide-react'
import styles from './Header.module.css'

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <div className={styles.brand}>
          <ShieldCheck className={styles.icon} aria-hidden="true" />
          <span className={styles.wordmark}>
            Job<span className={styles.wordmarkAccent}>Guard</span>
          </span>
        </div>

      </div>
    </header>
  )
}