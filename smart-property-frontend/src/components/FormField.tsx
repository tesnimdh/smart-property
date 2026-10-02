import styles from './FormField.module.css'

type Props = {
  label: string
  children: React.ReactNode
}

export default function FormField({ label, children }: Props) {
  return (
    <div className={styles.field}>
      <label className={styles.label}>{label}</label>
      {children}
    </div>
  )
}
