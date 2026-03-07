import styles from './StatCard.module.css'

interface Props {
  label: string
  value: string | number
  sub?: string
  positive?: boolean
  negative?: boolean
}

export default function StatCard({ label, value, sub, positive, negative }: Props) {
  const valueClass = positive ? styles.positive : negative ? styles.negative : ''
  return (
    <div className={styles.card}>
      <div className={styles.label}>{label}</div>
      <div className={`${styles.value} ${valueClass}`}>{value}</div>
      {sub && <div className={styles.sub}>{sub}</div>}
    </div>
  )
}
