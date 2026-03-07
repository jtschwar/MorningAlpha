import styles from './PeriodSelector.module.css'

const PERIODS = ['1M', '3M', '6M', '1Y', '5Y', 'MAX'] as const
export type DetailPeriod = (typeof PERIODS)[number]

interface Props {
  value: DetailPeriod
  onChange: (p: DetailPeriod) => void
}

export default function PeriodSelector({ value, onChange }: Props) {
  return (
    <div className={styles.group}>
      <span className={styles.label}>Period</span>
      {PERIODS.map(p => (
        <button
          key={p}
          className={`${styles.btn} ${value === p ? styles.active : ''}`}
          onClick={() => onChange(p)}
        >
          {p}
        </button>
      ))}
    </div>
  )
}
