import type { TickerEntry } from '../../hooks/useTickerIndex'
import styles from './StockChips.module.css'

const SERIES_COLORS = ['#3B82F6', '#F59E0B', '#22C55E', '#A78BFA', '#EF4444']

interface Props {
  tickers: TickerEntry[]
  activeIndex: number
  onActivate: (index: number) => void
  onRemove: (index: number) => void
}

export default function StockChips({ tickers, activeIndex, onActivate, onRemove }: Props) {
  if (tickers.length === 0) {
    return (
      <div className={styles.row}>
        <span className={styles.empty}>No stocks selected — search above</span>
      </div>
    )
  }

  return (
    <div className={styles.row}>
      {tickers.map((t, i) => (
        <div
          key={t.ticker}
          className={`${styles.chip} ${i === activeIndex ? styles.chipActive : ''}`}
          style={{ '--dot-color': SERIES_COLORS[i] } as React.CSSProperties}
          onClick={() => onActivate(i)}
        >
          <span className={styles.dot} />
          <span className={styles.label}>{t.ticker}</span>
          <button
            className={styles.remove}
            onClick={e => { e.stopPropagation(); onRemove(i) }}
            aria-label={`Remove ${t.ticker}`}
            title={`Remove ${t.ticker}`}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  )
}

export { SERIES_COLORS }
