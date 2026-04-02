import type { TickerEntry } from '../../hooks/useTickerIndex'
import type { DecileStats } from '../../hooks/useForecastCalibration'
import styles from './ForecastKPIStrip.module.css'

interface Props {
  ticker: TickerEntry | null
  horizon: 5 | 10 | 21 | 63
  decileStats: DecileStats | null
}

function scoreClass(v: number | null): string {
  if (v === null) return styles.muted
  if (v >= 70) return styles.pos
  if (v >= 40) return styles.amber
  return styles.neg
}

function ScoreCard({ label, value }: { label: string; value: number | null }) {
  return (
    <div className={styles.card}>
      <span className={styles.label}>{label}</span>
      <span className={`${styles.value} ${scoreClass(value)}`}>
        {value !== null ? Math.round(value) : '—'}
      </span>
    </div>
  )
}

export default function ForecastKPIStrip({ ticker, horizon, decileStats }: Props) {
  if (!ticker) {
    return (
      <div className={styles.strip}>
        <div className={styles.emptyCard}>Select a stock to view scores</div>
      </div>
    )
  }

  const expectedReturn = decileStats
    ? (decileStats.period_return_mean * 100).toFixed(1) + '%'
    : '—'

  const retClass = decileStats
    ? decileStats.period_return_mean >= 0 ? styles.pos : styles.neg
    : styles.muted

  return (
    <div className={styles.strip}>
      <ScoreCard label="Consensus" value={ticker.mlScore} />
      <ScoreCard label="Breakout" value={ticker.mlScore_breakout} />
      <ScoreCard label="Composite" value={ticker.mlScore_composite} />
      <ScoreCard label="Set Transformer" value={ticker.mlScore_st} />
      <div className={styles.card}>
        <span className={styles.label}>Expected {horizon}d</span>
        <span className={`${styles.value} ${retClass}`}>{expectedReturn}</span>
      </div>
      <div className={styles.card}>
        <span className={styles.label}>Sector</span>
        <span className={styles.sectorVal}>{ticker.sector ?? '—'}</span>
      </div>
    </div>
  )
}
