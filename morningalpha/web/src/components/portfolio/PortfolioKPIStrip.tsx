import type { Holding } from '../../lib/portfolioStorage'
import type { TickerEntry } from '../../hooks/useTickerIndex'
import styles from './PortfolioKPIStrip.module.css'

interface Props {
  holdings: Holding[]
  tickerIndex: TickerEntry[]
}

function scoreClass(v: number | null): string {
  if (v === null) return styles.muted
  if (v >= 70) return styles.pos
  if (v >= 40) return styles.amber
  return styles.neg
}

function getEntry(ticker: string, index: TickerEntry[]): TickerEntry | undefined {
  return index.find(t => t.ticker === ticker)
}

export default function PortfolioKPIStrip({ holdings, tickerIndex }: Props) {
  const holdingCount = holdings.length

  // Simple average mlScore across holdings
  const scoredHoldings = holdings.filter(h => {
    const entry = getEntry(h.ticker, tickerIndex)
    return entry?.mlScore !== null && entry?.mlScore !== undefined
  })
  const mlHealth = scoredHoldings.length > 0
    ? scoredHoldings.reduce((sum, h) => sum + (getEntry(h.ticker, tickerIndex)?.mlScore ?? 0), 0) / scoredHoldings.length
    : null

  // Top signal
  interface TopEntry { ticker: string; score: number }
  const topEntry: TopEntry | null = holdings.reduce<TopEntry | null>((best, h) => {
    const entry = getEntry(h.ticker, tickerIndex)
    const score = entry?.mlScore ?? null
    if (score === null) return best
    if (!best || score > best.score) return { ticker: h.ticker, score }
    return best
  }, null)

  // Bearish alert count
  const alertCount = holdings.filter(h => {
    const entry = getEntry(h.ticker, tickerIndex)
    return entry?.mlScore !== null && entry?.mlScore !== undefined && (entry.mlScore ?? 100) < 40
  }).length

  return (
    <div className={styles.strip}>
      <div className={styles.card}>
        <span className={styles.label}>Holdings</span>
        <span className={styles.value}>{holdingCount}</span>
      </div>

      <div className={styles.card}>
        <span className={styles.label}>Total Value</span>
        <span className={`${styles.value} ${styles.muted}`}>—</span>
        <span className={`${styles.sub} ${styles.muted}`}>No price data</span>
      </div>

      <div className={styles.card}>
        <span className={styles.label}>ML Health</span>
        <span className={`${styles.value} ${scoreClass(mlHealth)}`}>
          {mlHealth !== null ? Math.round(mlHealth) : '—'}
        </span>
        <span className={`${styles.sub} ${styles.muted}`}>avg score</span>
      </div>

      <div className={styles.card}>
        <span className={styles.label}>Top Signal</span>
        {topEntry !== null ? (
          <>
            <span className={`${styles.value} ${scoreClass(topEntry.score)}`}>
              {topEntry.ticker}
            </span>
            <span className={`${styles.sub} ${scoreClass(topEntry.score)}`}>
              {Math.round(topEntry.score)}
            </span>
          </>
        ) : (
          <span className={`${styles.value} ${styles.muted}`}>—</span>
        )}
      </div>

      <div className={styles.card}>
        <span className={styles.label}>Alerts</span>
        <span className={`${styles.value} ${alertCount > 0 ? styles.neg : styles.muted}`}>
          {alertCount}
        </span>
        <span className={`${styles.sub} ${styles.muted}`}>bearish signals</span>
      </div>
    </div>
  )
}
