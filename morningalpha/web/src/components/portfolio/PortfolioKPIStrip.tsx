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

  // Average mlScore across holdings
  const scoredHoldings = holdings.filter(h => {
    const entry = getEntry(h.ticker, tickerIndex)
    return entry?.mlScore !== null && entry?.mlScore !== undefined
  })
  const mlHealth = scoredHoldings.length > 0
    ? scoredHoldings.reduce((sum, h) => sum + (getEntry(h.ticker, tickerIndex)?.mlScore ?? 0), 0) / scoredHoldings.length
    : null

  // Average traditional investmentScore across holdings
  const tradScoredHoldings = holdings.filter(h => {
    const entry = getEntry(h.ticker, tickerIndex)
    return entry?.investmentScore !== null && entry?.investmentScore !== undefined
  })
  const tradHealth = tradScoredHoldings.length > 0
    ? tradScoredHoldings.reduce((sum, h) => sum + (getEntry(h.ticker, tickerIndex)?.investmentScore ?? 0), 0) / tradScoredHoldings.length
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

  // Count of holdings with multi-bagger signal (≥50% model confidence of doubling in 1y)
  const multiBaggerCount = holdings.filter(h => {
    const entry = getEntry(h.ticker, tickerIndex)
    return (entry?.breakoutProb_100pct_252d ?? 0) >= 0.5
  }).length

  return (
    <div className={styles.strip}>
      <div className={styles.card}>
        <span className={styles.label}>Holdings</span>
        <span className={styles.value}>{holdingCount}</span>
      </div>

      <div className={styles.card}>
        <span className={styles.label}>Trad. Health</span>
        <span className={`${styles.value} ${scoreClass(tradHealth)}`}>
          {tradHealth !== null ? Math.round(tradHealth) : '—'}
        </span>
        <span className={`${styles.sub} ${styles.muted}`}>avg score</span>
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

      <div className={styles.card}>
        <span className={styles.label}>Multi-bagger</span>
        <span className={`${styles.value} ${multiBaggerCount > 0 ? styles.pos : styles.muted}`}>
          {multiBaggerCount}
        </span>
        <span className={`${styles.sub} ${styles.muted}`}>252d ×2 ≥50%</span>
      </div>
    </div>
  )
}
