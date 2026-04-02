import type { Holding } from '../../lib/portfolioStorage'
import type { TickerEntry } from '../../hooks/useTickerIndex'
import styles from './MLScoreDistribution.module.css'

interface Props {
  holdings: Holding[]
  tickerIndex: TickerEntry[]
}

function barClass(score: number | null): string {
  if (score === null) return styles.barMuted
  if (score >= 70) return styles.barGreen
  if (score >= 40) return styles.barAmber
  return styles.barRed
}

function scoreClass(score: number | null): string {
  if (score === null) return styles.scoreMuted
  if (score >= 70) return styles.scoreGreen
  if (score >= 40) return styles.scoreAmber
  return styles.scoreRed
}

export default function MLScoreDistribution({ holdings, tickerIndex }: Props) {
  if (holdings.length === 0) {
    return (
      <div className={styles.wrap}>
        <div className={styles.title}>ML Score Distribution</div>
        <div className={styles.empty}>No holdings</div>
      </div>
    )
  }

  const enriched = holdings
    .map(h => {
      const entry = tickerIndex.find(t => t.ticker === h.ticker)
      return { ticker: h.ticker, score: entry?.mlScore ?? null }
    })
    .sort((a, b) => {
      const sa = a.score ?? -1
      const sb = b.score ?? -1
      return sb - sa
    })

  return (
    <div className={styles.wrap}>
      <div className={styles.title}>ML Score Distribution</div>
      {enriched.map(({ ticker, score }) => (
        <div key={ticker} className={styles.row}>
          <span className={styles.ticker}>{ticker}</span>
          <div className={styles.barTrack}>
            <div
              className={`${styles.bar} ${barClass(score)}`}
              style={{ width: score !== null ? `${score}%` : '0%' }}
            />
          </div>
          <span className={`${styles.scoreLabel} ${scoreClass(score)}`}>
            {score !== null ? Math.round(score) : '—'}
          </span>
        </div>
      ))}
    </div>
  )
}
