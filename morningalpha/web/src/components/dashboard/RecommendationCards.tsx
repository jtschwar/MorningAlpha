import { useNavigate } from 'react-router-dom'
import type { Stock } from '../../store/types'
import { ScoreBadge, RiskBadge, ReturnBadge } from '../common/Badge'
import styles from './RecommendationCards.module.css'

interface Props {
  stocks: Stock[]
}

export default function RecommendationCards({ stocks }: Props) {
  const navigate = useNavigate()

  const top = [...stocks]
    .filter(s => s.investmentScore != null)
    .sort((a, b) => (b.investmentScore ?? 0) - (a.investmentScore ?? 0))
    .slice(0, 3)

  if (top.length === 0) return null

  return (
    <div className={styles.section}>
      <div className={styles.heading}>Top Picks</div>
      <div className={styles.grid}>
        {top.map((s, i) => (
          <div
            key={s.Ticker}
            className={styles.card}
            onClick={() => navigate(`/stock/${s.Ticker}`)}
            role="button"
            tabIndex={0}
            onKeyDown={e => e.key === 'Enter' && navigate(`/stock/${s.Ticker}`)}
          >
            <div className={styles.rank}>#{i + 1}</div>
            <div className={styles.header}>
              <span className={styles.ticker}>{s.Ticker}</span>
              <ReturnBadge value={s.ReturnPct} />
            </div>
            <div className={styles.name}>{s.Name}</div>

            <div className={styles.metrics}>
              <div className={styles.metric}>
                <span className={styles.mLabel}>Score</span>
                <ScoreBadge value={s.investmentScore} />
              </div>
              <div className={styles.metric}>
                <span className={styles.mLabel}>Quality</span>
                <ScoreBadge value={s.QualityScore} />
              </div>
              <div className={styles.metric}>
                <span className={styles.mLabel}>Entry</span>
                <ScoreBadge value={s.EntryScore} />
              </div>
              <div className={styles.metric}>
                <span className={styles.mLabel}>Risk</span>
                <RiskBadge level={s.riskLevel} />
              </div>
              <div className={styles.metric}>
                <span className={styles.mLabel}>Sharpe</span>
                <span className={styles.mVal}>
                  {s.SharpeRatio != null ? s.SharpeRatio.toFixed(2) : '—'}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
