import type { Stock } from '../../store/types'
import { ScoreBadge, RsiBadge } from '../common/Badge'
import styles from './CsvMetricsStrip.module.css'

interface Props {
  stock: Stock
}

function scoreLabel(v: number | null, thresholds: [number, string][], fallback = '—'): string {
  if (v == null) return fallback
  for (const [min, label] of thresholds) {
    if (v >= min) return label
  }
  return thresholds[thresholds.length - 1][1]
}

export default function CsvMetricsStrip({ stock: s }: Props) {
  function fmt(v: number | null, suffix = '') {
    return v != null ? `${v.toFixed(2)}${suffix}` : '—'
  }

  const invLabel  = scoreLabel(s.investmentScore, [[75,'Strong candidate'],[60,'Solid pick'],[45,'Average']], 'Weak candidate')
  const qualLabel = scoreLabel(s.QualityScore,    [[70,'High quality'],[50,'Average quality']], 'Low quality')
  const entryLabel= scoreLabel(s.EntryScore,      [[70,'Good entry'],[50,'Fair entry']], 'Poor entry')

  const sharpeLabel =
    s.SharpeRatio == null ? '—'
    : s.SharpeRatio >= 1.5 ? 'Excellent risk-adjusted'
    : s.SharpeRatio >= 1.0 ? 'Good risk-adjusted'
    : s.SharpeRatio >= 0.5 ? 'Acceptable'
    : 'Poor risk-adjusted'

  const sortinoLabel =
    s.SortinoRatio == null ? '—'
    : s.SortinoRatio >= 2.0 ? 'Strong downside protection'
    : s.SortinoRatio >= 1.0 ? 'Decent downside protection'
    : 'High downside risk'

  const ddLabel =
    s.MaxDrawdown == null ? '—'
    : s.MaxDrawdown > -15 ? 'Mild pullback'
    : s.MaxDrawdown > -30 ? 'Moderate decline'
    : s.MaxDrawdown > -50 ? 'Severe drawdown'
    : 'Extreme drawdown'

  const momLabel =
    s.MomentumAccel == null ? '—'
    : s.MomentumAccel > 0.5 ? 'Accelerating fast'
    : s.MomentumAccel > 0 ? 'Momentum building'
    : s.MomentumAccel > -0.5 ? 'Momentum fading'
    : 'Decelerating'

  return (
    <div className={styles.strip}>
      <div className={styles.sectionLabel}>Performance</div>
      <div className={styles.items}>
        <div className={styles.item}>
          <span className={styles.label}>Inv. Score</span>
          <ScoreBadge value={s.investmentScore} />
          <span className={styles.sub}>{invLabel}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>Quality</span>
          <ScoreBadge value={s.QualityScore} />
          <span className={styles.sub}>{qualLabel}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>Entry Score</span>
          <ScoreBadge value={s.EntryScore} />
          <span className={styles.sub}>{entryLabel}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>Sharpe</span>
          <span className={styles.val}>{fmt(s.SharpeRatio)}</span>
          <span className={styles.sub}>{sharpeLabel}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>Sortino</span>
          <span className={styles.val}>{fmt(s.SortinoRatio)}</span>
          <span className={styles.sub}>{sortinoLabel}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>Max DD</span>
          <span className={`${styles.val} ${s.MaxDrawdown != null && s.MaxDrawdown < -20 ? styles.neg : ''}`}>
            {fmt(s.MaxDrawdown, '%')}
          </span>
          <span className={styles.sub}>{ddLabel}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>RSI (CSV)</span>
          <RsiBadge value={s.RSI} />
          <span className={styles.sub}>
            {s.RSI == null ? '—' : s.RSI >= 70 ? 'Overbought' : s.RSI <= 30 ? 'Oversold' : 'Neutral'}
          </span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>Momentum</span>
          <span className={styles.val}>{fmt(s.MomentumAccel)}</span>
          <span className={styles.sub}>{momLabel}</span>
        </div>
      </div>
    </div>
  )
}
