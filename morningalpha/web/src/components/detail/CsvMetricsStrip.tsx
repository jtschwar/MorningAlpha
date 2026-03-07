import type { Stock } from '../../store/types'
import { ScoreBadge, RsiBadge } from '../common/Badge'
import styles from './CsvMetricsStrip.module.css'

interface Props {
  stock: Stock
}

export default function CsvMetricsStrip({ stock: s }: Props) {
  function fmt(v: number | null, suffix = '') {
    return v != null ? `${v.toFixed(2)}${suffix}` : '—'
  }

  return (
    <div className={styles.strip}>
      <div className={styles.item}>
        <span className={styles.label}>Inv. Score</span>
        <ScoreBadge value={s.investmentScore} />
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Quality</span>
        <ScoreBadge value={s.QualityScore} />
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Entry Score</span>
        <ScoreBadge value={s.EntryScore} />
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Sharpe</span>
        <span className={styles.val}>{fmt(s.SharpeRatio)}</span>
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Sortino</span>
        <span className={styles.val}>{fmt(s.SortinoRatio)}</span>
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Max DD</span>
        <span className={`${styles.val} ${s.MaxDrawdown != null && s.MaxDrawdown < -20 ? styles.neg : ''}`}>
          {fmt(s.MaxDrawdown, '%')}
        </span>
      </div>
      <div className={styles.item}>
        <span className={styles.label}>RSI (CSV)</span>
        <RsiBadge value={s.RSI} />
      </div>
      <div className={styles.item}>
        <span className={styles.label}>Momentum</span>
        <span className={styles.val}>{fmt(s.MomentumAccel)}</span>
      </div>
    </div>
  )
}
