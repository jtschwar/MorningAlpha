import type { RiskLevel } from '../../store/types'
import styles from './Badge.module.css'

// ── Score badge (0–100) ──────────────────────────────────────────────────────

interface ScoreBadgeProps {
  value: number | null
  label?: string
}

export function ScoreBadge({ value, label }: ScoreBadgeProps) {
  if (value == null) return <span className={styles.na}>N/A</span>
  const tier = value >= 70 ? 'high' : value >= 40 ? 'mid' : 'low'
  return (
    <span className={`${styles.badge} ${styles[tier]}`} title={label}>
      {value.toFixed(1)}
    </span>
  )
}

// ── RSI badge ────────────────────────────────────────────────────────────────

interface RsiBadgeProps {
  value: number | null
}

export function RsiBadge({ value }: RsiBadgeProps) {
  if (value == null) return <span className={styles.na}>N/A</span>
  const tier = value >= 70 ? 'overbought' : value <= 30 ? 'oversold' : 'neutral'
  const label = tier === 'overbought' ? 'Overbought' : tier === 'oversold' ? 'Oversold' : 'Neutral'
  return (
    <span className={`${styles.badge} ${styles[tier]}`} title={`RSI: ${label}`}>
      {value.toFixed(0)}
    </span>
  )
}

// ── Risk badge ───────────────────────────────────────────────────────────────

interface RiskBadgeProps {
  level: RiskLevel
}

export function RiskBadge({ level }: RiskBadgeProps) {
  return (
    <span className={`${styles.badge} ${styles['risk-' + level]}`}>
      {level.replace('-', ' ')}
    </span>
  )
}

// ── Return badge ─────────────────────────────────────────────────────────────

interface ReturnBadgeProps {
  value: number
}

export function ReturnBadge({ value }: ReturnBadgeProps) {
  const cls = value >= 0 ? styles.positive : styles.negative
  return (
    <span className={`${styles.badge} ${cls}`}>
      {value >= 0 ? '+' : ''}{value.toFixed(2)}%
    </span>
  )
}
