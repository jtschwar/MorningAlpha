import type { TickerEntry } from '../../hooks/useTickerIndex'
import type { ForecastCalibration } from '../../hooks/useForecastCalibration'
import styles from './DecileContextChart.module.css'

interface Props {
  ticker: TickerEntry | null
  calibration: ForecastCalibration | null
  loading: boolean
}

export default function DecileContextChart({ ticker, calibration, loading }: Props) {
  if (loading) {
    return (
      <div className={styles.wrap}>
        <div className={styles.title}>Decile Context (63d)</div>
        <div className={styles.loading}>Loading…</div>
      </div>
    )
  }

  if (!calibration) {
    return (
      <div className={styles.wrap}>
        <div className={styles.title}>Decile Context (63d)</div>
        <div className={styles.empty}>No calibration data</div>
      </div>
    )
  }

  const deciles = calibration.horizons['63'] ?? []
  if (deciles.length === 0) {
    return (
      <div className={styles.wrap}>
        <div className={styles.title}>Decile Context (63d)</div>
        <div className={styles.empty}>No 63d decile data</div>
      </div>
    )
  }

  // Active ticker's decile (1-based)
  const activeDecile = ticker !== null
    ? Math.min(10, Math.max(1, Math.ceil((ticker.mlScore ?? 50) / 10)))
    : null

  const maxAbs = Math.max(...deciles.map(d => Math.abs(d.ann_return)), 0.01)

  return (
    <div className={styles.wrap}>
      <div className={styles.title}>Decile Context (63d · ann. return)</div>
      {deciles.map(d => {
        const isActive = activeDecile !== null && d.decile === activeDecile
        const pct = Math.min(100, (Math.abs(d.ann_return) / maxAbs) * 100)
        let barClass = d.ann_return >= 0 ? styles.barPos : styles.barNeg
        if (isActive) barClass = styles.barHighlight

        return (
          <div key={d.decile} className={styles.row}>
            <span className={styles.decileLabel}>D{d.decile}</span>
            <div className={styles.barTrack} style={{ opacity: isActive ? 1 : 0.5 }}>
              <div
                className={`${styles.bar} ${barClass}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className={styles.pctLabel}>
              {d.ann_return >= 0 ? '+' : ''}{(d.ann_return * 100).toFixed(1)}%
            </span>
          </div>
        )
      })}
      {activeDecile !== null && (
        <div className={styles.legend}>
          <span className={styles.legendDot} />
          Active stock in D{activeDecile}
        </div>
      )}
    </div>
  )
}
