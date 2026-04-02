import type { ForecastCalibration } from '../../hooks/useForecastCalibration'
import styles from './CalibrationDisclaimer.module.css'

interface Props {
  calibration: ForecastCalibration | null
}

export default function CalibrationDisclaimer({ calibration }: Props) {
  return (
    <div className={styles.card}>
      <div className={styles.header}>
        ⚠ Calibration Notice
      </div>
      <div className={styles.body}>
        Projections are based on test-period decile returns and represent a relative ranking signal, not an absolute price target. Past model performance is not indicative of future results. MorningAlpha signals are for informational purposes only and do not constitute investment advice.
      </div>
      {calibration && (
        <div className={styles.period}>
          Test period: {calibration.test_period.start} → {calibration.test_period.end}
        </div>
      )}
    </div>
  )
}
