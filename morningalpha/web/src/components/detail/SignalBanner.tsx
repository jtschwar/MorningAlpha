import type { Signal } from '../../lib/signal'
import styles from './SignalBanner.module.css'

interface Props {
  signal: Signal
}

export default function SignalBanner({ signal }: Props) {
  const cls =
    signal.level === 'STRONG BUY'  ? styles.strongBuy
    : signal.level === 'BUY'       ? styles.buy
    : signal.level === 'HOLD'      ? styles.hold
    : signal.level === 'SELL'      ? styles.sell
    : styles.strongSell

  return (
    <div className={`${styles.banner} ${cls}`}>
      <div className={styles.left}>
        <span className={styles.level}>{signal.level}</span>
        <span className={styles.score}>Score {signal.score}/100</span>
      </div>
      {signal.reasons.length > 0 && (
        <ul className={styles.reasons}>
          {signal.reasons.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      )}
      <span className={styles.disclaimer}>Rule-based · not financial advice</span>
    </div>
  )
}
