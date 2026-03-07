import { useMemo } from 'react'
import type { StockDetailData } from '../../store/types'
import { calculateEMA, calculateRSI, calculateAnnualizedVolatility, calculateMaxDrawdown } from '../../lib/technicals'
import { RsiBadge } from '../common/Badge'
import styles from './TechnicalsPanel.module.css'

interface Props {
  data: StockDetailData
}

export default function TechnicalsPanel({ data }: Props) {
  const m = useMemo(() => {
    const prices = data.close
    const ema20 = calculateEMA(prices, 20)
    const ema50 = calculateEMA(prices, 50)
    const lastEma20 = ema20.filter(Boolean).at(-1) ?? null
    const lastEma50 = ema50.filter(Boolean).at(-1) ?? null
    const rsi = calculateRSI(prices, 14)
    const vol = calculateAnnualizedVolatility(prices)
    const maxDD = calculateMaxDrawdown(prices)
    const bullish = lastEma20 != null && lastEma50 != null ? lastEma20 > lastEma50 : null

    const rsiLabel =
      rsi == null ? 'Insufficient data'
      : rsi >= 70 ? 'Overbought — potential reversal'
      : rsi <= 30 ? 'Oversold — potential bounce'
      : rsi >= 55 ? 'Mild bullish momentum'
      : rsi <= 45 ? 'Mild bearish pressure'
      : 'Neutral zone'

    const volLabel =
      vol < 20 ? 'Low volatility'
      : vol < 40 ? 'Moderate volatility'
      : vol < 80 ? 'High volatility'
      : 'Extreme volatility'

    return { lastEma20, lastEma50, rsi, vol, maxDD, bullish, rsiLabel, volLabel }
  }, [data])

  function fmt(v: number | null, dec = 2): string {
    return v != null ? v.toFixed(dec) : '—'
  }

  return (
    <div className={styles.strip}>
      <div className={styles.sectionLabel}>Technical Indicators</div>
      <div className={styles.items}>
        <div className={styles.item}>
          <span className={styles.label}>EMA 20</span>
          <span className={styles.val}>${fmt(m.lastEma20)}</span>
          <span className={styles.sub}>Short-term trend</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>EMA 50</span>
          <span className={styles.val}>${fmt(m.lastEma50)}</span>
          <span className={styles.sub}>Medium-term trend</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>EMA Signal</span>
          <span className={`${styles.val} ${m.bullish === true ? styles.pos : m.bullish === false ? styles.neg : ''}`}>
            {m.bullish === true ? 'Bullish' : m.bullish === false ? 'Bearish' : '—'}
          </span>
          <span className={styles.sub}>
            {m.bullish === true ? 'EMA20 > EMA50'
              : m.bullish === false ? 'EMA20 < EMA50'
              : 'Insufficient data'}
          </span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>Ann. Vol</span>
          <span className={styles.val}>{m.vol.toFixed(1)}%</span>
          <span className={styles.sub}>{m.volLabel}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>Max Drawdown</span>
          <span className={`${styles.val} ${styles.neg}`}>{fmt(m.maxDD)}%</span>
          <span className={styles.sub}>Worst decline this period</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>RSI (14-day)</span>
          <span className={styles.val}><RsiBadge value={m.rsi} /></span>
          <span className={styles.sub}>{m.rsiLabel}</span>
        </div>
      </div>
    </div>
  )
}
