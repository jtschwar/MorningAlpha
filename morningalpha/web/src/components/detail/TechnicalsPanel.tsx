import { useMemo } from 'react'
import type { StockDetailData } from '../../store/types'
import { calculateEMA, calculateRSI, calculateAnnualizedVolatility, calculateMaxDrawdown } from '../../lib/technicals'
import { RsiBadge } from '../common/Badge'
import styles from './TechnicalsPanel.module.css'

interface Props {
  data: StockDetailData
}

export default function TechnicalsPanel({ data }: Props) {
  const metrics = useMemo(() => {
    const prices = data.close
    const ema20 = calculateEMA(prices, 20)
    const ema50 = calculateEMA(prices, 50)
    const lastEma20 = ema20.filter(Boolean).at(-1) ?? null
    const lastEma50 = ema50.filter(Boolean).at(-1) ?? null
    const lastPrice = prices.at(-1) ?? null
    const rsi = calculateRSI(prices, 14)
    const vol = calculateAnnualizedVolatility(prices)
    const maxDD = calculateMaxDrawdown(prices)

    const emaSignal =
      lastEma20 != null && lastEma50 != null
        ? lastEma20 > lastEma50
          ? 'EMA20 > EMA50 (Short-term Bullish)'
          : 'EMA20 < EMA50 (Short-term Bearish)'
        : null

    return { lastPrice, lastEma20, lastEma50, rsi, vol, maxDD, emaSignal }
  }, [data])

  function fmt(v: number | null, dec = 2): string {
    return v != null ? v.toFixed(dec) : '—'
  }

  return (
    <div className={styles.panel}>
      <div className={styles.title}>Technical Indicators</div>
      <div className={styles.grid}>
        <div className={styles.card}>
          <div className={styles.label}>EMA 20</div>
          <div className={styles.value}>${fmt(metrics.lastEma20)}</div>
          <div className={styles.sub}>Short-term trend</div>
        </div>
        <div className={styles.card}>
          <div className={styles.label}>EMA 50</div>
          <div className={styles.value}>${fmt(metrics.lastEma50)}</div>
          <div className={styles.sub}>Medium-term trend</div>
        </div>
        <div className={styles.card}>
          <div className={styles.label}>EMA Signal</div>
          <div className={`${styles.value} ${metrics.emaSignal?.includes('Bullish') ? styles.pos : styles.neg}`}>
            {metrics.emaSignal?.split(' ')[0] ?? '—'}
          </div>
          <div className={styles.sub}>{metrics.emaSignal ?? 'Insufficient data'}</div>
        </div>
        <div className={styles.card}>
          <div className={styles.label}>Annualized Vol</div>
          <div className={styles.value}>{metrics.vol.toFixed(1)}%</div>
          <div className={styles.sub}>σ_daily × √252 × 100</div>
        </div>
        <div className={styles.card}>
          <div className={styles.label}>Max Drawdown</div>
          <div className={`${styles.value} ${styles.neg}`}>{fmt(metrics.maxDD)}%</div>
          <div className={styles.sub}>Worst decline this period</div>
        </div>
        <div className={styles.card}>
          <div className={styles.label}>RSI (14-day)</div>
          <div className={styles.value}>
            <RsiBadge value={metrics.rsi} />
          </div>
          <div className={styles.sub}>
            {metrics.rsi != null
              ? metrics.rsi >= 70
                ? 'Overbought'
                : metrics.rsi <= 30
                  ? 'Oversold'
                  : 'Neutral'
              : 'Insufficient data'}
          </div>
        </div>
      </div>
    </div>
  )
}
