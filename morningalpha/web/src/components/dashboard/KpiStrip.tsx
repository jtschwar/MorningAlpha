import { useMemo } from 'react'
import StatCard from '../common/StatCard'
import type { Stock } from '../../store/types'
import styles from './KpiStrip.module.css'

interface Props {
  stocks: Stock[]
}

export default function KpiStrip({ stocks }: Props) {
  const stats = useMemo(() => {
    if (stocks.length === 0) return null

    const returns = stocks.map(s => s.ReturnPct)
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length
    const bestReturn = Math.max(...returns)

    const sharpes = stocks.map(s => s.SharpeRatio).filter((v): v is number => v != null)
    const avgSharpe = sharpes.length ? sharpes.reduce((a, b) => a + b, 0) / sharpes.length : null

    const qualities = stocks.map(s => s.QualityScore).filter((v): v is number => v != null)
    const avgQuality = qualities.length
      ? qualities.reduce((a, b) => a + b, 0) / qualities.length
      : null

    return { count: stocks.length, avgReturn, bestReturn, avgSharpe, avgQuality }
  }, [stocks])

  if (!stats) return null

  return (
    <div className={styles.strip}>
      <StatCard label="Stocks" value={stats.count} />
      <StatCard
        label="Avg Return"
        value={`${stats.avgReturn >= 0 ? '+' : ''}${stats.avgReturn.toFixed(2)}%`}
        positive={stats.avgReturn >= 0}
        negative={stats.avgReturn < 0}
      />
      <StatCard
        label="Best Return"
        value={`+${stats.bestReturn.toFixed(2)}%`}
        positive
      />
      <StatCard
        label="Avg Sharpe"
        value={stats.avgSharpe != null ? stats.avgSharpe.toFixed(2) : 'N/A'}
        positive={stats.avgSharpe != null && stats.avgSharpe > 1}
      />
      <StatCard
        label="Avg Quality"
        value={stats.avgQuality != null ? stats.avgQuality.toFixed(1) : 'N/A'}
      />
    </div>
  )
}
