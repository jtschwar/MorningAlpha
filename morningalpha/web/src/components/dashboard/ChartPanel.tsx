import { useState } from 'react'
import type { Stock, Metadata } from '../../store/types'
import RiskRewardChart from '../charts/RiskRewardChart'
import BarChart from '../charts/BarChart'
import ScatterChart from '../charts/ScatterChart'
import styles from './ChartPanel.module.css'

type Tab = 'riskReward' | 'bar' | 'scatter'

const TABS: { key: Tab; label: string }[] = [
  { key: 'riskReward', label: 'Risk vs Reward' },
  { key: 'bar', label: 'Returns' },
  { key: 'scatter', label: 'Scatter' },
]

const TAB_DESCRIPTIONS: Record<Tab, string> = {
  riskReward:
    'Each dot is a stock. The x-axis shows max drawdown — the largest peak-to-trough decline a stock experienced over the period (e.g. 30% means it fell 30% from its high before recovering). Further right = bigger historical loss. The y-axis shows total return — higher is better. The ideal zone is top-left: strong returns with a small drawdown. Colour indicates risk level. Dots in the bottom-right suffered large losses with little reward.',
  bar:
    'Each bar shows the price return for a stock over the selected period. Taller bars = stronger performance. Use this to quickly rank which stocks gained or lost the most, and to spot outliers at either end of the distribution.',
  scatter:
    'A two-dimensional view that plots any two metrics against each other — each dot is a stock. Hover a dot to see the ticker. Look for clusters in the top-right (strong on both axes) and outliers. Useful for spotting correlations, such as whether high-quality stocks tend to have better risk-adjusted returns.',
}

interface Props {
  stocks: Stock[]
  metadata: Metadata | null
}

export default function ChartPanel({ stocks, metadata }: Props) {
  const [tab, setTab] = useState<Tab>('riskReward')

  if (stocks.length === 0) return null

  return (
    <div className={styles.panel}>
      <div className={styles.tabs}>
        {TABS.map(t => (
          <button
            key={t.key}
            className={`${styles.tab} ${tab === t.key ? styles.active : ''}`}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <p className={styles.description}>{TAB_DESCRIPTIONS[tab]}</p>

      <div className={styles.chart}>
        {tab === 'riskReward' && <RiskRewardChart stocks={stocks} />}
        {tab === 'bar' && <BarChart stocks={stocks} metricLabel={metadata?.metric ?? '3M'} />}
        {tab === 'scatter' && <ScatterChart stocks={stocks} />}
      </div>
    </div>
  )
}
