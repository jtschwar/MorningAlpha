import { useState } from 'react'
import type { Stock, Metadata } from '../../store/types'
import RiskRewardChart from '../charts/RiskRewardChart'
import BarChart from '../charts/BarChart'
import ScatterChart from '../charts/ScatterChart'
import TreemapChart from '../charts/TreemapChart'
import styles from './ChartPanel.module.css'

type Tab = 'riskReward' | 'bar' | 'scatter' | 'treemap'

const TABS: { key: Tab; label: string }[] = [
  { key: 'riskReward', label: 'Risk vs Reward' },
  { key: 'bar', label: 'Returns' },
  { key: 'scatter', label: 'Scatter' },
  { key: 'treemap', label: 'Treemap' },
]

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

      <div className={styles.chart}>
        {tab === 'riskReward' && <RiskRewardChart stocks={stocks} />}
        {tab === 'bar' && <BarChart stocks={stocks} metricLabel={metadata?.metric ?? '3M'} />}
        {tab === 'scatter' && <ScatterChart stocks={stocks} />}
        {tab === 'treemap' && <TreemapChart stocks={stocks} />}
      </div>
    </div>
  )
}
