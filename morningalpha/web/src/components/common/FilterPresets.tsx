import { useState } from 'react'
import { useStock } from '../../store/StockContext'
import type { FilterState } from '../../store/types'
import styles from './FilterPresets.module.css'

interface Preset {
  label: string
  filters: Partial<FilterState>
}

const PRESETS: Preset[] = [
  {
    label: 'Conservative',
    filters: {
      betaMax: 1.0,
      maxDrawdown: -20,
      minSharpe: 1.0,
      dividend: 'has_dividend',
      sortBy: 'investmentScore',
    },
  },
  {
    label: 'Momentum',
    filters: {
      rsiMin: 50,
      rsiMax: 70,
      smaPosition: 'above_sma50',
      sortBy: 'momentumAccel',
    },
  },
  {
    label: 'Value Entry',
    filters: {
      peMax: 20,
      smaPosition: 'below_sma200',
      minQuality: 40,
      sortBy: 'entryScore',
    },
  },
  {
    label: 'Quality Growth',
    filters: {
      minQuality: 60,
      minSharpe: 1.5,
      sortBy: 'quality',
    },
  },
  {
    label: 'Breakout',
    filters: {
      rsiMin: 50,
      rsiMax: 70,
      smaPosition: 'above_sma50',
      sortBy: 'momentumAccel',
    },
  },
  {
    label: 'Undervalued',
    filters: {
      peMax: 15,
      rsiMax: 40,
      sortBy: 'entryScore',
    },
  },
]

export default function FilterPresets() {
  const { dispatch } = useStock()
  const [active, setActive] = useState<string | null>(null)

  function apply(p: Preset) {
    if (active === p.label) {
      dispatch({ type: 'RESET_FILTERS' })
      setActive(null)
    } else {
      dispatch({ type: 'RESET_FILTERS' })
      dispatch({ type: 'SET_FILTERS', filters: p.filters })
      setActive(p.label)
    }
  }

  return (
    <div className={styles.grid}>
      {PRESETS.map(p => (
        <button
          key={p.label}
          className={`${styles.chip} ${active === p.label ? styles.active : ''}`}
          onClick={() => apply(p)}
        >
          {p.label}
        </button>
      ))}
    </div>
  )
}
