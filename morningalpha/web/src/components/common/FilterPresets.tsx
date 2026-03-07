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
    filters: { riskTolerance: 'conservative', minQuality: 60, maxDrawdown: -20, sortBy: 'investmentScore' },
  },
  {
    label: 'Momentum',
    filters: { riskTolerance: 'all', minQuality: 0, maxDrawdown: -100, sortBy: 'momentumAccel' },
  },
  {
    label: 'Value Entry',
    filters: { riskTolerance: 'all', minQuality: 40, maxDrawdown: -100, sortBy: 'entryScore' },
  },
  {
    label: 'Quality First',
    filters: { riskTolerance: 'all', minQuality: 70, maxDrawdown: -100, sortBy: 'quality' },
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
