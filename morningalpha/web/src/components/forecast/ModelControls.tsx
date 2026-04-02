import styles from './ModelControls.module.css'

const MODELS = [
  { id: 'lgbm_breakout_v5', label: 'Breakout' },
  { id: 'lgbm_composite_v6', label: 'Composite' },
  { id: 'st_sector_relative_v1', label: 'Set Transformer' },
]

const HORIZONS: { value: 5 | 10 | 21 | 63; label: string }[] = [
  { value: 5, label: '5D' },
  { value: 10, label: '10D' },
  { value: 21, label: '21D' },
  { value: 63, label: '63D' },
]

interface Props {
  selectedModels: string[]
  horizon: 5 | 10 | 21 | 63
  showBands: boolean
  onToggleModel: (id: string) => void
  onSetHorizon: (h: 5 | 10 | 21 | 63) => void
  onToggleBands: () => void
}

export default function ModelControls({
  selectedModels,
  horizon,
  showBands,
  onToggleModel,
  onSetHorizon,
  onToggleBands,
}: Props) {
  return (
    <div className={styles.wrap}>
      <span className={styles.groupLabel}>Models</span>
      <div className={styles.group}>
        {MODELS.map(m => (
          <label key={m.id} className={styles.checkLabel}>
            <input
              type="checkbox"
              checked={selectedModels.includes(m.id)}
              onChange={() => onToggleModel(m.id)}
            />
            {m.label}
          </label>
        ))}
      </div>

      <div className={styles.divider} />

      <span className={styles.groupLabel}>Horizon</span>
      <div className={styles.horizonGroup}>
        {HORIZONS.map(h => (
          <button
            key={h.value}
            className={`${styles.horizonBtn} ${horizon === h.value ? styles.horizonActive : ''}`}
            onClick={() => onSetHorizon(h.value)}
          >
            {h.label}
          </button>
        ))}
      </div>

      <div className={styles.divider} />

      <label className={styles.checkLabel}>
        <input type="checkbox" checked={showBands} onChange={onToggleBands} />
        Bands
      </label>
    </div>
  )
}
