import { useState, useRef, useEffect } from 'react'
import { useStock } from '../../store/StockContext'
import styles from './ColumnPicker.module.css'

interface ColumnGroup {
  label: string
  columns: string[]
}

export const COLUMN_GROUPS: ColumnGroup[] = [
  {
    label: 'Core',
    columns: ['Name', 'Exchange', 'Return %', 'Sharpe', 'Sortino', 'Quality', 'Entry', 'Max DD', 'RSI', 'Score', 'ML Score'],
  },
  {
    label: 'ML Signals',
    columns: ['63d Brk%', '252d 50%', '252d ×2'],
  },
  {
    label: 'Technical — Trend',
    columns: ['SMA20 Dist%', 'SMA50 Dist%', 'SMA200 Dist%', 'MACD', 'MACD Hist', 'EMA7', 'EMA200'],
  },
  {
    label: 'Technical — Momentum',
    columns: ['RSI(7)', 'RSI(21)', 'Stoch %K', 'Stoch %D', 'ROC(5)', 'ROC(10)', 'ROC(21)'],
  },
  {
    label: 'Technical — Volatility',
    columns: ['ATR', 'Boll %B', 'Boll BW%', 'Ann. Vol%'],
  },
  {
    label: 'Technical — Volume',
    columns: ['Rel Volume', 'Vol ROC%', 'OBV'],
  },
  {
    label: 'Fundamental',
    columns: ['Sector', 'Industry', 'Mkt Cap', 'P/E', 'Fwd P/E', 'P/B', 'P/S', 'PEG', 'EPS', 'ROE', 'ROA', 'Gross Margin', 'Debt/Eq', 'Div Yield', 'Beta', 'Short Float%'],
  },
]

export const PINNED_COLUMNS = ['Rank', 'Ticker']

export default function ColumnPicker() {
  const { state, dispatch } = useStock()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const visible = state.columnConfig.visibleColumns

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    if (open) document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open])

  function toggle(col: string) {
    const next = visible.includes(col)
      ? visible.filter(c => c !== col)
      : [...visible, col]
    dispatch({ type: 'SET_COLUMN_CONFIG', columns: next })
  }

  return (
    <div className={styles.wrapper} ref={ref}>
      <button
        className={styles.gearBtn}
        onClick={() => setOpen(v => !v)}
        title="Choose columns"
        aria-label="Column picker"
      >
        ⚙
      </button>
      {open && (
        <div className={styles.popover}>
          <div className={styles.popoverHeader}>
            <span className={styles.popoverTitle}>Columns</span>
            <button
              className={styles.resetBtn}
              onClick={() => dispatch({ type: 'RESET_COLUMN_CONFIG' })}
            >
              Reset
            </button>
          </div>
          <div className={styles.pinned}>
            {PINNED_COLUMNS.map(col => (
              <label key={col} className={`${styles.checkLabel} ${styles.pinnedRow}`}>
                <input type="checkbox" checked readOnly disabled />
                <span>{col}</span>
                <span className={styles.pinnedTag}>pinned</span>
              </label>
            ))}
          </div>
          {COLUMN_GROUPS.map(group => (
            <div key={group.label} className={styles.group}>
              <div className={styles.groupLabel}>{group.label}</div>
              {group.columns.map(col => (
                <label key={col} className={styles.checkLabel}>
                  <input
                    type="checkbox"
                    checked={visible.includes(col)}
                    onChange={() => toggle(col)}
                  />
                  <span>{col}</span>
                </label>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
