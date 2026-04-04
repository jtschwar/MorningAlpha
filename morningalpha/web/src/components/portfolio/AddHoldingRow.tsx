import { useState, useRef, useCallback, useEffect } from 'react'
import type { Holding } from '../../lib/portfolioStorage'
import type { TickerEntry } from '../../hooks/useTickerIndex'
import { useTickerIndex } from '../../hooks/useTickerIndex'
import styles from './AddHoldingRow.module.css'

interface Props {
  onAdd: (holding: Holding) => void
}

function scoreBadgeClass(score: number | null): string {
  if (score === null) return styles.scoreMuted
  if (score >= 70) return styles.scoreGreen
  if (score >= 40) return styles.scoreAmber
  return styles.scoreRed
}

export default function AddHoldingRow({ onAdd }: Props) {
  const { tickers, loading } = useTickerIndex()
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [focusIndex, setFocusIndex] = useState(0)
  const [selected, setSelected] = useState<TickerEntry | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const filtered = query.trim().length === 0
    ? []
    : tickers
        .filter(t => {
          const q = query.toLowerCase()
          return t.ticker.toLowerCase().includes(q) || t.name.toLowerCase().includes(q)
        })
        .slice(0, 10)

  const handleSelect = useCallback((entry: TickerEntry) => {
    setSelected(entry)
    setQuery('')
    setOpen(false)
    setFocusIndex(0)
  }, [])

  function handleClear() {
    setSelected(null)
    setQuery('')
    setTimeout(() => inputRef.current?.focus(), 0)
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && (!open || filtered.length === 0)) {
      handleSubmit()
      return
    }
    if (!open || filtered.length === 0) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setFocusIndex(i => Math.min(i + 1, filtered.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setFocusIndex(i => Math.max(i - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (filtered[focusIndex]) handleSelect(filtered[focusIndex])
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  useEffect(() => { setFocusIndex(0) }, [query])

  const canSubmit = selected !== null

  function handleSubmit() {
    if (!selected) return
    onAdd({
      id: crypto.randomUUID(),
      ticker: selected.ticker,
      addedAt: new Date().toISOString(),
    })
    setSelected(null)
    setQuery('')
  }

  return (
    <div className={styles.row}>
      <span className={styles.label}>Add holding:</span>

      <div className={styles.tickerWrap}>
        {selected ? (
          <div className={styles.selectedChip}>
            <span className={styles.chipTicker}>{selected.ticker}</span>
            <span className={styles.chipName}>{selected.name}</span>
            <button className={styles.chipClear} onClick={handleClear} aria-label="Clear ticker">×</button>
          </div>
        ) : (
          <input
            ref={inputRef}
            className={styles.input}
            placeholder={loading ? 'Loading…' : 'Search ticker or name…'}
            value={query}
            disabled={loading}
            onChange={e => { setQuery(e.target.value); setOpen(true) }}
            onFocus={() => { if (query.trim()) setOpen(true) }}
            onBlur={() => setTimeout(() => setOpen(false), 150)}
            onKeyDown={handleKeyDown}
            autoComplete="off"
          />
        )}
        {open && !selected && query.trim().length > 0 && (
          <div className={styles.dropdown}>
            {filtered.length === 0 ? (
              <div className={styles.noResults}>No matches</div>
            ) : (
              filtered.map((entry, i) => (
                <div
                  key={entry.ticker}
                  className={`${styles.dropdownItem} ${i === focusIndex ? styles.active : ''}`}
                  onMouseDown={() => handleSelect(entry)}
                  onMouseEnter={() => setFocusIndex(i)}
                >
                  <span className={styles.dropTicker}>{entry.ticker}</span>
                  <span className={styles.dropName}>{entry.name}</span>
                  <span className={`${styles.scoreBadge} ${scoreBadgeClass(entry.mlScore)}`}>
                    {entry.mlScore !== null ? Math.round(entry.mlScore) : '—'}
                  </span>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      <button className={styles.addBtn} onClick={handleSubmit} disabled={!canSubmit}>
        + Add
      </button>
    </div>
  )
}
