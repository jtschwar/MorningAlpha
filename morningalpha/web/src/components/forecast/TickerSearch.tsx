import { useState, useRef, useCallback, useEffect } from 'react'
import type { TickerEntry } from '../../hooks/useTickerIndex'
import { useTickerIndex } from '../../hooks/useTickerIndex'
import styles from './TickerSearch.module.css'

interface Props {
  selectedTickers: TickerEntry[]
  onAdd: (entry: TickerEntry) => void
}

function scoreBadgeClass(score: number | null): string {
  if (score === null) return styles.scoreMuted
  if (score >= 70) return styles.scoreGreen
  if (score >= 40) return styles.scoreAmber
  return styles.scoreRed
}

export default function TickerSearch({ selectedTickers, onAdd }: Props) {
  const { tickers, loading } = useTickerIndex()
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [focusIndex, setFocusIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const selectedSet = new Set(selectedTickers.map(t => t.ticker))
  const atMax = selectedTickers.length >= 5

  const filtered = query.trim().length === 0
    ? []
    : tickers
        .filter(t => {
          if (selectedSet.has(t.ticker)) return false
          const q = query.toLowerCase()
          return t.ticker.toLowerCase().includes(q) || t.name.toLowerCase().includes(q)
        })
        .slice(0, 10)

  const handleSelect = useCallback((entry: TickerEntry) => {
    if (atMax) return
    onAdd(entry)
    setQuery('')
    setOpen(false)
    setFocusIndex(0)
    inputRef.current?.focus()
  }, [atMax, onAdd])

  function handleKeyDown(e: React.KeyboardEvent) {
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

  useEffect(() => {
    setFocusIndex(0)
  }, [query])

  return (
    <div className={styles.wrap}>
      <input
        ref={inputRef}
        className={styles.input}
        placeholder={loading ? 'Loading tickers…' : 'Search ticker or name…'}
        value={query}
        disabled={loading || atMax}
        onChange={e => { setQuery(e.target.value); setOpen(true) }}
        onFocus={() => { if (query.trim()) setOpen(true) }}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={handleKeyDown}
        autoComplete="off"
        aria-label="Search tickers"
      />
      {open && query.trim().length > 0 && (
        <div className={styles.dropdown}>
          {atMax ? (
            <div className={styles.maxNote}>Max 5 tickers selected</div>
          ) : filtered.length === 0 ? (
            <div className={styles.noResults}>No matches</div>
          ) : (
            filtered.map((entry, i) => (
              <div
                key={entry.ticker}
                className={`${styles.dropdownItem} ${i === focusIndex ? styles.active : ''}`}
                onMouseDown={() => handleSelect(entry)}
                onMouseEnter={() => setFocusIndex(i)}
              >
                <span className={styles.ticker}>{entry.ticker}</span>
                <span className={styles.name}>{entry.name}</span>
                <span className={`${styles.scoreBadge} ${scoreBadgeClass(entry.mlScore)}`}>
                  {entry.mlScore !== null ? Math.round(entry.mlScore) : '—'}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}
