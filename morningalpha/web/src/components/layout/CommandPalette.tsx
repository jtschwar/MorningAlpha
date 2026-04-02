import { useState, useEffect, useRef, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTickerIndex } from '../../hooks/useTickerIndex'
import styles from './CommandPalette.module.css'

const PAGES = [
  { label: 'Screener', path: '/', description: 'Stock screener & rankings' },
  { label: 'Forecast', path: '/forecast', description: 'ML price fan chart' },
  { label: 'Portfolio', path: '/portfolio', description: 'Holdings tracker' },
  { label: 'Backtest', path: '/backtest', description: 'Model evaluation' },
]

interface Props {
  open: boolean
  onClose: () => void
}

export default function CommandPalette({ open, onClose }: Props) {
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()
  const { tickers } = useTickerIndex()

  useEffect(() => {
    if (open) {
      setQuery('')
      setCursor(0)
      // Defer focus so the element is visible first
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [open])

  const filteredPages = useMemo(() => {
    if (!query.trim()) return PAGES
    const q = query.toLowerCase()
    return PAGES.filter(
      p => p.label.toLowerCase().includes(q) || p.description.toLowerCase().includes(q)
    )
  }, [query])

  const filteredTickers = useMemo(() => {
    if (!query.trim()) return []
    const q = query.toLowerCase()
    return tickers
      .filter(t => t.ticker.toLowerCase().includes(q) || t.name.toLowerCase().includes(q))
      .slice(0, 6)
  }, [query, tickers])

  // Flat ordered list for keyboard navigation
  const allResults = useMemo(() => [
    ...filteredPages.map(p => ({ kind: 'page' as const, label: p.label, description: p.description, path: p.path })),
    ...filteredTickers.map(t => ({
      kind: 'ticker' as const,
      label: t.ticker,
      description: t.name,
      path: `/forecast?ticker=${t.ticker}`,
      mlScore: t.mlScore,
    })),
  ], [filteredPages, filteredTickers])

  function select(idx: number) {
    const item = allResults[idx]
    if (!item) return
    navigate(item.path)
    onClose()
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setCursor(c => Math.min(c + 1, allResults.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setCursor(c => Math.max(c - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      select(cursor)
    } else if (e.key === 'Escape') {
      onClose()
    }
  }

  if (!open) return null

  // Track result index across sections for cursor matching
  let ri = 0

  return (
    <div className={styles.backdrop} onMouseDown={onClose}>
      <div className={styles.modal} onMouseDown={e => e.stopPropagation()}>
        <div className={styles.inputRow}>
          <span className={styles.searchIcon}>⌘</span>
          <input
            ref={inputRef}
            className={styles.input}
            placeholder="Search tickers, pages…"
            value={query}
            onChange={e => { setQuery(e.target.value); setCursor(0) }}
            onKeyDown={handleKeyDown}
            autoComplete="off"
            spellCheck={false}
          />
          {query && (
            <button className={styles.clearBtn} onMouseDown={() => setQuery('')}>✕</button>
          )}
        </div>

        <div className={styles.results}>
          {filteredPages.length > 0 && (
            <section>
              <div className={styles.sectionLabel}>Pages</div>
              {filteredPages.map(page => {
                const idx = ri++
                return (
                  <button
                    key={page.path}
                    className={`${styles.result} ${cursor === idx ? styles.active : ''}`}
                    onMouseDown={() => select(idx)}
                    onMouseEnter={() => setCursor(idx)}
                  >
                    <span className={styles.resultIcon}>⊞</span>
                    <span className={styles.resultLabel}>{page.label}</span>
                    <span className={styles.resultDesc}>{page.description}</span>
                  </button>
                )
              })}
            </section>
          )}

          {filteredTickers.length > 0 && (
            <section>
              <div className={styles.sectionLabel}>Tickers</div>
              {filteredTickers.map(ticker => {
                const idx = ri++
                const scoreLevel = ticker.mlScore == null ? null
                  : ticker.mlScore >= 70 ? 'high'
                  : ticker.mlScore >= 40 ? 'mid'
                  : 'low'
                return (
                  <button
                    key={ticker.ticker}
                    className={`${styles.result} ${cursor === idx ? styles.active : ''}`}
                    onMouseDown={() => select(idx)}
                    onMouseEnter={() => setCursor(idx)}
                  >
                    <span className={styles.resultTicker}>{ticker.ticker}</span>
                    <span className={styles.resultDesc}>{ticker.name}</span>
                    {scoreLevel && (
                      <span className={styles.resultScore} data-level={scoreLevel}>
                        {ticker.mlScore}
                      </span>
                    )}
                  </button>
                )
              })}
            </section>
          )}

          {allResults.length === 0 && query.trim() && (
            <div className={styles.empty}>No results for "{query}"</div>
          )}
        </div>

        <div className={styles.footer}>
          <span>↑↓ navigate</span>
          <span>↵ open</span>
          <span>esc close</span>
        </div>
      </div>
    </div>
  )
}
