import { useState } from 'react'
import type { Holding } from '../../lib/portfolioStorage'
import type { TickerEntry } from '../../hooks/useTickerIndex'
import styles from './AddHoldingRow.module.css'

interface Props {
  tickerIndex: TickerEntry[]
  onAdd: (holding: Holding) => void
}

export default function AddHoldingRow({ tickerIndex, onAdd }: Props) {
  const [ticker, setTicker] = useState('')
  const [shares, setShares] = useState('')
  const [avgCost, setAvgCost] = useState('')

  const tickerUpper = ticker.trim().toUpperCase()
  const sharesNum = parseFloat(shares)
  const costNum = parseFloat(avgCost) || 0

  function handleTickerChange(v: string) {
    setTicker(v)
    setTickerError(false)
  }

  function handleSubmit() {
    if (!tickerUpper) return
    if (!shares || isNaN(sharesNum) || sharesNum <= 0) return

    const holding: Holding = {
      id: crypto.randomUUID(),
      ticker: tickerUpper,
      shares: sharesNum,
      avgCost: costNum,
      addedAt: new Date().toISOString(),
    }
    onAdd(holding)
    setTicker('')
    setShares('')
    setAvgCost('')
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') handleSubmit()
  }

  const canSubmit = tickerUpper.length > 0 && shares.length > 0 && !isNaN(sharesNum) && sharesNum > 0

  return (
    <div className={styles.row}>
      <span className={styles.label}>Add holding:</span>
      <input
        className={`${styles.input} ${styles.tickerInput}`}
        placeholder="TICKER"
        value={ticker}
        onChange={e => handleTickerChange(e.target.value)}
        onKeyDown={handleKeyDown}
        maxLength={8}
      />
      <input
        className={`${styles.input} ${styles.sharesInput}`}
        placeholder="Shares"
        type="number"
        min="0"
        value={shares}
        onChange={e => setShares(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <input
        className={`${styles.input} ${styles.costInput}`}
        placeholder="Avg cost (opt.)"
        type="number"
        min="0"
        step="0.01"
        value={avgCost}
        onChange={e => setAvgCost(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <button className={styles.addBtn} onClick={handleSubmit} disabled={!canSubmit}>
        + Add
      </button>
    </div>
  )
}
