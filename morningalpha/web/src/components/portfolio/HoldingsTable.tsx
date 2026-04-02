import { useState } from 'react'
import type { Holding } from '../../lib/portfolioStorage'
import type { TickerEntry } from '../../hooks/useTickerIndex'
import styles from './HoldingsTable.module.css'

type TabId = 'holdings' | 'signals'
type SortKey = 'ticker' | 'shares' | 'avgCost' | 'mlScore' | 'investmentScore'

interface Props {
  holdings: Holding[]
  tickerIndex: TickerEntry[]
  onDelete: (id: string) => void
}

function getEntry(ticker: string, index: TickerEntry[]): TickerEntry | undefined {
  return index.find(t => t.ticker === ticker)
}

function signalProps(score: number | null): { text: string; cls: string } {
  if (score === null) return { text: 'N/A', cls: styles.signalNA }
  if (score >= 70) return { text: 'Bullish', cls: styles.signalBullish }
  if (score >= 40) return { text: 'Neutral', cls: styles.signalNeutral }
  return { text: 'Bearish', cls: styles.signalBearish }
}

function scoreClass(v: number | null): string {
  if (v === null) return styles.muted
  if (v >= 70) return styles.pos
  if (v >= 40) return styles.amber
  return styles.neg
}

export default function HoldingsTable({ holdings, tickerIndex, onDelete }: Props) {
  const [activeTab, setActiveTab] = useState<TabId>('holdings')
  const [sortKey, setSortKey] = useState<SortKey>('ticker')
  const [sortAsc, setSortAsc] = useState(true)

  function handleSort(key: SortKey) {
    if (sortKey === key) setSortAsc(a => !a)
    else { setSortKey(key); setSortAsc(true) }
  }

  const sorted = [...holdings].sort((a, b) => {
    let va: number | string = 0
    let vb: number | string = 0
    if (sortKey === 'ticker') { va = a.ticker; vb = b.ticker }
    else if (sortKey === 'shares') { va = a.shares; vb = b.shares }
    else if (sortKey === 'avgCost') { va = a.avgCost; vb = b.avgCost }
    else if (sortKey === 'mlScore') {
      va = getEntry(a.ticker, tickerIndex)?.mlScore ?? -1
      vb = getEntry(b.ticker, tickerIndex)?.mlScore ?? -1
    }
    else if (sortKey === 'investmentScore') {
      va = getEntry(a.ticker, tickerIndex)?.investmentScore ?? -1
      vb = getEntry(b.ticker, tickerIndex)?.investmentScore ?? -1
    }
    if (va < vb) return sortAsc ? -1 : 1
    if (va > vb) return sortAsc ? 1 : -1
    return 0
  })

  const thProps = (key: SortKey, label: string) => ({
    onClick: () => handleSort(key),
    title: `Sort by ${label}`,
    children: label + (sortKey === key ? (sortAsc ? ' ↑' : ' ↓') : ''),
  })

  if (holdings.length === 0) {
    return (
      <div className={styles.wrap}>
        <div className={styles.tabRow}>
          <button className={`${styles.tab} ${styles.tabActive}`}>Holdings</button>
          <button className={styles.tab}>Signals</button>
        </div>
        <div className={styles.emptyState}>
          No holdings yet. Add a ticker above.
        </div>
      </div>
    )
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.tabRow}>
        <button
          className={`${styles.tab} ${activeTab === 'holdings' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('holdings')}
        >
          Holdings
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'signals' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('signals')}
        >
          Signals
        </button>
      </div>

      <div className={styles.tableWrap}>
        {activeTab === 'holdings' && (
          <table className={styles.table}>
            <thead>
              <tr>
                <th {...thProps('ticker', 'Ticker')} style={{ textAlign: 'left' }} />
                <th {...thProps('shares', 'Shares')} />
                <th {...thProps('avgCost', 'Avg Cost')} />
                <th {...thProps('investmentScore', 'Score')} />
                <th {...thProps('mlScore', 'ML Score')} />
                <th>Signal</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(h => {
                const entry = getEntry(h.ticker, tickerIndex)
                const score = entry?.mlScore ?? null
                const { text, cls } = signalProps(score)
                return (
                  <tr key={h.id}>
                    <td>
                      <span className={styles.ticker}>{h.ticker}</span>
                      {entry?.name && <span className={styles.name}>{entry.name}</span>}
                    </td>
                    <td>{h.shares.toLocaleString()}</td>
                    <td>{h.avgCost > 0 ? `$${h.avgCost.toFixed(2)}` : '—'}</td>
                    <td className={scoreClass(entry?.investmentScore ?? null)}>
                      {entry?.investmentScore !== null && entry?.investmentScore !== undefined
                        ? entry.investmentScore.toFixed(1) : '—'}
                    </td>
                    <td className={scoreClass(score)}>
                      {score !== null ? Math.round(score) : '—'}
                    </td>
                    <td>
                      <span className={`${styles.signal} ${cls}`}>{text}</span>
                    </td>
                    <td>
                      <button
                        className={styles.deleteBtn}
                        onClick={() => onDelete(h.id)}
                        title={`Remove ${h.ticker}`}
                        aria-label={`Remove ${h.ticker}`}
                      >
                        🗑
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}

        {activeTab === 'signals' && (
          <table className={styles.table}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left' }}>Ticker</th>
                <th>Consensus</th>
                <th>Breakout</th>
                <th>Composite</th>
                <th>Set Transformer</th>
                <th>Signal</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(h => {
                const entry = getEntry(h.ticker, tickerIndex)
                const { text, cls } = signalProps(entry?.mlScore ?? null)
                return (
                  <tr key={h.id}>
                    <td>
                      <span className={styles.ticker}>{h.ticker}</span>
                    </td>
                    <td className={scoreClass(entry?.mlScore ?? null)}>
                      {entry?.mlScore !== null && entry?.mlScore !== undefined ? Math.round(entry.mlScore) : '—'}
                    </td>
                    <td className={scoreClass(entry?.mlScore_breakout ?? null)}>
                      {entry?.mlScore_breakout !== null && entry?.mlScore_breakout !== undefined
                        ? Math.round(entry.mlScore_breakout) : '—'}
                    </td>
                    <td className={scoreClass(entry?.mlScore_composite ?? null)}>
                      {entry?.mlScore_composite !== null && entry?.mlScore_composite !== undefined
                        ? Math.round(entry.mlScore_composite) : '—'}
                    </td>
                    <td className={scoreClass(entry?.mlScore_st ?? null)}>
                      {entry?.mlScore_st !== null && entry?.mlScore_st !== undefined
                        ? Math.round(entry.mlScore_st) : '—'}
                    </td>
                    <td>
                      <span className={`${styles.signal} ${cls}`}>{text}</span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
