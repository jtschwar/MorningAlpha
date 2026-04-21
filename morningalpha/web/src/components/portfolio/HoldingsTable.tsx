import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Holding } from '../../lib/portfolioStorage'
import type { TickerEntry } from '../../hooks/useTickerIndex'
import type { Stock } from '../../store/types'
import { useStock } from '../../store/StockContext'
import styles from './HoldingsTable.module.css'

type TabId = 'holdings' | 'signals' | 'info'
type DeltaPeriod = '2w' | '1m' | '3m' | '6m'

interface Props {
  holdings: Holding[]
  tickerIndex: TickerEntry[]
  onDelete: (id: string) => void
}

function getEntry(ticker: string, index: TickerEntry[]): TickerEntry | undefined {
  return index.find(t => t.ticker === ticker)
}

function scoreClass(v: number | null): string {
  if (v === null) return styles.muted
  if (v >= 70) return styles.pos
  if (v >= 40) return styles.amber
  return styles.neg
}

function signalLabel(score: number | null): { text: string; cls: string } {
  if (score === null) return { text: 'N/A', cls: styles.signalNA }
  if (score >= 70) return { text: 'Bullish', cls: styles.signalBullish }
  if (score >= 40) return { text: 'Neutral', cls: styles.signalNeutral }
  return { text: 'Bearish', cls: styles.signalBearish }
}

function riskClass(level: string | undefined): string {
  if (!level) return styles.muted
  if (level === 'low') return styles.pos
  if (level === 'moderate') return styles.amber
  return styles.neg
}

function ddClass(dd: number | null): string {
  if (dd == null) return styles.muted
  if (dd > -10) return styles.pos
  if (dd > -30) return styles.amber
  return styles.neg
}

function rsiClass(rsi: number | null | undefined): string {
  if (rsi == null) return styles.muted
  if (rsi >= 70) return styles.rsiHot
  if (rsi <= 30) return styles.rsiCold
  return ''
}

function relVolClass(rv: number | null | undefined): string {
  if (rv == null) return styles.muted
  if (rv >= 2) return styles.pos
  if (rv >= 1.5) return styles.amber
  if (rv < 0.5) return styles.muted
  return ''
}

function derivePrice(s: Stock): number | null {
  if (s.SMA20 != null && s.PriceToSMA20Pct != null)
    return s.SMA20 * (1 + s.PriceToSMA20Pct / 100)
  if (s.SMA200 != null && s.PriceToSMA200Pct != null)
    return s.SMA200 * (1 + s.PriceToSMA200Pct / 100)
  return null
}

function fmtPrice(v: number | null): string {
  return v != null ? `$${v.toFixed(2)}` : '—'
}

function fmtPct(v: number | null): string {
  if (v == null) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`
}

function fmtMarketCap(v: number | null | undefined): string {
  if (v == null) return '—'
  if (v >= 1e12) return `$${(v / 1e12).toFixed(1)}T`
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`
  if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}M`
  return `$${v.toFixed(0)}`
}

const PERIOD_LABELS: Record<DeltaPeriod, string> = {
  '2w': '2W', '1m': '1M', '3m': '3M', '6m': '6M',
}

export default function HoldingsTable({ holdings, tickerIndex, onDelete }: Props) {
  const navigate = useNavigate()
  const { state } = useStock()
  const [activeTab, setActiveTab] = useState<TabId>('holdings')
  const [deltaPeriod, setDeltaPeriod] = useState<DeltaPeriod>('3m')
  const [sortKey, setSortKey] = useState<string>('ticker')
  const [sortAsc, setSortAsc] = useState(true)

  // Build ticker → Stock lookup for the selected delta period
  const stockMap = useMemo(() => {
    const map: Record<string, Stock> = {}
    const stocks = state.windowData[deltaPeriod] ?? state.windowData['3m'] ?? []
    stocks.forEach(s => { map[s.Ticker] = s })
    return map
  }, [state.windowData, deltaPeriod])

  function handleSort(key: string) {
    if (sortKey === key) setSortAsc(a => !a)
    else { setSortKey(key); setSortAsc(true) }
  }

  const sorted = useMemo(() => [...holdings].sort((a, b) => {
    const ea = getEntry(a.ticker, tickerIndex)
    const eb = getEntry(b.ticker, tickerIndex)
    const sa = stockMap[a.ticker]
    const sb = stockMap[b.ticker]
    let va: number | string = 0
    let vb: number | string = 0
    if (sortKey === 'ticker') { va = a.ticker; vb = b.ticker }
    else if (sortKey === 'return') { va = sa?.ReturnPct ?? -999; vb = sb?.ReturnPct ?? -999 }
    else if (sortKey === 'maxdd') { va = sa?.MaxDrawdown ?? -999; vb = sb?.MaxDrawdown ?? -999 }
    else if (sortKey === 'rsi') { va = sa?.RSI ?? -1; vb = sb?.RSI ?? -1 }
    else if (sortKey === 'mlScore') { va = ea?.mlScore ?? -1; vb = eb?.mlScore ?? -1 }
    else if (sortKey === 'investmentScore') { va = ea?.investmentScore ?? -1; vb = eb?.investmentScore ?? -1 }
    else if (sortKey === 'breakoutProb252d100') { va = ea?.breakoutProb_100pct_252d ?? -1; vb = eb?.breakoutProb_100pct_252d ?? -1 }
    else if (sortKey === 'breakoutProb252d50') { va = ea?.breakoutProb_50pct_252d ?? -1; vb = eb?.breakoutProb_50pct_252d ?? -1 }
    else if (sortKey === 'sharpe') { va = sa?.SharpeRatio ?? -999; vb = sb?.SharpeRatio ?? -999 }
    else if (sortKey === 'quality') { va = sa?.QualityScore ?? -1; vb = sb?.QualityScore ?? -1 }
    else if (sortKey === 'entry') { va = sa?.EntryScore ?? -1; vb = sb?.EntryScore ?? -1 }
    if (va < vb) return sortAsc ? -1 : 1
    if (va > vb) return sortAsc ? 1 : -1
    return 0
  }), [holdings, tickerIndex, stockMap, sortKey, sortAsc])

  function th(key: string, label: string, alignLeft = false) {
    const active = sortKey === key
    return (
      <th
        onClick={() => handleSort(key)}
        style={alignLeft ? { textAlign: 'left' } : undefined}
        title={`Sort by ${label}`}
      >
        {label}{active ? (sortAsc ? ' ↑' : ' ↓') : ''}
      </th>
    )
  }

  const tabs: { id: TabId; label: string }[] = [
    { id: 'holdings', label: 'Holdings' },
    { id: 'signals', label: 'Signals' },
    { id: 'info', label: 'Info' },
  ]

  if (holdings.length === 0) {
    return (
      <div className={styles.wrap}>
        <div className={styles.tabRow}>
          {tabs.map(t => (
            <button key={t.id} className={`${styles.tab} ${t.id === 'holdings' ? styles.tabActive : ''}`}>
              {t.label}
            </button>
          ))}
        </div>
        <div className={styles.emptyState}>No holdings yet. Add a ticker above.</div>
      </div>
    )
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.tabRow}>
        {tabs.map(t => (
          <button
            key={t.id}
            className={`${styles.tab} ${activeTab === t.id ? styles.tabActive : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className={styles.tableWrap}>

        {/* ── Holdings tab ───────────────────────────────────────────── */}
        {activeTab === 'holdings' && (
          <table className={styles.table}>
            <thead>
              <tr>
                {th('ticker', 'Ticker', true)}
                <th>Price</th>
                <th>
                  <span className={styles.deltaHeader}>
                    Delta
                    <select
                      className={styles.periodSelect}
                      value={deltaPeriod}
                      onChange={e => setDeltaPeriod(e.target.value as DeltaPeriod)}
                      onClick={e => e.stopPropagation()}
                    >
                      {(Object.keys(PERIOD_LABELS) as DeltaPeriod[]).map(p => (
                        <option key={p} value={p}>{PERIOD_LABELS[p]}</option>
                      ))}
                    </select>
                  </span>
                </th>
                {th('maxdd', 'Max DD')}
                {th('rsi', 'RSI')}
                <th>Rel Vol</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(h => {
                const s = stockMap[h.ticker]
                const entry = getEntry(h.ticker, tickerIndex)
                const price = s ? derivePrice(s) : null
                const ret = s?.ReturnPct ?? null
                const dd = s?.MaxDrawdown ?? null
                return (
                  <tr key={h.id}>
                    <td>
                      <span
                        className={`${styles.ticker} ${styles.tickerLink}`}
                        onClick={() => navigate(`/stock/${h.ticker}`)}
                        title={`View ${h.ticker} detail`}
                      >
                        {h.ticker}
                      </span>
                      {entry?.name && <span className={styles.name}>{entry.name}</span>}
                    </td>
                    <td>{fmtPrice(price)}</td>
                    <td className={ret != null ? (ret >= 0 ? styles.pos : styles.neg) : styles.muted}>
                      {fmtPct(ret)}
                    </td>
                    <td className={ddClass(dd)}>
                      {dd != null ? `${dd.toFixed(1)}%` : '—'}
                    </td>
                    <td className={rsiClass(s?.RSI)}>{s?.RSI != null ? s.RSI.toFixed(1) : '—'}</td>
                    <td className={relVolClass(s?.RelativeVolume)}>{s?.RelativeVolume != null ? `${s.RelativeVolume.toFixed(2)}x` : '—'}</td>
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

        {/* ── Signals tab ───────────────────────────────────────────── */}
        {activeTab === 'signals' && (
          <table className={styles.table}>
            <thead>
              <tr>
                {th('ticker', 'Ticker', true)}
                {th('mlScore', 'ML Score')}
                {th('breakoutProb252d50', '252d 50%')}
                {th('breakoutProb252d100', '252d ×2')}
                {th('investmentScore', 'Trad. Score')}
                {th('sharpe', 'Sharpe')}
                {th('quality', 'Quality')}
                {th('entry', 'Entry')}
                <th>Signal</th>
                <th>Risk</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(h => {
                const entry = getEntry(h.ticker, tickerIndex)
                const s = stockMap[h.ticker]
                const { text, cls } = signalLabel(entry?.mlScore ?? null)
                return (
                  <tr key={h.id}>
                    <td>
                      <span
                        className={`${styles.ticker} ${styles.tickerLink}`}
                        onClick={() => navigate(`/stock/${h.ticker}`)}
                      >
                        {h.ticker}
                      </span>
                    </td>
                    <td className={scoreClass(entry?.mlScore ?? null)}>
                      {entry?.mlScore != null ? Math.round(entry.mlScore) : '—'}
                    </td>
                    <td className={entry?.breakoutProb_50pct_252d != null && entry.breakoutProb_50pct_252d >= 0.6 ? styles.pos : entry?.breakoutProb_50pct_252d != null && entry.breakoutProb_50pct_252d >= 0.4 ? styles.amber : styles.muted}>
                      {entry?.breakoutProb_50pct_252d != null ? `${(entry.breakoutProb_50pct_252d * 100).toFixed(0)}%` : '—'}
                    </td>
                    <td className={entry?.breakoutProb_100pct_252d != null && entry.breakoutProb_100pct_252d >= 0.5 ? styles.pos : entry?.breakoutProb_100pct_252d != null && entry.breakoutProb_100pct_252d >= 0.3 ? styles.amber : styles.muted}>
                      {entry?.breakoutProb_100pct_252d != null ? `${(entry.breakoutProb_100pct_252d * 100).toFixed(0)}%` : '—'}
                    </td>
                    <td className={scoreClass(entry?.investmentScore ?? null)}>
                      {entry?.investmentScore != null ? entry.investmentScore.toFixed(1) : '—'}
                    </td>
                    <td>{s?.SharpeRatio != null ? s.SharpeRatio.toFixed(2) : '—'}</td>
                    <td className={scoreClass(s?.QualityScore ?? null)}>
                      {s?.QualityScore != null ? Math.round(s.QualityScore) : '—'}
                    </td>
                    <td className={scoreClass(s?.EntryScore ?? null)}>
                      {s?.EntryScore != null ? Math.round(s.EntryScore) : '—'}
                    </td>
                    <td><span className={`${styles.signal} ${cls}`}>{text}</span></td>
                    <td className={riskClass(s?.riskLevel)}>
                      {s?.riskLevel ?? '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}

        {/* ── Info tab ──────────────────────────────────────────────── */}
        {activeTab === 'info' && (
          <table className={styles.table}>
            <thead>
              <tr>
                {th('ticker', 'Ticker', true)}
                <th>Mkt Cap</th>
                <th>P/E</th>
                <th>Beta</th>
                <th>Div Yield</th>
                <th>Exchange</th>
                <th style={{ textAlign: 'left' }}>Sector</th>
                <th style={{ textAlign: 'left' }}>Industry</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(h => {
                const s = stockMap[h.ticker]
                const f = s?.fundamentals
                return (
                  <tr key={h.id}>
                    <td>
                      <span
                        className={`${styles.ticker} ${styles.tickerLink}`}
                        onClick={() => navigate(`/stock/${h.ticker}`)}
                      >
                        {h.ticker}
                      </span>
                    </td>
                    <td>{fmtMarketCap(s?.MarketCap)}</td>
                    <td>{f?.pe != null ? f.pe.toFixed(1) : '—'}</td>
                    <td>{f?.beta != null ? f.beta.toFixed(2) : '—'}</td>
                    <td>{f?.divYield != null && f.divYield > 0
                      ? `${(f.divYield * 100).toFixed(2)}%` : '—'}
                    </td>
                    <td>{s?.Exchange ?? '—'}</td>
                    <td className={styles.infoLeft}>{f?.sector ?? '—'}</td>
                    <td className={styles.infoLeft}>{f?.industry ?? '—'}</td>
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
