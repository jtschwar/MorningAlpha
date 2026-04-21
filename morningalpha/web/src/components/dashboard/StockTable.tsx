import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Stock, SortKey } from '../../store/types'
import { useStock } from '../../store/StockContext'
import { useKeyboardNav } from '../../hooks/useKeyboardNav'
import { ScoreBadge, RsiBadge, RiskBadge } from '../common/Badge'
import Pagination from './Pagination'
import EmptyState from './EmptyState'
import { exportStocks } from '../../lib/exportCsv'
import ColumnPicker, { PINNED_COLUMNS } from './ColumnPicker'
import styles from './StockTable.module.css'

const PAGE_SIZES = [25, 50, 100] as const

function fmt(v: number | null, dec = 2): string {
  return v == null ? '—' : v.toFixed(dec)
}

function fmtMarketCap(v: number | null): string {
  if (v == null) return '—'
  if (v >= 1e12) return `$${(v / 1e12).toFixed(1)}T`
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`
  if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}M`
  return `$${v.toFixed(0)}`
}

function fmtLargeNum(v: number): string {
  if (Math.abs(v) >= 1e9) return `${(v / 1e9).toFixed(1)}B`
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(1)}M`
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(1)}K`
  return v.toFixed(0)
}

function openHelp(section: string) {
  const drawer = (window as unknown as Record<string, { openSection: (s: string) => void }>).__helpDrawer
  drawer?.openSection(section)
}

interface ColDef {
  sortKey?: SortKey
  label: string
  helpSection?: string
  render: (s: Stock) => React.ReactNode
}

// Map column label -> definition with render function
// Each render() returns a <td> element
const COLUMN_DEFS: Record<string, ColDef> = {
  'Rank': {
    label: '#',
    render: s => <td key="Rank" className={styles.tdMono}>{s.Rank}</td>,
  },
  'Ticker': {
    label: 'Ticker',
    render: s => <td key="Ticker" className={styles.tdTicker}>{s.Ticker}</td>,
  },
  'Name': {
    label: 'Name',
    render: s => <td key="Name" className={styles.tdName}>{s.Name}</td>,
  },
  'Exchange': {
    label: 'Exch',
    render: s => <td key="Exchange" className={styles.tdMono}>{s.Exchange}</td>,
  },
  'Return %': {
    label: 'Return %',
    sortKey: 'return',
    helpSection: 'returns',
    render: s => (
      <td key="Return %" className={`${styles.tdMono} ${s.ReturnPct >= 0 ? styles.pos : styles.neg}`}>
        {s.ReturnPct >= 0 ? '+' : ''}{s.ReturnPct.toFixed(2)}%
      </td>
    ),
  },
  'Sharpe': {
    label: 'Sharpe',
    sortKey: 'sharpe',
    helpSection: 'risk',
    render: s => <td key="Sharpe" className={styles.tdMono}>{fmt(s.SharpeRatio)}</td>,
  },
  'Sortino': {
    label: 'Sortino',
    render: s => <td key="Sortino" className={styles.tdMono}>{fmt(s.SortinoRatio)}</td>,
  },
  'Quality': {
    label: 'Quality',
    sortKey: 'quality',
    helpSection: 'scores',
    render: s => <td key="Quality"><ScoreBadge value={s.QualityScore} /></td>,
  },
  'Entry': {
    label: 'Entry',
    sortKey: 'entryScore',
    helpSection: 'scores',
    render: s => <td key="Entry"><ScoreBadge value={s.EntryScore} /></td>,
  },
  'Max DD': {
    label: 'Max DD',
    sortKey: 'maxDrawdown',
    helpSection: 'risk',
    render: s => (
      <td key="Max DD" className={`${styles.tdMono} ${s.MaxDrawdown != null && s.MaxDrawdown < -20 ? styles.neg : ''}`}>
        {fmt(s.MaxDrawdown)}%
      </td>
    ),
  },
  'RSI': {
    label: 'RSI',
    helpSection: 'technicals',
    render: s => <td key="RSI"><RsiBadge value={s.RSI} /></td>,
  },
  'Score': {
    label: 'Score',
    sortKey: 'investmentScore',
    helpSection: 'scores',
    render: s => <td key="Score"><ScoreBadge value={s.investmentScore} /></td>,
  },
  'ML Score': {
    label: 'ML Score',
    sortKey: 'mlScore',
    helpSection: 'scores',
    render: s => <td key="ML Score"><ScoreBadge value={s.mlScore} /></td>,
  },
  '63d Brk%': {
    label: '63d Brk%',
    sortKey: 'breakoutProb63d',
    render: s => {
      const v = s.BreakoutProb63d
      const cls = v == null ? '' : v >= 0.6 ? styles.pos : v >= 0.4 ? styles.amber : ''
      return <td key="63d Brk%" className={`${styles.tdMono} ${cls}`}>{v != null ? `${(v * 100).toFixed(0)}%` : '—'}</td>
    },
  },
  '252d 50%': {
    label: '252d 50%',
    render: s => {
      const v = s.BreakoutProb252d50
      const cls = v == null ? '' : v >= 0.6 ? styles.pos : v >= 0.4 ? styles.amber : ''
      return <td key="252d 50%" className={`${styles.tdMono} ${cls}`}>{v != null ? `${(v * 100).toFixed(0)}%` : '—'}</td>
    },
  },
  '252d ×2': {
    label: '252d ×2',
    sortKey: 'breakoutProb252d100',
    render: s => {
      const v = s.BreakoutProb252d100
      const cls = v == null ? '' : v >= 0.5 ? styles.pos : v >= 0.3 ? styles.amber : ''
      return <td key="252d ×2" className={`${styles.tdMono} ${cls}`}>{v != null ? `${(v * 100).toFixed(0)}%` : '—'}</td>
    },
  },
  'Mkt Cap': {
    label: 'Mkt Cap',
    sortKey: 'marketCap',
    render: s => <td key="Mkt Cap" className={styles.tdMono}>{fmtMarketCap(s.MarketCap)}</td>,
  },
  'Risk': {
    label: 'Risk',
    render: s => <td key="Risk"><RiskBadge level={s.riskLevel} /></td>,
  },
  // Technical — Trend
  'SMA20 Dist%': {
    label: 'SMA20 Dist%',
    render: s => (
      <td key="SMA20 Dist%" className={`${styles.tdMono} ${s.PriceToSMA20Pct != null ? (s.PriceToSMA20Pct > 0 ? styles.pos : styles.neg) : ''}`}>
        {s.PriceToSMA20Pct != null ? `${s.PriceToSMA20Pct > 0 ? '+' : ''}${s.PriceToSMA20Pct.toFixed(1)}%` : '—'}
      </td>
    ),
  },
  'SMA50 Dist%': {
    label: 'SMA50 Dist%',
    render: s => (
      <td key="SMA50 Dist%" className={`${styles.tdMono} ${s.PriceToSMA50Pct != null ? (s.PriceToSMA50Pct > 0 ? styles.pos : styles.neg) : ''}`}>
        {s.PriceToSMA50Pct != null ? `${s.PriceToSMA50Pct > 0 ? '+' : ''}${s.PriceToSMA50Pct.toFixed(1)}%` : '—'}
      </td>
    ),
  },
  'SMA200 Dist%': {
    label: 'SMA200 Dist%',
    render: s => (
      <td key="SMA200 Dist%" className={`${styles.tdMono} ${s.PriceToSMA200Pct != null ? (s.PriceToSMA200Pct > 0 ? styles.pos : styles.neg) : ''}`}>
        {s.PriceToSMA200Pct != null ? `${s.PriceToSMA200Pct > 0 ? '+' : ''}${s.PriceToSMA200Pct.toFixed(1)}%` : '—'}
      </td>
    ),
  },
  'MACD': {
    label: 'MACD',
    render: s => (
      <td key="MACD" className={`${styles.tdMono} ${s.MACD != null ? (s.MACD > 0 ? styles.pos : styles.neg) : ''}`}>
        {s.MACD != null ? s.MACD.toFixed(3) : '—'}
      </td>
    ),
  },
  'MACD Hist': {
    label: 'MACD Hist',
    render: s => (
      <td key="MACD Hist" className={`${styles.tdMono} ${s.MACDHist != null ? (s.MACDHist > 0 ? styles.pos : styles.neg) : ''}`}>
        {s.MACDHist != null ? s.MACDHist.toFixed(3) : '—'}
      </td>
    ),
  },
  'EMA7': {
    label: 'EMA7',
    render: s => <td key="EMA7" className={styles.tdMono}>{s.EMA7 != null ? `$${s.EMA7.toFixed(2)}` : '—'}</td>,
  },
  'EMA200': {
    label: 'EMA200',
    render: s => <td key="EMA200" className={styles.tdMono}>{s.EMA200 != null ? `$${s.EMA200.toFixed(2)}` : '—'}</td>,
  },
  // Technical — Momentum
  'RSI(7)': {
    label: 'RSI(7)',
    render: s => <td key="RSI(7)"><RsiBadge value={s.RSI7} /></td>,
  },
  'RSI(21)': {
    label: 'RSI(21)',
    render: s => <td key="RSI(21)"><RsiBadge value={s.RSI21} /></td>,
  },
  'Stoch %K': {
    label: 'Stoch %K',
    render: s => <td key="Stoch %K" className={styles.tdMono}>{s.StochK != null ? s.StochK.toFixed(1) : '—'}</td>,
  },
  'Stoch %D': {
    label: 'Stoch %D',
    render: s => <td key="Stoch %D" className={styles.tdMono}>{s.StochD != null ? s.StochD.toFixed(1) : '—'}</td>,
  },
  'ROC(5)': {
    label: 'ROC(5)',
    render: s => (
      <td key="ROC(5)" className={`${styles.tdMono} ${s.ROC5 != null ? (s.ROC5 > 0 ? styles.pos : styles.neg) : ''}`}>
        {s.ROC5 != null ? `${s.ROC5 > 0 ? '+' : ''}${s.ROC5.toFixed(1)}%` : '—'}
      </td>
    ),
  },
  'ROC(10)': {
    label: 'ROC(10)',
    render: s => (
      <td key="ROC(10)" className={`${styles.tdMono} ${s.ROC10 != null ? (s.ROC10 > 0 ? styles.pos : styles.neg) : ''}`}>
        {s.ROC10 != null ? `${s.ROC10 > 0 ? '+' : ''}${s.ROC10.toFixed(1)}%` : '—'}
      </td>
    ),
  },
  'ROC(21)': {
    label: 'ROC(21)',
    render: s => (
      <td key="ROC(21)" className={`${styles.tdMono} ${s.ROC21 != null ? (s.ROC21 > 0 ? styles.pos : styles.neg) : ''}`}>
        {s.ROC21 != null ? `${s.ROC21 > 0 ? '+' : ''}${s.ROC21.toFixed(1)}%` : '—'}
      </td>
    ),
  },
  // Technical — Volatility
  'ATR': {
    label: 'ATR',
    render: s => <td key="ATR" className={styles.tdMono}>{s.ATR14 != null ? s.ATR14.toFixed(2) : '—'}</td>,
  },
  'Boll %B': {
    label: 'Boll %B',
    render: s => (
      <td key="Boll %B" className={`${styles.tdMono} ${s.BollingerPctB != null ? (s.BollingerPctB > 0.8 ? styles.neg : s.BollingerPctB < 0.2 ? styles.pos : '') : ''}`}>
        {s.BollingerPctB != null ? s.BollingerPctB.toFixed(2) : '—'}
      </td>
    ),
  },
  'Boll BW%': {
    label: 'Boll BW%',
    render: s => <td key="Boll BW%" className={styles.tdMono}>{s.BollingerBandwidth != null ? `${s.BollingerBandwidth.toFixed(1)}%` : '—'}</td>,
  },
  'Ann. Vol%': {
    label: 'Ann. Vol%',
    render: s => <td key="Ann. Vol%" className={styles.tdMono}>{s.AnnualizedVol != null ? `${s.AnnualizedVol.toFixed(1)}%` : '—'}</td>,
  },
  // Technical — Volume
  'Rel Volume': {
    label: 'Rel Vol',
    render: s => (
      <td key="Rel Volume" className={`${styles.tdMono} ${s.RelativeVolume != null && s.RelativeVolume > 1.5 ? styles.pos : ''}`}>
        {s.RelativeVolume != null ? `${s.RelativeVolume.toFixed(2)}x` : '—'}
      </td>
    ),
  },
  'Vol ROC%': {
    label: 'Vol ROC%',
    render: s => (
      <td key="Vol ROC%" className={`${styles.tdMono} ${s.VolumeROC != null ? (s.VolumeROC > 0 ? styles.pos : styles.neg) : ''}`}>
        {s.VolumeROC != null ? `${s.VolumeROC > 0 ? '+' : ''}${s.VolumeROC.toFixed(1)}%` : '—'}
      </td>
    ),
  },
  'OBV': {
    label: 'OBV',
    render: s => <td key="OBV" className={styles.tdMono}>{s.OBV != null ? fmtLargeNum(s.OBV) : '—'}</td>,
  },
  // Fundamental columns
  'Sector': {
    label: 'Sector',
    render: s => <td key="Sector" className={styles.tdMono}>{s.fundamentals?.sector || '—'}</td>,
  },
  'Industry': {
    label: 'Industry',
    render: s => <td key="Industry" className={styles.tdMono}>{s.fundamentals?.industry || '—'}</td>,
  },
  'P/E': {
    label: 'P/E',
    render: s => <td key="P/E" className={styles.tdMono}>{s.fundamentals?.pe != null ? s.fundamentals.pe.toFixed(1) : '—'}</td>,
  },
  'Fwd P/E': {
    label: 'Fwd P/E',
    render: s => <td key="Fwd P/E" className={styles.tdMono}>{s.fundamentals?.forwardPe != null ? s.fundamentals.forwardPe.toFixed(1) : '—'}</td>,
  },
  'P/B': {
    label: 'P/B',
    render: s => <td key="P/B" className={styles.tdMono}>{s.fundamentals?.pb != null ? s.fundamentals.pb.toFixed(1) : '—'}</td>,
  },
  'P/S': {
    label: 'P/S',
    render: s => <td key="P/S" className={styles.tdMono}>{s.fundamentals?.ps != null ? s.fundamentals.ps.toFixed(1) : '—'}</td>,
  },
  'PEG': {
    label: 'PEG',
    render: s => <td key="PEG" className={styles.tdMono}>{s.fundamentals?.peg != null ? s.fundamentals.peg.toFixed(2) : '—'}</td>,
  },
  'EPS': {
    label: 'EPS',
    render: s => <td key="EPS" className={styles.tdMono}>{s.fundamentals?.eps != null ? `$${s.fundamentals.eps.toFixed(2)}` : '—'}</td>,
  },
  'ROE': {
    label: 'ROE',
    render: s => (
      <td key="ROE" className={`${styles.tdMono} ${s.fundamentals?.roe != null ? (s.fundamentals.roe > 0 ? styles.pos : styles.neg) : ''}`}>
        {s.fundamentals?.roe != null ? `${(s.fundamentals.roe * 100).toFixed(1)}%` : '—'}
      </td>
    ),
  },
  'ROA': {
    label: 'ROA',
    render: s => (
      <td key="ROA" className={`${styles.tdMono} ${s.fundamentals?.roa != null ? (s.fundamentals.roa > 0 ? styles.pos : styles.neg) : ''}`}>
        {s.fundamentals?.roa != null ? `${(s.fundamentals.roa * 100).toFixed(1)}%` : '—'}
      </td>
    ),
  },
  'Gross Margin': {
    label: 'Gross Mgn',
    render: s => (
      <td key="Gross Margin" className={`${styles.tdMono} ${s.fundamentals?.grossMargin != null && s.fundamentals.grossMargin > 0.4 ? styles.pos : ''}`}>
        {s.fundamentals?.grossMargin != null ? `${(s.fundamentals.grossMargin * 100).toFixed(1)}%` : '—'}
      </td>
    ),
  },
  'Debt/Eq': {
    label: 'Debt/Eq',
    render: s => (
      <td key="Debt/Eq" className={`${styles.tdMono} ${s.fundamentals?.debtEquity != null && s.fundamentals.debtEquity > 200 ? styles.neg : ''}`}>
        {s.fundamentals?.debtEquity != null ? `${s.fundamentals.debtEquity.toFixed(0)}%` : '—'}
      </td>
    ),
  },
  'Div Yield': {
    label: 'Div Yield',
    render: s => (
      <td key="Div Yield" className={`${styles.tdMono} ${s.fundamentals?.divYield != null && s.fundamentals.divYield > 0 ? styles.pos : ''}`}>
        {s.fundamentals?.divYield != null && s.fundamentals.divYield > 0 ? `${(s.fundamentals.divYield * 100).toFixed(2)}%` : '—'}
      </td>
    ),
  },
  'Beta': {
    label: 'Beta',
    render: s => <td key="Beta" className={styles.tdMono}>{s.fundamentals?.beta != null ? s.fundamentals.beta.toFixed(2) : '—'}</td>,
  },
  'Short Float%': {
    label: 'Short %',
    render: s => (
      <td key="Short Float%" className={`${styles.tdMono} ${s.fundamentals?.shortFloat != null && s.fundamentals.shortFloat > 0.15 ? styles.neg : ''}`}>
        {s.fundamentals?.shortFloat != null ? `${(s.fundamentals.shortFloat * 100).toFixed(1)}%` : '—'}
      </td>
    ),
  },
}

interface Props {
  stocks: Stock[]
}

export default function StockTable({ stocks }: Props) {
  const { state, dispatch } = useStock()
  const navigate = useNavigate()
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState<number>(25)
  const [focusedIndex, setFocusedIndex] = useState(0)

  const totalPages = Math.ceil(stocks.length / pageSize)

  // Reset to page 1 when stocks or page size changes
  useEffect(() => { setPage(1) }, [stocks, pageSize])

  const pageStocks = stocks.slice((page - 1) * pageSize, page * pageSize)
  const globalFocused = focusedIndex - (page - 1) * pageSize

  useKeyboardNav(stocks, focusedIndex, setFocusedIndex)

  function handleSort(key: SortKey) {
    dispatch({ type: 'SET_FILTER', key: 'sortBy', value: key })
  }

  // Build active column list: pinned first, then visible (excluding pinned)
  const visibleColumns = state.columnConfig.visibleColumns
  const activeCols = [
    ...PINNED_COLUMNS,
    ...visibleColumns.filter(c => !PINNED_COLUMNS.includes(c)),
  ]

  if (stocks.length === 0) return <EmptyState />

  return (
    <div className={styles.wrapper}>
      <div className={styles.toolbar}>
        <span className={styles.count}>
          {stocks.length} stock{stocks.length !== 1 ? 's' : ''} · J/K to navigate
        </span>
        <div className={styles.toolbarRight}>
          <label className={styles.pageSizeLabel}>Rows</label>
          <select
            className={styles.pageSizeSelect}
            value={pageSize}
            onChange={e => setPageSize(Number(e.target.value))}
          >
            {PAGE_SIZES.map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
          <button
            className={styles.exportBtn}
            onClick={() => exportStocks(stocks, 'morningalpha_export.csv')}
          >
            Export CSV
          </button>
          <ColumnPicker />
        </div>
      </div>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              {activeCols.map(colName => {
                const def = COLUMN_DEFS[colName]
                if (!def) return null
                return (
                  <th
                    key={colName}
                    className={def.sortKey ? styles.sortable : ''}
                    onClick={() => def.sortKey && handleSort(def.sortKey)}
                  >
                    {def.label}
                    {def.sortKey && <span className={styles.sortIcon}>↕</span>}
                    {def.helpSection && (
                      <span
                        className={styles.infoIcon}
                        onClick={e => { e.stopPropagation(); openHelp(def.helpSection!) }}
                      >ⓘ</span>
                    )}
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {pageStocks.map((s, i) => (
              <tr
                key={s.Ticker}
                className={`${styles.row} ${i === globalFocused ? styles.focused : ''}`}
                onClick={() => {
                  setFocusedIndex((page - 1) * pageSize + i)
                  navigate(`/stock/${s.Ticker}`)
                }}
                role="button"
                tabIndex={0}
                onKeyDown={e => {
                  if (e.key === 'Enter') {
                    setFocusedIndex((page - 1) * pageSize + i)
                    navigate(`/stock/${s.Ticker}`)
                  }
                }}
              >
                {activeCols.map(colName => {
                  const def = COLUMN_DEFS[colName]
                  if (!def) return null
                  return def.render(s)
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Pagination
        page={page}
        totalPages={totalPages}
        onPage={setPage}
        totalRows={stocks.length}
        pageSize={pageSize}
      />
    </div>
  )
}
