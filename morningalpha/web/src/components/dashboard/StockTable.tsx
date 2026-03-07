import { useState, useEffect } from 'react'
import type { Stock, SortKey } from '../../store/types'
import { useStock } from '../../store/StockContext'
import { useKeyboardNav } from '../../hooks/useKeyboardNav'
import TableRow from './TableRow'
import Pagination from './Pagination'
import EmptyState from './EmptyState'
import { exportStocks } from '../../lib/exportCsv'
import styles from './StockTable.module.css'

const PAGE_SIZES = [25, 50, 100] as const

interface ColDef {
  key: SortKey | null
  label: string
  title?: string
}

const COLS: ColDef[] = [
  { key: null, label: '#' },
  { key: null, label: 'Ticker' },
  { key: null, label: 'Name' },
  { key: null, label: 'Exch' },
  { key: 'return', label: 'Return %' },
  { key: 'sharpe', label: 'Sharpe' },
  { key: 'quality', label: 'Quality', title: 'Composite quality score (0–100)' },
  { key: 'entryScore', label: 'Entry', title: 'Short-term entry timing score' },
  { key: null, label: 'Max DD', title: 'Maximum drawdown during period' },
  { key: null, label: 'RSI', title: '14-day Relative Strength Index' },
  { key: 'marketCap', label: 'Mkt Cap' },
  { key: null, label: 'Risk' },
  { key: 'investmentScore', label: 'Score', title: 'Composite investment score (0–100)' },
]

interface Props {
  stocks: Stock[]
}

export default function StockTable({ stocks }: Props) {
  const { dispatch } = useStock()
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
        </div>
      </div>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              {COLS.map((c, i) => (
                <th
                  key={i}
                  className={c.key ? styles.sortable : ''}
                  title={c.title}
                  onClick={() => c.key && handleSort(c.key)}
                >
                  {c.label}
                  {c.key && <span className={styles.sortIcon}>↕</span>}
                  {c.title && <span className={styles.infoIcon}>ⓘ</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageStocks.map((s, i) => (
              <TableRow
                key={s.Ticker}
                stock={s}
                focused={i === globalFocused}
                onClick={() => setFocusedIndex((page - 1) * pageSize + i)}
              />
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
