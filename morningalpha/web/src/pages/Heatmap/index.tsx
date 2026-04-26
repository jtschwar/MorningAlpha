import { useDeferredValue, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useStock } from '../../store/StockContext'
import type { WindowPeriod } from '../../store/types'
import PlotlyChart from '../../components/charts/PlotlyChart'
import AppShell from '../../components/layout/AppShell'
import styles from './Heatmap.module.css'

type CapFilter = 'all' | 'large' | 'mid' | 'small'
type ExchangeFilter = 'all' | 'NASDAQ' | 'NYSE' | 'S&P500'

const CAP_LABELS: Record<CapFilter, string> = {
  all: 'All',
  large: 'Large Cap',
  mid: 'Mid Cap',
  small: 'Small Cap',
}

const EXCHANGE_LABELS: Record<ExchangeFilter, string> = {
  all: 'All',
  NASDAQ: 'NASDAQ',
  NYSE: 'NYSE',
  'S&P500': 'S&P 500',
}

const WINDOW_PERIODS: WindowPeriod[] = ['2w', '1m', '3m', '6m']
const WINDOW_LABELS: Record<WindowPeriod, string> = {
  '2w': '2W',
  '1m': '1M',
  '3m': '3M',
  '6m': '6M',
}

export default function HeatmapPage() {
  const { state } = useStock()
  const navigate = useNavigate()
  const [cap, setCap] = useState<CapFilter>('all')
  const [exchange, setExchange] = useState<ExchangeFilter>('all')
  const [window, setWindow] = useState<WindowPeriod>(state.activePeriod)

  const isLoading = state.autoLoadStatus === 'loading'
  const stocks = state.windowData[window] ?? []

  // Only show periods that have data loaded
  const availablePeriods = WINDOW_PERIODS.filter(p => (state.windowData[p] ?? []).length > 0)

  const filtered = useMemo(() => stocks.filter(s => {
    if (cap === 'large' && !['Mega', 'Large'].includes(s.MarketCapCategory ?? '')) return false
    if (cap === 'mid' && s.MarketCapCategory !== 'Mid') return false
    if (cap === 'small' && !['Small', 'Micro'].includes(s.MarketCapCategory ?? '')) return false
    if (exchange !== 'all' && s.Exchange !== exchange) return false
    return true
  }), [stocks, cap, exchange])

  // Cap at top 500 by market cap — smaller stocks are sub-pixel and only slow Plotly down
  const MAX_TILES = 500
  const topStocks = useMemo(() =>
    [...filtered].sort((a, b) => (b.MarketCap ?? 0) - (a.MarketCap ?? 0)).slice(0, MAX_TILES),
  [filtered])

  // Defer the expensive trace rebuild so filter clicks feel instant
  const deferredStocks = useDeferredValue(topStocks)
  const isStale = deferredStocks !== topStocks

  const traceData = useMemo(() => {
    const ids: string[] = []
    const labels: string[] = []
    const parents: string[] = []
    const values: number[] = []
    const colors: number[] = []
    const customdata: string[] = []
    const text: string[] = []

    const sectors = [...new Set(deferredStocks.map(s => s.fundamentals?.sector ?? 'Other'))]

    sectors.forEach(sector => {
      ids.push(sector)
      labels.push(sector)
      parents.push('')
      values.push(0)
      colors.push(0)
      customdata.push('')
      text.push('')
    })

    deferredStocks.forEach(s => {
      const sector = s.fundamentals?.sector ?? 'Other'
      const ret = s.ReturnPct
      ids.push(`${sector}||${s.Ticker}`)
      labels.push(s.Ticker)
      parents.push(sector)
      values.push(s.MarketCap ?? 1e8)
      colors.push(ret)
      customdata.push(s.Ticker)
      text.push(`${ret >= 0 ? '+' : ''}${ret.toFixed(2)}%`)
    })

    const stockColors = colors.filter(c => c !== 0)
    const minReturn = stockColors.length ? Math.min(...stockColors) : -100
    const maxReturn = stockColors.length ? Math.max(...stockColors) : 100
    const absMax = Math.min(100, Math.max(Math.abs(minReturn), maxReturn))

    return { ids, labels, parents, values, colors, customdata, text, cmin: -absMax, cmax: absMax }
  }, [deferredStocks])

  const trace = {
    type: 'treemap',
    ids: traceData.ids,
    labels: traceData.labels,
    parents: traceData.parents,
    values: traceData.values,
    text: traceData.text,
    customdata: traceData.customdata,
    texttemplate: '<b>%{label}</b><br>%{text}',
    hovertemplate: '<b>%{label}</b><br>Return: %{color:.2f}%<extra></extra>',
    marker: {
      colors: traceData.colors,
      colorscale: [
        [0, '#ef4444'],
        [0.5, '#334155'],
        [1, '#10b981'],
      ],
      cmin: traceData.cmin,
      cmax: traceData.cmax,
      showscale: true,
      colorbar: {
        title: { text: 'Return %' },
        thickness: 14,
        len: 0.6,
      },
    },
    branchvalues: 'remainder',
    tiling: { pad: 2 },
  } as Plotly.Data

  function onClick(event: Readonly<Plotly.PlotMouseEvent>) {
    const pt = event.points[0] as { customdata?: string }
    if (pt?.customdata) navigate(`/stock/${pt.customdata}`)
  }

  return (
    <AppShell showSidebar={false}>
      <div className={styles.page}>
        <div className={styles.header}>
          <h2 className={styles.title}>Market Heatmap</h2>
          <div className={styles.filters}>
            <div className={styles.filterGroup}>
              <span className={styles.filterLabel}>Cap</span>
              {(['all', 'large', 'mid', 'small'] as CapFilter[]).map(v => (
                <button
                  key={v}
                  className={`${styles.filterBtn} ${cap === v ? styles.active : ''}`}
                  onClick={() => setCap(v)}
                >
                  {CAP_LABELS[v]}
                </button>
              ))}
            </div>
            <div className={styles.filterGroup}>
              <span className={styles.filterLabel}>Exchange</span>
              {(['all', 'NASDAQ', 'NYSE', 'S&P500'] as ExchangeFilter[]).map(v => (
                <button
                  key={v}
                  className={`${styles.filterBtn} ${exchange === v ? styles.active : ''}`}
                  onClick={() => setExchange(v)}
                >
                  {EXCHANGE_LABELS[v]}
                </button>
              ))}
            </div>
            {availablePeriods.length > 1 && (
              <div className={styles.filterGroup}>
                <span className={styles.filterLabel}>Window</span>
                {availablePeriods.map(p => (
                  <button
                    key={p}
                    className={`${styles.filterBtn} ${window === p ? styles.active : ''}`}
                    onClick={() => setWindow(p)}
                  >
                    {WINDOW_LABELS[p]}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {isLoading ? (
          <div className={styles.loading}>
            <div className={styles.spinner} />
            <span>Building heatmap…</span>
          </div>
        ) : stocks.length === 0 ? (
          <div className={styles.empty}>Load a CSV from the screener to view the heatmap.</div>
        ) : filtered.length === 0 ? (
          <div className={styles.empty}>No stocks match the selected filters.</div>
        ) : (
          <>
            <p className={styles.countNote}>
              Showing top {Math.min(filtered.length, MAX_TILES).toLocaleString()} of {filtered.length.toLocaleString()} stocks by market cap
            </p>
            <div style={{ opacity: isStale ? 0.5 : 1, transition: 'opacity 0.15s' }}>
              <PlotlyChart
                data={[trace]}
                layout={{
                  margin: { t: 10, r: 20, b: 10, l: 10 },
                  height: 680,
                }}
                onClick={onClick}
              />
            </div>
          </>
        )}
      </div>
    </AppShell>
  )
}
