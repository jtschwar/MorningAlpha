import { useState, useMemo } from 'react'
import { useParams } from 'react-router-dom'
import { useStock } from '../../store/StockContext'
import { useStockData } from '../../hooks/useStockData'
import { useFundamentals } from '../../hooks/useFundamentals'
import { computeSignal } from '../../lib/signal'
import AppShell from '../../components/layout/AppShell'
import StockHeader from '../../components/detail/StockHeader'
import SignalBanner from '../../components/detail/SignalBanner'
import DenseIndicatorGrid from '../../components/detail/DenseIndicatorGrid'
import PeriodSelector from '../../components/detail/PeriodSelector'
import type { DetailPeriod } from '../../components/detail/PeriodSelector'
import PriceChart from '../../components/detail/PriceChart'
import RsiChart from '../../components/detail/RsiChart'
import VolumeChart from '../../components/detail/VolumeChart'
import MacdChart from '../../components/detail/MacdChart'
import HelpDrawer from '../../components/common/HelpDrawer'
import styles from './StockDetail.module.css'

// Map DetailPeriod to the simpler Period type expected by DenseIndicatorGrid
// DetailPeriod = '1M' | '3M' | '6M' | '1Y' | '5Y' | 'MAX'
type GridPeriod = '1M' | '2W' | '3M' | '6M' | '1Y'

function toGridPeriod(p: DetailPeriod): GridPeriod {
  if (p === '1M') return '1M'
  if (p === '3M') return '3M'
  if (p === '6M') return '6M'
  if (p === '1Y') return '1Y'
  // '5Y' and 'MAX' fall back to '1Y'
  return '1Y'
}

export default function StockDetail() {
  const { ticker } = useParams<{ ticker: string }>()
  const { state } = useStock()
  const [period, setPeriod] = useState<DetailPeriod>('3M')
  const [expandedChart, setExpandedChart] = useState(true)

  // Find the stock in any loaded window
  const stock =
    Object.values(state.windowData)
      .flat()
      .find(s => s.Ticker === ticker) ?? null

  const { data, loading, error } = useStockData(ticker, period)
  const { data: proxyFundamentals } = useFundamentals(ticker)
  const meta = state.metadata[state.activePeriod]

  // Fundamentals are now embedded in the stock object from the period CSV
  const csvFundamentals = stock?.fundamentals ?? null

  const signal = useMemo(
    () => computeSignal(stock, proxyFundamentals, data),
    [stock, proxyFundamentals, data]
  )

  return (
    <AppShell showSidebar={false}>
      <div className={styles.page}>
        <StockHeader
          ticker={ticker ?? ''}
          stock={stock}
          metric={meta?.metric ?? '3M'}
          currentPrice={data ? data.close.at(-1) ?? null : null}
        />

        <SignalBanner signal={signal} />

        <DenseIndicatorGrid
          stock={stock}
          fundamentals={csvFundamentals}
          ohlcv={data ?? null}
          period={toGridPeriod(period)}
          section="overview"
        />

        <div className={styles.periodRow}>
          <PeriodSelector value={period} onChange={setPeriod} className={styles.periodSelectorInRow} />
          {data && !loading && (
            <button
              className={styles.toggleBtn}
              onClick={() => setExpandedChart(v => !v)}
            >
              {expandedChart ? 'Hide Full Technicals' : 'Show Full Technicals'}
            </button>
          )}
        </div>

        {loading && (
          <div className={styles.status}>Loading {ticker} ({period})&hellip;</div>
        )}

        {error && (
          <div className={styles.error}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {data && !loading && (
          <>
            <PriceChart data={data} ticker={ticker ?? ''} expanded={expandedChart} />
            <VolumeChart data={data} />
            <RsiChart data={data} />
            {expandedChart && <MacdChart data={data} />}
          </>
        )}

        <DenseIndicatorGrid
          stock={stock}
          fundamentals={csvFundamentals}
          ohlcv={data ?? null}
          period={toGridPeriod(period)}
          section="technicals"
        />

        {!data && !loading && !error && (
          <div className={styles.status}>
            Start the proxy server with <code>alpha launch</code> to load live data.
          </div>
        )}
      </div>
      <HelpDrawer />
    </AppShell>
  )
}
