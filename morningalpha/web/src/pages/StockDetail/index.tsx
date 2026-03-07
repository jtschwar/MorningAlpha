import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { useStock } from '../../store/StockContext'
import { useStockData } from '../../hooks/useStockData'
import AppShell from '../../components/layout/AppShell'
import StockHeader from '../../components/detail/StockHeader'
import CsvMetricsStrip from '../../components/detail/CsvMetricsStrip'
import PeriodSelector from '../../components/detail/PeriodSelector'
import type { DetailPeriod } from '../../components/detail/PeriodSelector'
import PriceChart from '../../components/detail/PriceChart'
import RsiChart from '../../components/detail/RsiChart'
import VolumeChart from '../../components/detail/VolumeChart'
import TechnicalsPanel from '../../components/detail/TechnicalsPanel'
import HelpDrawer from '../../components/common/HelpDrawer'
import styles from './StockDetail.module.css'

export default function StockDetail() {
  const { ticker } = useParams<{ ticker: string }>()
  const { state } = useStock()
  const [period, setPeriod] = useState<DetailPeriod>('3M')

  // Find the stock in any loaded window
  const stock =
    Object.values(state.windowData)
      .flat()
      .find(s => s.Ticker === ticker) ?? null

  const { data, loading, error } = useStockData(ticker, period)
  const meta = state.metadata[state.activePeriod]

  return (
    <AppShell showSidebar={false}>
      <div className={styles.page}>
        <StockHeader
          ticker={ticker ?? ''}
          stock={stock}
          metric={meta?.metric ?? '3M'}
        />

        {stock && <CsvMetricsStrip stock={stock} />}

        <PeriodSelector value={period} onChange={setPeriod} />

        {loading && (
          <div className={styles.status}>Loading {ticker} ({period})…</div>
        )}

        {error && (
          <div className={styles.error}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {data && !loading && (
          <>
            <PriceChart data={data} ticker={ticker ?? ''} />
            <RsiChart data={data} />
            <VolumeChart data={data} />
            <TechnicalsPanel data={data} />
          </>
        )}

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
