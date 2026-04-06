import { useState, useMemo, useEffect, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import AppShell from '../../components/layout/AppShell'
import TickerSearch from '../../components/forecast/TickerSearch'
import StockChips from '../../components/forecast/StockChips'
import ModelControls from '../../components/forecast/ModelControls'
import ForecastKPIStrip from '../../components/forecast/ForecastKPIStrip'
import PriceFanChart from '../../components/forecast/PriceFanChart'
import ModelComparisonTable from '../../components/forecast/ModelComparisonTable'
import DecileContextChart from '../../components/forecast/DecileContextChart'
import CalibrationDisclaimer from '../../components/forecast/CalibrationDisclaimer'
import { useForecastCalibration } from '../../hooks/useForecastCalibration'
import { useForecast } from '../../hooks/useForecast'
import { useTickerIndex, type TickerEntry } from '../../hooks/useTickerIndex'
import type { DecileStats } from '../../hooks/useForecastCalibration'
import styles from './Forecast.module.css'

// Fixed model IDs — hooks are always called for all 3 (React rules)
const ALL_MODEL_IDS = [
  'lgbm_breakout_v7',
  'lgbm_composite_v7',
  'lstm_clip_v1',
] as const

// Inner component that always calls all 3 calibration hooks
function ForecastInner() {
  const [searchParams, setSearchParams] = useSearchParams()
  const { tickers: tickerIndex } = useTickerIndex()

  const [selectedTickers, setSelectedTickers] = useState<TickerEntry[]>([])
  const [activeIndex, setActiveIndex] = useState<number>(0)
  const [selectedModels, setSelectedModels] = useState<string[]>([
    'lgbm_breakout_v7',
    'lgbm_composite_v7',
    'lstm_clip_v1',
  ])
  const [horizon, setHorizon] = useState<5 | 10 | 21 | 63>(63)
  const [showBands, setShowBands] = useState(false)

  // Initialize from URL params once ticker index loads
  const urlInitialized = useRef(false)
  useEffect(() => {
    if (urlInitialized.current || tickerIndex.length === 0) return
    urlInitialized.current = true

    const urlTickers = searchParams.getAll('ticker')
    const urlHorizon = Number(searchParams.get('horizon'))

    if (urlTickers.length > 0) {
      const found = urlTickers
        .map(t => tickerIndex.find(e => e.ticker === t.toUpperCase()))
        .filter(Boolean) as TickerEntry[]
      if (found.length > 0) setSelectedTickers(found.slice(0, 5))
    }
    if ([5, 10, 21, 63].includes(urlHorizon)) {
      setHorizon(urlHorizon as 5 | 10 | 21 | 63)
    }
  }, [tickerIndex])

  // Keep URL in sync whenever tickers or horizon change
  useEffect(() => {
    if (!urlInitialized.current) return
    const params = new URLSearchParams()
    selectedTickers.forEach(t => params.append('ticker', t.ticker))
    if (horizon !== 63) params.set('horizon', String(horizon))
    setSearchParams(params, { replace: true })
  }, [selectedTickers, horizon])

  // Always call all 3 calibration hooks (React rules of hooks)
  const cal0 = useForecastCalibration(ALL_MODEL_IDS[0])
  const cal1 = useForecastCalibration(ALL_MODEL_IDS[1])
  const cal2 = useForecastCalibration(ALL_MODEL_IDS[2])

  // Always call useForecast for all 5 ticker slots (React rules of hooks)
  // Only fires when LSTM is selected and the slot is populated
  const lstmEnabled = selectedModels.includes('lstm_clip_v1')
  const fp0 = useForecast(selectedTickers[0]?.ticker, lstmEnabled)
  const fp1 = useForecast(selectedTickers[1]?.ticker, lstmEnabled)
  const fp2 = useForecast(selectedTickers[2]?.ticker, lstmEnabled)
  const fp3 = useForecast(selectedTickers[3]?.ticker, lstmEnabled)
  const fp4 = useForecast(selectedTickers[4]?.ticker, lstmEnabled)

  const allCals = [cal0, cal1, cal2]

  const calibrations = useMemo(() => {
    const map: Record<string, typeof cal0.calibration> = {}
    ALL_MODEL_IDS.forEach((id, i) => {
      map[id] = allCals[i].calibration
    })
    return map
  }, [cal0.calibration, cal1.calibration, cal2.calibration])

  const loadingModels = useMemo(() => {
    const map: Record<string, boolean> = {}
    ALL_MODEL_IDS.forEach((id, i) => {
      map[id] = allCals[i].loading
    })
    return map
  }, [cal0.loading, cal1.loading, cal2.loading])

  // Build forecast path maps keyed by ticker symbol
  const forecastPaths = useMemo(() => {
    const map: Record<string, typeof fp0.data> = {}
    ;[fp0, fp1, fp2, fp3, fp4].forEach((fp, i) => {
      const ticker = selectedTickers[i]?.ticker
      if (ticker) map[ticker] = fp.data
    })
    return map
  }, [fp0.data, fp1.data, fp2.data, fp3.data, fp4.data, selectedTickers])

  const forecastLoading = useMemo(() => {
    const map: Record<string, boolean> = {}
    ;[fp0, fp1, fp2, fp3, fp4].forEach((fp, i) => {
      const ticker = selectedTickers[i]?.ticker
      if (ticker) map[ticker] = fp.loading
    })
    return map
  }, [fp0.loading, fp1.loading, fp2.loading, fp3.loading, fp4.loading, selectedTickers])

  // Primary calibration = first selected model
  const primaryModelId = selectedModels[0] ?? ALL_MODEL_IDS[0]
  const primaryCal = calibrations[primaryModelId] ?? null
  const primaryLoading = loadingModels[primaryModelId] ?? false

  // Active ticker
  const activeTicker = selectedTickers[activeIndex] ?? null

  // Decile stats for the active ticker + primary model + selected horizon
  const activeDecileStats = useMemo((): DecileStats | null => {
    if (!activeTicker || !primaryCal) return null
    const deciles = primaryCal.horizons[String(horizon)]
    if (!deciles || deciles.length === 0) return null
    const decileIdx = Math.min(9, Math.max(0, Math.floor((activeTicker.mlScore ?? 50) / 10)))
    return deciles[Math.min(decileIdx, deciles.length - 1)] ?? null
  }, [activeTicker, primaryCal, horizon])

  function handleAdd(entry: TickerEntry) {
    setSelectedTickers(prev => {
      if (prev.length >= 5) return prev
      if (prev.some(t => t.ticker === entry.ticker)) return prev
      return [...prev, entry]
    })
  }

  function handleRemove(index: number) {
    setSelectedTickers(prev => prev.filter((_, i) => i !== index))
    setActiveIndex(prev => Math.min(prev, Math.max(0, selectedTickers.length - 2)))
  }

  function handleToggleModel(id: string) {
    setSelectedModels(prev =>
      prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]
    )
  }

  function handleActivate(index: number) {
    setActiveIndex(index)
  }

  return (
    <AppShell showSidebar={false}>
      <div className={styles.page}>
        <div className={styles.pageIntro}>
          <p className={styles.introDesc}>
            Compare ML-driven price forecasts across up to 5 tickers. The fan chart shows
            historical percentile bands derived from backtest-calibrated models — not a
            prediction, but a map of how similar setups have resolved. Use the decile context
            and KPI strip to understand each ticker's score tier and expected win rate at your
            chosen horizon.
          </p>
        </div>

        {/* Search + chips row */}
        <div className={styles.searchRow}>
          <TickerSearch selectedTickers={selectedTickers} onAdd={handleAdd} />
          <StockChips
            tickers={selectedTickers}
            activeIndex={activeIndex}
            onActivate={handleActivate}
            onRemove={handleRemove}
          />
        </div>

        {/* Model controls */}
        <ModelControls
          selectedModels={selectedModels}
          horizon={horizon}
          showBands={showBands}
          onToggleModel={handleToggleModel}
          onSetHorizon={setHorizon}
          onToggleBands={() => setShowBands(v => !v)}
        />

        {/* Price fan chart */}
        <PriceFanChart
          tickers={selectedTickers}
          activeIndex={activeIndex}
          selectedModels={selectedModels}
          calibrations={calibrations}
          forecastPaths={forecastPaths}
          forecastLoading={forecastLoading}
          horizon={horizon}
          showBands={showBands}
          loadingModels={loadingModels}
        />

        {/* KPI strip */}
        <ForecastKPIStrip
          ticker={activeTicker}
          horizon={horizon}
          decileStats={activeDecileStats}
        />

        {/* Bottom split row */}
        <div className={styles.splitRow}>
          <div className={styles.splitLeft}>
            <ModelComparisonTable tickers={selectedTickers} calibration={primaryCal} />
          </div>
          <div className={styles.splitRight}>
            <DecileContextChart
              ticker={activeTicker}
              calibration={primaryCal}
              loading={primaryLoading}
            />
          </div>
        </div>

        {/* Disclaimer */}
        <CalibrationDisclaimer calibration={primaryCal} />
      </div>
    </AppShell>
  )
}

export default function ForecastPage() {
  return <ForecastInner />
}
