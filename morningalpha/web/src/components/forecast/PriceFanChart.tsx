import Plot from 'react-plotly.js'
import type { TickerEntry } from '../../hooks/useTickerIndex'
import type { ForecastCalibration } from '../../hooks/useForecastCalibration'
import { addTradingDays } from '../../lib/dateUtils'
import styles from './PriceFanChart.module.css'

const SERIES_COLORS = ['#3B82F6', '#F59E0B', '#22C55E', '#A78BFA', '#EF4444']

const MODEL_LABELS: Record<string, string> = {
  'lgbm_breakout_v5': 'Breakout',
  'lgbm_composite_v6': 'Composite',
  'st_sector_relative_v1': 'Set Transformer',
}

const PLOT_CONFIG: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
}

interface ProjectionData {
  dates: Date[]
  median: number[]
  upper1: number[]
  lower1: number[]
  upper2: number[]
  lower2: number[]
}

function computeProjection(
  lastPrice: number,
  periodReturnMean: number,
  periodReturnStd: number,
  horizon: number
): ProjectionData {
  const dailyReturn = periodReturnMean / horizon
  const dailyStd = periodReturnStd / Math.sqrt(horizon)
  const dates: Date[] = []
  const median: number[] = []
  const upper1: number[] = []
  const lower1: number[] = []
  const upper2: number[] = []
  const lower2: number[] = []

  dates.push(new Date())
  median.push(lastPrice)
  upper1.push(lastPrice)
  lower1.push(lastPrice)
  upper2.push(lastPrice)
  lower2.push(lastPrice)

  for (let i = 1; i <= horizon; i++) {
    const date = addTradingDays(new Date(), i)
    const m = lastPrice * (1 + dailyReturn * i)
    const s = dailyStd * Math.sqrt(i) * lastPrice
    dates.push(date)
    median.push(m)
    upper1.push(m + s)
    lower1.push(m - s)
    upper2.push(m + 2 * s)
    lower2.push(m - 2 * s)
  }

  return { dates, median, upper1, lower1, upper2, lower2 }
}

// Median decile (D5 = index 4, 0-based) as the "consensus" decile for a given ticker score
function getDecileIndex(score: number | null): number {
  if (score === null) return 4 // default to middle decile
  // score 0-100 maps to decile 1-10
  return Math.min(9, Math.max(0, Math.floor(score / 10)))
}

interface Props {
  tickers: TickerEntry[]
  activeIndex: number
  selectedModels: string[]
  calibrations: Record<string, ForecastCalibration | null>
  horizon: 5 | 10 | 21 | 63
  showBands: boolean
  loadingModels: Record<string, boolean>
}

export default function PriceFanChart({
  tickers,
  activeIndex,
  selectedModels,
  calibrations,
  horizon,
  showBands,
  loadingModels,
}: Props) {
  const anyLoading = selectedModels.some(m => loadingModels[m])

  if (tickers.length === 0) {
    return (
      <div className={styles.emptyState}>
        Search for a stock above to view forecast
      </div>
    )
  }

  if (anyLoading) {
    return <div className={styles.loading}>Loading calibration data…</div>
  }

  const LAST_PRICE = 100
  const today = new Date()
  const todayStr = today.toISOString().split('T')[0]

  const plotData: Plotly.Data[] = []

  tickers.forEach((ticker, tickerIdx) => {
    const isActive = tickerIdx === activeIndex
    const seriesColor = SERIES_COLORS[tickerIdx % SERIES_COLORS.length]
    const opacity = isActive ? 1 : 0.3

    selectedModels.forEach((modelId, modelIdx) => {
      const cal = calibrations[modelId]
      if (!cal) return

      const horizonStr = String(horizon)
      const deciles = cal.horizons[horizonStr]
      if (!deciles || deciles.length === 0) return

      const decileIdx = getDecileIndex(ticker.mlScore)
      const stats = deciles[Math.min(decileIdx, deciles.length - 1)]
      if (!stats) return

      const proj = computeProjection(LAST_PRICE, stats.period_return_mean, stats.period_return_std, horizon)
      const dateStrs = proj.dates.map(d => d.toISOString().split('T')[0])

      const modelLabel = MODEL_LABELS[modelId] ?? modelId
      const lineName = `${ticker.ticker} · ${modelLabel}`

      // For multi-model, use slightly different line styles
      const dash = modelIdx === 0 ? 'solid' : modelIdx === 1 ? 'dash' : 'dot'

      // Bands: only for active ticker and first selected model
      if (isActive && showBands && modelIdx === 0) {
        // ±2σ fill
        plotData.push({
          type: 'scatter',
          x: [...dateStrs, ...dateStrs.slice().reverse()],
          y: [...proj.upper2, ...proj.lower2.slice().reverse()],
          fill: 'toself',
          fillcolor: `rgba(${hexToRgb(seriesColor)}, 0.06)`,
          line: { width: 0 },
          showlegend: false,
          hoverinfo: 'skip',
          name: '±2σ',
        })
        // ±1σ fill
        plotData.push({
          type: 'scatter',
          x: [...dateStrs, ...dateStrs.slice().reverse()],
          y: [...proj.upper1, ...proj.lower1.slice().reverse()],
          fill: 'toself',
          fillcolor: `rgba(${hexToRgb(seriesColor)}, 0.15)`,
          line: { width: 0 },
          showlegend: false,
          hoverinfo: 'skip',
          name: '±1σ',
        })
      }

      // Median line
      plotData.push({
        type: 'scatter',
        mode: 'lines',
        x: dateStrs,
        y: proj.median,
        name: lineName,
        line: {
          color: seriesColor,
          width: isActive ? 2 : 1,
          dash,
        },
        opacity,
        hovertemplate: `<b>${lineName}</b><br>%{x}: %{y:.2f}<extra></extra>`,
      })
    })
  })

  // TODAY line shape
  const todayShape: Partial<Plotly.Shape> = {
    type: 'line',
    x0: todayStr,
    x1: todayStr,
    y0: 0,
    y1: 1,
    yref: 'paper',
    line: { color: 'rgba(255,255,255,0.25)', width: 1, dash: 'dot' },
  }

  return (
    <div className={styles.wrap}>
      <Plot
        data={plotData}
        layout={{
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { family: "'IBM Plex Mono', monospace", color: '#5A6577', size: 10 },
          margin: { t: 16, r: 20, b: 50, l: 60 },
          hovermode: 'x unified',
          showlegend: true,
          legend: {
            bgcolor: 'rgba(0,0,0,0)',
            font: { family: "'IBM Plex Mono', monospace", size: 10, color: '#8B95A5' },
          },
          xaxis: {
            gridcolor: 'rgba(255,255,255,0.03)',
            linecolor: 'rgba(255,255,255,0.06)',
            tickcolor: 'rgba(255,255,255,0.06)',
            type: 'date',
          },
          yaxis: {
            gridcolor: 'rgba(255,255,255,0.03)',
            linecolor: 'rgba(255,255,255,0.06)',
            tickcolor: 'rgba(255,255,255,0.06)',
            title: { text: 'Normalized Price (base 100)', standoff: 8 },
          },
          shapes: [todayShape],
          annotations: [{
            x: todayStr,
            y: 1,
            yref: 'paper',
            text: 'Today',
            showarrow: false,
            font: { color: 'rgba(255,255,255,0.3)', size: 9, family: "'IBM Plex Mono', monospace" },
            xshift: 4,
            yshift: -8,
          }],
          height: 340,
        }}
        config={PLOT_CONFIG}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  )
}

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '')
  const r = parseInt(h.substring(0, 2), 16)
  const g = parseInt(h.substring(2, 4), 16)
  const b = parseInt(h.substring(4, 6), 16)
  return `${r}, ${g}, ${b}`
}
