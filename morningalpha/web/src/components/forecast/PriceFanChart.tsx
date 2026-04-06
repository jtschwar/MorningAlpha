import Plot from 'react-plotly.js'
import type { TickerEntry } from '../../hooks/useTickerIndex'
import type { ForecastCalibration } from '../../hooks/useForecastCalibration'
import type { ForecastData } from '../../hooks/useForecast'
import { addTradingDays } from '../../lib/dateUtils'
import styles from './PriceFanChart.module.css'

const SERIES_COLORS = ['#3B82F6', '#F59E0B', '#22C55E', '#A78BFA', '#EF4444']
const PATH_PALETTE  = ['#4e8df5', '#f5934e', '#4ef574', '#f54e4e', '#b04ef5', '#f5e04e',
                       '#4ef5e8', '#f54eb0', '#9af54e', '#f5c84e']

const MODEL_LABELS: Record<string, string> = {
  'lgbm_breakout_v7':  'Breakout',
  'lgbm_composite_v7': 'Composite',
  'lstm_clip_v1':      'LSTM',
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
  horizon: number,
): ProjectionData {
  const dailyReturn = periodReturnMean / horizon
  const dailyStd    = periodReturnStd / Math.sqrt(horizon)
  const dates: Date[]    = []
  const median: number[] = []
  const upper1: number[] = []
  const lower1: number[] = []
  const upper2: number[] = []
  const lower2: number[] = []

  dates.push(new Date())
  median.push(lastPrice)
  upper1.push(lastPrice); lower1.push(lastPrice)
  upper2.push(lastPrice); lower2.push(lastPrice)

  for (let i = 1; i <= horizon; i++) {
    const date = addTradingDays(new Date(), i)
    const m    = lastPrice * (1 + dailyReturn * i)
    const s    = dailyStd * Math.sqrt(i) * lastPrice
    dates.push(date)
    median.push(m)
    upper1.push(m + s);     lower1.push(m - s)
    upper2.push(m + 2 * s); lower2.push(m - 2 * s)
  }
  return { dates, median, upper1, lower1, upper2, lower2 }
}

function getDecileIndex(score: number | null): number {
  if (score === null) return 4
  return Math.min(9, Math.max(0, Math.floor(score / 10)))
}

function toDateStr(d: Date): string {
  return d.toISOString().split('T')[0]
}

interface Props {
  tickers: TickerEntry[]
  activeIndex: number
  selectedModels: string[]
  calibrations: Record<string, ForecastCalibration | null>
  forecastPaths: Record<string, ForecastData | null>   // keyed by ticker symbol
  forecastLoading: Record<string, boolean>
  horizon: 5 | 10 | 21 | 63
  showBands: boolean
  loadingModels: Record<string, boolean>
}

export default function PriceFanChart({
  tickers,
  activeIndex,
  selectedModels,
  calibrations,
  forecastPaths,
  forecastLoading,
  horizon,
  showBands,
  loadingModels,
}: Props) {
  const lstmSelected   = selectedModels.includes('lstm_clip_v1')
  const lgbmSelected   = selectedModels.some(m => m !== 'lstm_clip_v1')
  const anyCalLoading  = selectedModels.filter(m => m !== 'lstm_clip_v1').some(m => loadingModels[m])
  const anyPathLoading = lstmSelected && tickers.some(t => forecastLoading[t.ticker])

  if (tickers.length === 0) {
    return (
      <div className={styles.emptyState}>
        Search for a stock above to view forecast
      </div>
    )
  }

  if (anyCalLoading || anyPathLoading) {
    return <div className={styles.loading}>Loading forecast data…</div>
  }

  const today    = new Date()
  const todayStr = toDateStr(today)
  const BASE     = 100

  const plotData: Plotly.Data[] = []

  // -----------------------------------------------------------------------
  // LGBM calibration bands (existing behaviour)
  // -----------------------------------------------------------------------
  if (lgbmSelected) {
    tickers.forEach((ticker, tickerIdx) => {
      const isActive    = tickerIdx === activeIndex
      const seriesColor = SERIES_COLORS[tickerIdx % SERIES_COLORS.length]
      const opacity     = isActive ? 1 : 0.3

      selectedModels
        .filter(m => m !== 'lstm_clip_v1')
        .forEach((modelId, modelIdx) => {
          const cal = calibrations[modelId]
          if (!cal) return
          const deciles = cal.horizons[String(horizon)]
          if (!deciles || deciles.length === 0) return
          const stats = deciles[Math.min(getDecileIndex(ticker.mlScore), deciles.length - 1)]
          if (!stats) return

          const proj     = computeProjection(BASE, stats.period_return_mean, stats.period_return_std, horizon)
          const dateStrs = proj.dates.map(toDateStr)
          const label    = `${ticker.ticker} · ${MODEL_LABELS[modelId] ?? modelId}`
          const dash     = modelIdx === 0 ? 'solid' : modelIdx === 1 ? 'dash' : 'dot'

          if (isActive && showBands && modelIdx === 0) {
            plotData.push({
              type: 'scatter',
              x: [...dateStrs, ...dateStrs.slice().reverse()],
              y: [...proj.upper2, ...proj.lower2.slice().reverse()],
              fill: 'toself',
              fillcolor: `rgba(${hexToRgb(seriesColor)}, 0.06)`,
              line: { width: 0 },
              showlegend: false,
              hoverinfo: 'skip',
            })
            plotData.push({
              type: 'scatter',
              x: [...dateStrs, ...dateStrs.slice().reverse()],
              y: [...proj.upper1, ...proj.lower1.slice().reverse()],
              fill: 'toself',
              fillcolor: `rgba(${hexToRgb(seriesColor)}, 0.15)`,
              line: { width: 0 },
              showlegend: false,
              hoverinfo: 'skip',
            })
          }

          plotData.push({
            type: 'scatter', mode: 'lines',
            x: dateStrs,
            y: proj.median,
            name: label,
            line: { color: seriesColor, width: isActive ? 2 : 1, dash },
            opacity,
            hovertemplate: `<b>${label}</b><br>%{x}: %{y:.1f}<extra></extra>`,
          })
        })
    })
  }

  // -----------------------------------------------------------------------
  // LSTM MC dropout paths
  // -----------------------------------------------------------------------
  if (lstmSelected) {
    tickers.forEach((ticker, tickerIdx) => {
      const fd       = forecastPaths[ticker.ticker]
      const isActive = tickerIdx === activeIndex
      const opacity  = isActive ? 1 : 0.25

      if (!fd || fd.paths.length === 0) return

      // horizon dates: [today, +1d, +5d, +10d, +21d, +63d]
      const horizonDates = [today, ...fd.horizons.map(h => addTradingDays(today, h))]
      const horizonStrs  = horizonDates.map(toDateStr)

      // Normalized to base 100 so multiple tickers are comparable
      // path values are cumulative log-returns → base * exp(log_ret)
      fd.paths.forEach((path, pathIdx) => {
        const prices = [BASE, ...path.map(lr => BASE * Math.exp(lr))]
        const color  = PATH_PALETTE[pathIdx % PATH_PALETTE.length]
        const name   = pathIdx === 0 ? `${ticker.ticker} · LSTM` : undefined

        plotData.push({
          type: 'scatter', mode: 'lines+markers',
          x: horizonStrs,
          y: prices,
          name: name ?? `${ticker.ticker} · LSTM path ${pathIdx + 1}`,
          showlegend: pathIdx === 0,
          line: { color, width: isActive ? 1.5 : 0.8 },
          marker: { size: 3, color },
          opacity,
          hovertemplate: pathIdx === 0
            ? `<b>${ticker.ticker} LSTM</b><br>%{x}: %{y:.1f}<extra></extra>`
            : `<extra></extra>`,
        })
      })

      // Median path — bold overlay
      if (fd.paths.length > 1) {
        const medianPrices = [BASE, ...fd.horizons.map((_, hi) => {
          const vals = fd.paths.map(p => BASE * Math.exp(p[hi]))
          vals.sort((a, b) => a - b)
          return vals[Math.floor(vals.length / 2)]
        })]
        plotData.push({
          type: 'scatter', mode: 'lines',
          x: horizonStrs,
          y: medianPrices,
          name: `${ticker.ticker} · LSTM median`,
          line: { color: SERIES_COLORS[tickerIdx % SERIES_COLORS.length], width: isActive ? 2.5 : 1.5, dash: 'solid' },
          opacity,
          hovertemplate: `<b>${ticker.ticker} LSTM median</b><br>%{x}: %{y:.1f}<extra></extra>`,
        })
      }
    })
  }

  const todayShape: Partial<Plotly.Shape> = {
    type: 'line',
    x0: todayStr, x1: todayStr, y0: 0, y1: 1, yref: 'paper',
    line: { color: 'rgba(255,255,255,0.25)', width: 1, dash: 'dot' },
  }

  const yAxisLabel = lstmSelected && !lgbmSelected
    ? 'Normalized (base 100)'
    : 'Normalized Price (base 100)'

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
            title: { text: yAxisLabel, standoff: 8 },
          },
          shapes: [todayShape],
          annotations: [{
            x: todayStr, y: 1, yref: 'paper',
            text: 'Today', showarrow: false,
            font: { color: 'rgba(255,255,255,0.3)', size: 9, family: "'IBM Plex Mono', monospace" },
            xshift: 4, yshift: -8,
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
