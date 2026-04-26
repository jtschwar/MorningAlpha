import PlotlyChart from '../charts/PlotlyChart'
import { calculateEMA, calculateRSISeries } from '../../lib/technicals'
import { calculateBollingerBands } from '../../lib/indicators'
import type { StockDetailData } from '../../store/types'
import { colors } from '../../tokens/theme'

interface Props {
  data: StockDetailData
  ticker: string
  expanded?: boolean
}

function toDateStr(ts: number) {
  return new Date(ts * 1000).toISOString().slice(0, 10)
}

// RSI oversold recovery: RSI crosses above 30 from below
function findRsiSignals(dates: string[], lows: number[], rsi: (number | null)[]) {
  const x: string[] = [], y: number[] = [], text: string[] = []
  let last = -99
  for (let i = 1; i < rsi.length; i++) {
    const r0 = rsi[i - 1], r1 = rsi[i]
    if (r0 !== null && r1 !== null && r0 < 30 && r1 >= 30 && i - last > 10) {
      x.push(dates[i])
      y.push(lows[i] * 0.97)
      text.push('RSI Oversold Recovery — RSI crossed above 30')
      last = i
    }
  }
  return { x, y, text }
}

// EMA20 buy: shallow pullback to the 20-day EMA in an uptrend — aggressive entry
function findEma20Signals(dates: string[], closes: number[], lows: number[], ema20: (number | null)[]) {
  const x: string[] = [], y: number[] = [], text: string[] = []
  let last = -99
  for (let i = 1; i < closes.length; i++) {
    const e0 = ema20[i - 1], e1 = ema20[i]
    if (e0 === null || e1 === null) continue
    const wasTesting = closes[i - 1] <= e0 * 1.015
    const nowAbove = closes[i] > e1 * 1.01
    if (wasTesting && nowAbove && i - last > 5) {
      x.push(dates[i])
      y.push(lows[i] * 0.97)
      text.push('EMA20 Buy — shallow pullback held at 20-day EMA')
      last = i
    }
  }
  return { x, y, text }
}

// EMA50 support hold: price pulls back to within 2% of EMA50 and recovers
function findEma50Signals(dates: string[], closes: number[], highs: number[], ema50: (number | null)[]) {
  const x: string[] = [], y: number[] = [], text: string[] = []
  let last = -99
  for (let i = 1; i < closes.length; i++) {
    const e0 = ema50[i - 1], e1 = ema50[i]
    if (e0 === null || e1 === null) continue
    const wasTesting = closes[i - 1] <= e0 * 1.02
    const nowAbove = closes[i] > e1 * 1.01
    if (wasTesting && nowAbove && i - last > 10) {
      x.push(dates[i])
      y.push(highs[i] * 1.03)
      text.push('EMA50 Support Hold — price bounced from 50-day EMA')
      last = i
    }
  }
  return { x, y, text }
}

export default function PriceChart({ data, ticker, expanded = false }: Props) {
  const dates = data.timestamps.map(toDateStr)
  const closes = data.close
  const ema20 = calculateEMA(closes, 20)
  const ema50 = calculateEMA(closes, 50)
  const rsi14 = calculateRSISeries(closes, 14)

  const rsiSignals = findRsiSignals(dates, data.low, rsi14)
  const ema20Signals = findEma20Signals(dates, closes, data.low, ema20)
  const ema50Signals = findEma50Signals(dates, closes, data.high, ema50)

  // ── Shared traces ────────────────────────────────────────────────────────

  const candlestick: Plotly.Data = {
    type: 'candlestick',
    x: dates,
    open: data.open,
    high: data.high,
    low: data.low,
    close: closes,
    name: ticker,
    increasing: { line: { color: colors.accentGreen } },
    decreasing: { line: { color: colors.accentRed } },
  }

  const ema20Trace: Plotly.Data = {
    type: 'scatter',
    x: dates,
    y: ema20,
    mode: 'lines',
    name: 'EMA 20',
    line: { color: colors.accentBlue, width: 1.5 },
    connectgaps: false,
  }

  const ema50Trace: Plotly.Data = {
    type: 'scatter',
    x: dates,
    y: ema50,
    mode: 'lines',
    name: 'EMA 50',
    line: { color: colors.accentPurple, width: 1.5 },
    connectgaps: false,
  }

  const rsiOversoldTrace: Plotly.Data = {
    type: 'scatter',
    x: rsiSignals.x,
    y: rsiSignals.y,
    mode: 'markers',
    name: 'RSI Oversold',
    marker: { symbol: 'triangle-up', size: 12, color: colors.accentGreen, line: { color: '#fff', width: 1 } },
    text: rsiSignals.text,
    hovertemplate: '%{text}<br>%{x}<extra></extra>',
  }

  const ema20BuyTrace: Plotly.Data = {
    type: 'scatter',
    x: ema20Signals.x,
    y: ema20Signals.y,
    mode: 'markers',
    name: 'EMA20 Buy',
    marker: { symbol: 'triangle-up', size: 24, color: colors.accentGreen, line: { color: '#fff', width: 1.5 } },
    text: ema20Signals.text,
    hovertemplate: '%{text}<br>%{x}<extra></extra>',
  }

  const ema50BounceTrace: Plotly.Data = {
    type: 'scatter',
    x: ema50Signals.x,
    y: ema50Signals.y,
    mode: 'markers',
    name: 'EMA50 Hold',
    marker: { symbol: 'triangle-down', size: 24, color: colors.accentPurple, line: { color: '#fff', width: 1.5 } },
    text: ema50Signals.text,
    hovertemplate: '%{text}<br>%{x}<extra></extra>',
  }

  // ── Simple (collapsed) view ───────────────────────────────────────────────

  if (!expanded) {
    return (
      <PlotlyChart
        data={[candlestick, ema20Trace, ema50Trace, rsiOversoldTrace, ema20BuyTrace, ema50BounceTrace]}
        layout={{
          title: { text: `${ticker} — Price + EMAs` },
          xaxis: { rangeslider: { visible: false }, type: 'date' },
          yaxis: { autorange: true, tickformat: '.2f', tickprefix: '$' },
          height: 380,
        }}
      />
    )
  }

  // ── Expanded view: price + Bollinger Bands only (RSI/Volume/MACD rendered separately) ──

  const { upper: bollUpper, lower: bollLower, mid: bollMid } = calculateBollingerBands(closes, 20, 2)

  const bollUpperTrace: Plotly.Data = {
    type: 'scatter', x: dates, y: bollUpper, mode: 'lines', name: 'BB Upper',
    line: { color: colors.border, width: 1, dash: 'dot' }, connectgaps: false, showlegend: false,
  }
  const bollLowerTrace: Plotly.Data = {
    type: 'scatter', x: dates, y: bollLower, mode: 'lines', name: 'BB Lower',
    line: { color: colors.border, width: 1, dash: 'dot' },
    fill: 'tonexty', fillcolor: 'rgba(59, 130, 246, 0.06)', connectgaps: false, showlegend: false,
  }
  const bollMidTrace: Plotly.Data = {
    type: 'scatter', x: dates, y: bollMid, mode: 'lines', name: 'BB Mid',
    line: { color: colors.border, width: 1 }, connectgaps: false, showlegend: false,
  }

  return (
    <PlotlyChart
      data={[bollUpperTrace, bollLowerTrace, bollMidTrace, candlestick, ema20Trace, ema50Trace, rsiOversoldTrace, ema20BuyTrace, ema50BounceTrace]}
      layout={{
        title: { text: `${ticker} — Price + Bollinger Bands` },
        xaxis: { rangeslider: { visible: false }, type: 'date' },
        yaxis: { autorange: true, tickformat: '.2f', tickprefix: '$' },
        height: 420,
      }}
    />
  )
}
