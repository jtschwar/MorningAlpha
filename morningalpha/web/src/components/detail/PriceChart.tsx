import PlotlyChart from '../charts/PlotlyChart'
import { calculateEMA } from '../../lib/technicals'
import { calculateBollingerBands } from '../../lib/indicators'
import type { StockDetailData } from '../../store/types'
import { colors } from '../../tokens/theme'

interface Props {
  data: StockDetailData
  ticker: string
  expanded?: boolean
}

export default function PriceChart({ data, ticker, expanded = false }: Props) {
  const dates = data.timestamps.map(ts => new Date(ts * 1000).toISOString().slice(0, 10))
  const closes = data.close
  const ema20 = calculateEMA(closes, 20)
  const ema50 = calculateEMA(closes, 50)

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

  // ── Simple (collapsed) view ───────────────────────────────────────────────

  if (!expanded) {
    return (
      <PlotlyChart
        data={[candlestick, ema20Trace, ema50Trace]}
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
      data={[bollUpperTrace, bollLowerTrace, bollMidTrace, candlestick, ema20Trace, ema50Trace]}
      layout={{
        title: { text: `${ticker} — Price + Bollinger Bands` },
        xaxis: { rangeslider: { visible: false }, type: 'date' },
        yaxis: { autorange: true, tickformat: '.2f', tickprefix: '$' },
        height: 420,
      }}
    />
  )
}
