import PlotlyChart from '../charts/PlotlyChart'
import { calculateEMA, calculateRSISeries } from '../../lib/technicals'
import { calculateMACD, calculateBollingerBands } from '../../lib/indicators'
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

  // ── Expanded view: 4 subplots ─────────────────────────────────────────────

  // Bollinger Bands (subplot 1)
  const { upper: bollUpper, lower: bollLower, mid: bollMid } = calculateBollingerBands(closes, 20, 2)

  const bollUpperTrace: Plotly.Data = {
    type: 'scatter',
    x: dates,
    y: bollUpper,
    mode: 'lines',
    name: 'BB Upper',
    line: { color: colors.border, width: 1, dash: 'dot' },
    connectgaps: false,
    xaxis: 'x',
    yaxis: 'y',
    showlegend: false,
  }

  // Fill between lower and upper using tonexty
  const bollLowerTrace: Plotly.Data = {
    type: 'scatter',
    x: dates,
    y: bollLower,
    mode: 'lines',
    name: 'BB Lower',
    line: { color: colors.border, width: 1, dash: 'dot' },
    fill: 'tonexty',
    fillcolor: 'rgba(59, 130, 246, 0.06)',
    connectgaps: false,
    xaxis: 'x',
    yaxis: 'y',
    showlegend: false,
  }

  const bollMidTrace: Plotly.Data = {
    type: 'scatter',
    x: dates,
    y: bollMid,
    mode: 'lines',
    name: 'BB Mid',
    line: { color: colors.border, width: 1 },
    connectgaps: false,
    xaxis: 'x',
    yaxis: 'y',
    showlegend: false,
  }

  const candlestickSub1: Plotly.Data = { ...candlestick, xaxis: 'x', yaxis: 'y' }
  const ema20Sub1: Plotly.Data = { ...ema20Trace, xaxis: 'x', yaxis: 'y' }
  const ema50Sub1: Plotly.Data = { ...ema50Trace, xaxis: 'x', yaxis: 'y' }

  // Volume bars (subplot 2)
  const volColors = dates.map((_, i) => {
    if (i === 0) return colors.accentGreen
    return closes[i] >= closes[i - 1] ? colors.accentGreen : colors.accentRed
  })

  const volumeTrace: Plotly.Data = {
    type: 'bar',
    x: dates,
    y: data.volume,
    name: 'Volume',
    marker: { color: volColors, opacity: 0.8 },
    xaxis: 'x2',
    yaxis: 'y2',
  }

  // RSI(14) (subplot 3)
  const rsi14Series = calculateRSISeries(closes, 14)

  const rsiTrace: Plotly.Data = {
    type: 'scatter',
    x: dates,
    y: rsi14Series,
    mode: 'lines',
    name: 'RSI 14',
    line: { color: colors.accentBlue, width: 1.5 },
    connectgaps: false,
    xaxis: 'x3',
    yaxis: 'y3',
  }

  const rsiOverbought: Plotly.Data = {
    type: 'scatter',
    x: [dates[0], dates[dates.length - 1]],
    y: [70, 70],
    mode: 'lines',
    name: 'Overbought',
    line: { color: colors.accentRed, width: 1, dash: 'dash' },
    xaxis: 'x3',
    yaxis: 'y3',
    showlegend: false,
  }

  const rsiOversold: Plotly.Data = {
    type: 'scatter',
    x: [dates[0], dates[dates.length - 1]],
    y: [30, 30],
    mode: 'lines',
    name: 'Oversold',
    line: { color: colors.accentGreen, width: 1, dash: 'dash' },
    xaxis: 'x3',
    yaxis: 'y3',
    showlegend: false,
  }

  // MACD (subplot 4)
  const { macd, signal: macdSignal, histogram } = calculateMACD(closes)

  const macdLine: Plotly.Data = {
    type: 'scatter',
    x: dates,
    y: macd,
    mode: 'lines',
    name: 'MACD',
    line: { color: colors.accentBlue, width: 1.5 },
    connectgaps: false,
    xaxis: 'x4',
    yaxis: 'y4',
  }

  const macdSignalLine: Plotly.Data = {
    type: 'scatter',
    x: dates,
    y: macdSignal,
    mode: 'lines',
    name: 'Signal',
    line: { color: '#f97316', width: 1.5 },
    connectgaps: false,
    xaxis: 'x4',
    yaxis: 'y4',
  }

  const histColors = histogram.map(v => {
    if (v == null) return colors.accentGreen
    return v >= 0 ? colors.accentGreen : colors.accentRed
  })

  const macdHistogram: Plotly.Data = {
    type: 'bar',
    x: dates,
    y: histogram,
    name: 'Histogram',
    marker: { color: histColors, opacity: 0.7 },
    xaxis: 'x4',
    yaxis: 'y4',
  }

  const macdZero: Plotly.Data = {
    type: 'scatter',
    x: [dates[0], dates[dates.length - 1]],
    y: [0, 0],
    mode: 'lines',
    name: 'Zero',
    line: { color: colors.textSecondary, width: 1 },
    xaxis: 'x4',
    yaxis: 'y4',
    showlegend: false,
  }

  const expandedLayout: Partial<Plotly.Layout> = {
    height: 700,
    title: { text: `${ticker} — Full Technicals` },
    // Subplot x-axes
    xaxis: { domain: [0, 1], rangeslider: { visible: false }, type: 'date', anchor: 'y' },
    xaxis2: { matches: 'x', showticklabels: false, anchor: 'y2' },
    xaxis3: { matches: 'x', showticklabels: false, anchor: 'y3' },
    xaxis4: { matches: 'x', anchor: 'y4' },
    // Subplot y-axes with domains
    yaxis: { domain: [0.52, 1.0], tickformat: '.2f', tickprefix: '$', anchor: 'x' },
    yaxis2: { domain: [0.37, 0.50], anchor: 'x2', showticklabels: false },
    yaxis3: { domain: [0.20, 0.35], anchor: 'x3', range: [0, 100], tickvals: [30, 50, 70] },
    yaxis4: { domain: [0.0, 0.18], anchor: 'x4' },
    // Annotations for subplot labels
    annotations: [
      { text: 'Volume', x: 0, y: 0.435, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 10, color: colors.textSecondary }, xanchor: 'left' },
      { text: 'RSI 14', x: 0, y: 0.30, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 10, color: colors.textSecondary }, xanchor: 'left' },
      { text: 'MACD', x: 0, y: 0.16, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 10, color: colors.textSecondary }, xanchor: 'left' },
    ],
    margin: { t: 40, r: 20, b: 40, l: 60 },
    showlegend: true,
    legend: { y: 1.02, x: 0, orientation: 'h', bgcolor: 'transparent', font: { color: colors.textSecondary } },
  }

  return (
    <PlotlyChart
      data={[
        bollUpperTrace,
        bollLowerTrace,
        bollMidTrace,
        candlestickSub1,
        ema20Sub1,
        ema50Sub1,
        volumeTrace,
        rsiTrace,
        rsiOverbought,
        rsiOversold,
        macdZero,
        macdHistogram,
        macdLine,
        macdSignalLine,
      ]}
      layout={expandedLayout}
    />
  )
}
