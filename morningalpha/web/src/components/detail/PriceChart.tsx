import PlotlyChart from '../charts/PlotlyChart'
import { calculateEMA } from '../../lib/technicals'
import type { StockDetailData } from '../../store/types'
import { colors } from '../../tokens/theme'

interface Props {
  data: StockDetailData
  ticker: string
}

export default function PriceChart({ data, ticker }: Props) {
  const dates = data.timestamps.map(ts => new Date(ts * 1000).toISOString().slice(0, 10))
  const closes = data.close
  const ema20 = calculateEMA(closes, 20)
  const ema50 = calculateEMA(closes, 50)

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

  return (
    <PlotlyChart
      data={[candlestick, ema20Trace, ema50Trace]}
      layout={{
        title: { text: `${ticker} — Price + EMAs` },
        xaxis: { rangeslider: { visible: false } },
        height: 380,
      }}
    />
  )
}
