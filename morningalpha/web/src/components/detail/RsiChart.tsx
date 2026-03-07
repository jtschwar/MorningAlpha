import PlotlyChart from '../charts/PlotlyChart'
import { calculateRSISeries } from '../../lib/technicals'
import type { StockDetailData } from '../../store/types'
import { colors } from '../../tokens/theme'

interface Props {
  data: StockDetailData
}

export default function RsiChart({ data }: Props) {
  const dates = data.timestamps.map(ts => new Date(ts * 1000).toISOString().slice(0, 10))
  const rsi = calculateRSISeries(data.close, 14)

  const rsiTrace: Plotly.Data = {
    type: 'scatter',
    x: dates,
    y: rsi,
    mode: 'lines',
    name: 'RSI (14)',
    line: { color: colors.accentPurple, width: 1.5 },
    connectgaps: false,
  }

  // Reference lines at 30 and 70
  const xRange = [dates[0], dates[dates.length - 1]]
  const ob: Plotly.Data = {
    type: 'scatter',
    x: xRange,
    y: [70, 70],
    mode: 'lines',
    name: 'Overbought (70)',
    line: { color: colors.accentRed, width: 1, dash: 'dash' },
  }
  const os: Plotly.Data = {
    type: 'scatter',
    x: xRange,
    y: [30, 30],
    mode: 'lines',
    name: 'Oversold (30)',
    line: { color: colors.accentGreen, width: 1, dash: 'dash' },
  }

  return (
    <PlotlyChart
      data={[rsiTrace, ob, os]}
      layout={{
        title: { text: 'RSI (14-period)' },
        yaxis: { range: [0, 100], title: { text: 'RSI' } },
        height: 220,
      }}
    />
  )
}
