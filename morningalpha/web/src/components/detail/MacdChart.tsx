import PlotlyChart from '../charts/PlotlyChart'
import { calculateMACD } from '../../lib/indicators'
import type { StockDetailData } from '../../store/types'
import { colors } from '../../tokens/theme'

interface Props {
  data: StockDetailData
}

export default function MacdChart({ data }: Props) {
  const dates = data.timestamps.map(ts => new Date(ts * 1000).toISOString().slice(0, 10))
  const { macd, signal, histogram } = calculateMACD(data.close)

  const histColors = histogram.map(v => (v == null || v >= 0 ? colors.accentGreen : colors.accentRed))

  const histTrace: Plotly.Data = {
    type: 'bar',
    x: dates,
    y: histogram,
    name: 'Histogram',
    marker: { color: histColors, opacity: 0.7 },
  }

  const macdLine: Plotly.Data = {
    type: 'scatter',
    x: dates,
    y: macd,
    mode: 'lines',
    name: 'MACD',
    line: { color: colors.accentBlue, width: 1.5 },
    connectgaps: false,
  }

  const signalLine: Plotly.Data = {
    type: 'scatter',
    x: dates,
    y: signal,
    mode: 'lines',
    name: 'Signal',
    line: { color: '#f97316', width: 1.5 },
    connectgaps: false,
  }

  const zeroLine: Plotly.Data = {
    type: 'scatter',
    x: [dates[0], dates[dates.length - 1]],
    y: [0, 0],
    mode: 'lines',
    name: 'Zero',
    line: { color: colors.textSecondary, width: 1 },
    showlegend: false,
  }

  return (
    <PlotlyChart
      data={[histTrace, macdLine, signalLine, zeroLine]}
      layout={{
        title: { text: 'MACD' },
        height: 220,
      }}
    />
  )
}
