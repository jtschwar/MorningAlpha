import PlotlyChart from '../charts/PlotlyChart'
import type { StockDetailData } from '../../store/types'
import { colors } from '../../tokens/theme'

interface Props {
  data: StockDetailData
}

export default function VolumeChart({ data }: Props) {
  const dates = data.timestamps.map(ts => new Date(ts * 1000).toISOString().slice(0, 10))

  const trace: Plotly.Data = {
    type: 'bar',
    x: dates,
    y: data.volume,
    name: 'Volume',
    marker: { color: colors.accentBlue, opacity: 0.6 },
  }

  return (
    <PlotlyChart
      data={[trace]}
      layout={{
        title: { text: 'Trading Volume' },
        yaxis: { title: { text: 'Volume' } },
        height: 200,
      }}
    />
  )
}
