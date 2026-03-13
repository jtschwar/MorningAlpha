import { useNavigate } from 'react-router-dom'
import PlotlyChart from './PlotlyChart'
import type { Stock } from '../../store/types'
import { colors } from '../../tokens/theme'

interface Props {
  stocks: Stock[]
  metricLabel?: string
}

export default function BarChart({ stocks, metricLabel = '3M' }: Props) {
  const navigate = useNavigate()
  const sorted = [...stocks].sort((a, b) => a.ReturnPct - b.ReturnPct)

  const trace: Plotly.Data = {
    x: sorted.map(s => s.ReturnPct),
    y: sorted.map(s => s.Ticker),
    type: 'bar',
    orientation: 'h',
    marker: {
      color: sorted.map(s => (s.ReturnPct >= 0 ? colors.accentGreen : colors.accentRed)),
      opacity: 0.8,
    },
    text: sorted.map(s => `${s.ReturnPct >= 0 ? '+' : ''}${s.ReturnPct.toFixed(2)}%`),
    textposition: 'outside',
    hovertemplate: '<b>%{y}</b><br>Return: %{x:.2f}%<extra></extra>',
    customdata: sorted.map(s => s.Ticker),
  }

  const allX = sorted.map(s => s.ReturnPct)
  const xPad = (Math.max(...allX) - Math.min(...allX)) * 0.08

  return (
    <PlotlyChart
      data={[trace]}
      layout={{
        title: { text: `Returns — ${metricLabel}` },
        xaxis: { title: { text: 'Return (%)' }, minallowed: Math.min(...allX) - xPad, maxallowed: Math.max(...allX) + xPad },
        height: Math.max(400, sorted.length * 22),
        margin: { l: 80, r: 80, t: 40, b: 40 },
      }}
      onClick={e => {
        const ticker = e.points[0]?.customdata as string | undefined
        if (ticker) navigate(`/stock/${ticker}`)
      }}
    />
  )
}
