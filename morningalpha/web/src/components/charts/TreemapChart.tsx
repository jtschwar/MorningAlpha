import { useNavigate } from 'react-router-dom'
import PlotlyChart from './PlotlyChart'
import type { Stock } from '../../store/types'

interface Props {
  stocks: Stock[]
}

export default function TreemapChart({ stocks }: Props) {
  const navigate = useNavigate()

  const exchanges = [...new Set(stocks.map(s => s.Exchange))]

  // Plotly treemap needs parent nodes (exchanges) + leaf nodes (tickers)
  const labels = [...exchanges, ...stocks.map(s => s.Ticker)]
  const parents = [
    ...exchanges.map(() => ''),
    ...stocks.map(s => s.Exchange),
  ]
  const values = [
    ...exchanges.map(() => null as unknown as number),
    ...stocks.map(s => Math.abs(s.ReturnPct) + 0.01), // +0.01 so 0% still shows
  ]
  const colors = [
    ...exchanges.map(() => 0),
    ...stocks.map(s => s.ReturnPct),
  ]
  const customdata = ['', ...stocks.map(s => s.Ticker)]

  const trace: Plotly.Data = {
    type: 'treemap',
    labels,
    parents,
    values,
    marker: {
      colors,
      colorscale: [
        [0, '#ef4444'],
        [0.5, '#3b82f6'],
        [1, '#10b981'],
      ],
      showscale: false,
    },
    hovertemplate: '<b>%{label}</b><br>Return: %{color:.2f}%<extra></extra>',
    customdata,
  } as Plotly.Data

  return (
    <PlotlyChart
      data={[trace]}
      layout={{ height: 480, margin: { t: 20, r: 0, b: 0, l: 0 } }}
      onClick={e => {
        const ticker = e.points[0]?.customdata as string | undefined
        if (ticker) navigate(`/stock/${ticker}`)
      }}
    />
  )
}
