import { useNavigate } from 'react-router-dom'
import PlotlyChart from './PlotlyChart'
import type { Stock } from '../../store/types'

const EXCHANGE_COLORS: Record<string, string> = {
  NASDAQ: '#3b82f6',
  NYSE: '#8b5cf6',
  'S&P 500': '#10b981',
}

interface Props {
  stocks: Stock[]
}

export default function ScatterChart({ stocks }: Props) {
  const navigate = useNavigate()

  const exchanges = [...new Set(stocks.map(s => s.Exchange))]

  const traces: Plotly.Data[] = exchanges.map(ex => {
    const group = stocks.filter(s => s.Exchange === ex)
    return {
      x: group.map((_, i) => i + 1),
      y: group.map(s => s.ReturnPct),
      mode: 'markers',
      type: 'scatter',
      name: ex,
      marker: { size: 9, opacity: 0.75, color: EXCHANGE_COLORS[ex] ?? '#64748b' },
      text: group.map(s => `${s.Ticker}<br>${s.Name}<br>${s.ReturnPct.toFixed(2)}%`),
      hovertemplate: '%{text}<extra></extra>',
      customdata: group.map(s => s.Ticker),
    }
  })

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: 'Rank' } },
        yaxis: { title: { text: 'Return (%)' } },
        hovermode: 'closest',
        height: 420,
      }}
      onClick={e => {
        const ticker = e.points[0]?.customdata as string | undefined
        if (ticker) navigate(`/stock/${ticker}`)
      }}
    />
  )
}
