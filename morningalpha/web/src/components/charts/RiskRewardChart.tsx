import { useNavigate } from 'react-router-dom'
import PlotlyChart from './PlotlyChart'
import type { Stock } from '../../store/types'
import { colors } from '../../tokens/theme'

const RISK_COLORS: Record<string, string> = {
  low: colors.accentGreen,
  moderate: colors.accentBlue,
  high: '#f59e0b',
  'very-high': colors.accentRed,
  unknown: colors.textSecondary,
}

interface Props {
  stocks: Stock[]
}

export default function RiskRewardChart({ stocks }: Props) {
  const navigate = useNavigate()

  const grouped: Record<string, { x: number[]; y: number[]; text: string[]; tickers: string[] }> =
    {}

  for (const s of stocks) {
    const level = s.riskLevel
    if (!grouped[level]) grouped[level] = { x: [], y: [], text: [], tickers: [] }
    grouped[level].x.push(Math.abs(s.MaxDrawdown ?? 0))
    grouped[level].y.push(s.ReturnPct)
    grouped[level].text.push(
      `${s.Ticker}<br>Return: ${s.ReturnPct.toFixed(2)}%<br>Max DD: ${(s.MaxDrawdown ?? 0).toFixed(1)}%`,
    )
    grouped[level].tickers.push(s.Ticker)
  }

  const traces: Plotly.Data[] = Object.entries(grouped).map(([level, d]) => ({
    x: d.x,
    y: d.y,
    mode: 'markers',
    type: 'scatter',
    name: level.charAt(0).toUpperCase() + level.slice(1).replace('-', ' ') + ' Risk',
    text: d.text,
    hovertemplate: '%{text}<extra></extra>',
    customdata: d.tickers,
    marker: { size: 9, opacity: 0.75, color: RISK_COLORS[level] ?? '#64748b' },
  }))

  const allX = Object.values(grouped).flatMap(d => d.x)
  const allY = Object.values(grouped).flatMap(d => d.y)
  const xPad = (Math.max(...allX) - Math.min(...allX)) * 0.08
  const yPad = (Math.max(...allY) - Math.min(...allY)) * 0.08

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: 'Max Drawdown (%)' }, minallowed: Math.min(...allX) - xPad, maxallowed: Math.max(...allX) + xPad },
        yaxis: { title: { text: 'Return (%)' }, minallowed: Math.min(...allY) - yPad, maxallowed: Math.max(...allY) + yPad },
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
