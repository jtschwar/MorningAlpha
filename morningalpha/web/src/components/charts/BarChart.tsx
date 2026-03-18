import { useNavigate } from 'react-router-dom'
import { useState, useEffect, useRef } from 'react'
import PlotlyChart from './PlotlyChart'
import type { Stock } from '../../store/types'
import { colors } from '../../tokens/theme'
import { computeSignal } from '../../lib/signal'

interface Props {
  stocks: Stock[]
  metricLabel?: string
}

export default function BarChart({ stocks, metricLabel = '3M' }: Props) {
  const navigate = useNavigate()
  const wrapperRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(0)

  useEffect(() => {
    const el = wrapperRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      setContainerWidth(entries[0].contentRect.width)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const sorted = [...stocks].sort((a, b) => a.ReturnPct - b.ReturnPct)

  const signals = sorted.map(s => computeSignal(s, null, null))

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
    hovertemplate: '<b>%{y}</b><br>Return: %{x:.2f}%<br>Score: %{customdata[1]}<br>Signal: <b>%{customdata[2]}</b><extra></extra>',
    customdata: sorted.map((s, i) => [
      s.Ticker,
      (s.investmentScore ?? 0).toFixed(1),
      signals[i].level,
    ]),
  }

  const allX = sorted.map(s => s.ReturnPct)
  const xPad = (Math.max(...allX) - Math.min(...allX)) * 0.08
  const chartHeight = Math.max(400, sorted.length * 22)

  return (
    <div
      ref={wrapperRef}
      style={{ height: 580, overflowY: 'auto', overflowX: 'hidden', scrollbarWidth: 'thin' }}
    >
      <PlotlyChart
        data={[trace]}
        responsive={false}
        layout={{
          title: { text: `Returns — ${metricLabel}` },
          xaxis: { title: { text: 'Return (%)' }, minallowed: Math.min(...allX) - xPad, maxallowed: Math.max(...allX) + xPad },
          height: chartHeight,
          width: containerWidth || undefined,
          autosize: false,
          bargap: 0.2,
          margin: { l: 80, r: 80, t: 40, b: 40 },
        }}
        onClick={e => {
          const cd = e.points[0]?.customdata as unknown as string[] | undefined
          const ticker = cd?.[0]
          if (ticker) navigate(`/stock/${ticker}`)
        }}
      />
    </div>
  )
}
