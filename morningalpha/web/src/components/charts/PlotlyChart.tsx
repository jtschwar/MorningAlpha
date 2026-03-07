import Plot from 'react-plotly.js'
import type { CSSProperties } from 'react'
import { plotlyBase } from '../../tokens/theme'

interface Props {
  data: Plotly.Data[]
  layout?: Partial<Plotly.Layout>
  style?: CSSProperties
  onClick?: (event: Readonly<Plotly.PlotMouseEvent>) => void
}

export default function PlotlyChart({ data, layout = {}, style, onClick }: Props) {
  return (
    <Plot
      data={data}
      layout={{ ...(plotlyBase as Partial<Plotly.Layout>), ...layout }}
      config={{ responsive: true, displayModeBar: 'hover', scrollZoom: true }}
      style={{ width: '100%', ...style }}
      onClick={onClick}
      useResizeHandler
    />
  )
}
