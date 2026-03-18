import Plot from 'react-plotly.js'
import type { CSSProperties } from 'react'
import { plotlyBase, plotlyBaseLight } from '../../tokens/theme'
import { useTheme } from '../../store/ThemeContext'

interface Props {
  data: Plotly.Data[]
  layout?: Partial<Plotly.Layout>
  style?: CSSProperties
  responsive?: boolean
  onClick?: (event: Readonly<Plotly.PlotMouseEvent>) => void
}

export default function PlotlyChart({ data, layout = {}, style, responsive = true, onClick }: Props) {
  const { isDark } = useTheme()
  const base = isDark ? plotlyBase : plotlyBaseLight

  return (
    <Plot
      data={data}
      layout={{ ...(base as Partial<Plotly.Layout>), ...layout }}
      config={{ responsive, displayModeBar: 'hover' }}
      style={{ width: '100%', ...style }}
      onClick={onClick}
      useResizeHandler={responsive}
    />
  )
}
