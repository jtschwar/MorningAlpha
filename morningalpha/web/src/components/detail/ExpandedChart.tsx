import type { StockDetailData } from '../../store/types'
import PriceChart from './PriceChart'

interface Props {
  data: StockDetailData
  ticker: string
}

export default function ExpandedChart({ data, ticker }: Props) {
  return <PriceChart data={data} ticker={ticker} expanded={true} />
}
