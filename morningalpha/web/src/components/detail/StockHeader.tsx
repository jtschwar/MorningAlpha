import { useNavigate } from 'react-router-dom'
import type { Stock } from '../../store/types'
import { ReturnBadge, RiskBadge } from '../common/Badge'
import styles from './StockHeader.module.css'

interface Props {
  ticker: string
  stock: Stock | null
  metric: string
}

export default function StockHeader({ ticker, stock, metric }: Props) {
  const navigate = useNavigate()

  function goBack() {
    if (window.history.length > 1) navigate(-1)
    else navigate('/')
  }

  return (
    <div className={styles.header}>
      <button className={styles.back} onClick={goBack}>← Back</button>

      <div className={styles.info}>
        <div className={styles.row}>
          <a
            className={styles.ticker}
            href={`https://finance.yahoo.com/quote/${ticker}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            {ticker}
            <span className={styles.extIcon}>↗</span>
          </a>
          {stock && <ReturnBadge value={stock.ReturnPct} />}
          {stock && <RiskBadge level={stock.riskLevel} />}
        </div>
        {stock && (
          <div className={styles.sub}>
            {stock.Name} · {stock.Exchange} · {metric} Return
          </div>
        )}
      </div>
    </div>
  )
}
