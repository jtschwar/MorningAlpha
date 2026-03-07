import { useNavigate } from 'react-router-dom'
import type { Stock } from '../../store/types'
import { ScoreBadge, RsiBadge, RiskBadge } from '../common/Badge'
import styles from './StockTable.module.css'

function fmt(v: number | null, dec = 2): string {
  return v == null ? '—' : v.toFixed(dec)
}

function fmtMarketCap(v: number | null): string {
  if (v == null) return '—'
  if (v >= 1e12) return `$${(v / 1e12).toFixed(1)}T`
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`
  if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}M`
  return `$${v.toFixed(0)}`
}

interface Props {
  stock: Stock
  focused: boolean
  onClick?: () => void
}

export default function TableRow({ stock: s, focused, onClick }: Props) {
  const navigate = useNavigate()

  function go() {
    onClick?.()
    navigate(`/stock/${s.Ticker}`)
  }

  return (
    <tr
      className={`${styles.row} ${focused ? styles.focused : ''}`}
      onClick={go}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && go()}
    >
      <td className={styles.tdMono}>{s.Rank}</td>
      <td className={styles.tdTicker}>{s.Ticker}</td>
      <td className={styles.tdName}>{s.Name}</td>
      <td className={styles.tdMono}>{s.Exchange}</td>
      <td className={`${styles.tdMono} ${s.ReturnPct >= 0 ? styles.pos : styles.neg}`}>
        {s.ReturnPct >= 0 ? '+' : ''}{s.ReturnPct.toFixed(2)}%
      </td>
      <td className={styles.tdMono}>{fmt(s.SharpeRatio)}</td>
      <td><ScoreBadge value={s.QualityScore} /></td>
      <td><ScoreBadge value={s.EntryScore} /></td>
      <td className={`${styles.tdMono} ${s.MaxDrawdown != null && s.MaxDrawdown < -20 ? styles.neg : ''}`}>
        {fmt(s.MaxDrawdown)}%
      </td>
      <td><RsiBadge value={s.RSI} /></td>
      <td className={styles.tdMono}>{fmtMarketCap(s.MarketCap)}</td>
      <td><RiskBadge level={s.riskLevel} /></td>
      <td><ScoreBadge value={s.investmentScore} /></td>
    </tr>
  )
}
