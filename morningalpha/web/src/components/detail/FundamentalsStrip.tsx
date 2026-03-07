import type { FundamentalsData } from '../../hooks/useFundamentals'
import styles from './FundamentalsStrip.module.css'

interface Props {
  data: FundamentalsData
}

export default function FundamentalsStrip({ data: f }: Props) {
  function mult(v: number | null): string {
    return v != null ? `${v.toFixed(1)}x` : '—'
  }
  function pct(v: number | null, isDecimal = true): string {
    if (v == null) return '—'
    const val = isDecimal ? v * 100 : v
    const sign = val > 0 ? '+' : ''
    return `${sign}${val.toFixed(1)}%`
  }

  const peLabel =
    f.pe == null ? 'No earnings'
    : f.pe < 0   ? 'Negative earnings'
    : f.pe < 12  ? 'Deep value'
    : f.pe < 20  ? 'Value range'
    : f.pe < 30  ? 'Fair value'
    : f.pe < 50  ? 'Growth premium'
    : 'Expensive'

  const fwdPeLabel =
    f.forwardPE == null ? 'No estimate'
    : f.forwardPE < 0   ? 'Loss expected'
    : f.forwardPE < f.pe! ? 'Earnings growing'
    : f.forwardPE < 20  ? 'Reasonable outlook'
    : f.forwardPE < 30  ? 'Priced for growth'
    : 'Rich valuation'

  const pbLabel =
    f.pb == null ? '—'
    : f.pb < 1   ? 'Below book value'
    : f.pb < 3   ? 'Reasonable'
    : f.pb < 5   ? 'Premium to book'
    : 'Expensive vs book'

  const debtLabel =
    f.debtToEquity == null ? '—'
    : f.debtToEquity < 50  ? 'Low leverage'
    : f.debtToEquity < 150 ? 'Moderate leverage'
    : f.debtToEquity < 300 ? 'High leverage'
    : 'Highly leveraged'

  const marginLabel =
    f.netMargin == null      ? '—'
    : f.netMargin < 0        ? 'Loss-making'
    : f.netMargin < 0.05     ? 'Thin margins'
    : f.netMargin < 0.10     ? 'Decent margins'
    : f.netMargin < 0.20     ? 'Strong margins'
    : 'Excellent margins'

  const growthLabel =
    f.revenueGrowth == null   ? '—'
    : f.revenueGrowth < -0.05 ? 'Revenue declining'
    : f.revenueGrowth < 0.05  ? 'Flat revenue'
    : f.revenueGrowth < 0.10  ? 'Slow growth'
    : f.revenueGrowth < 0.20  ? 'Strong growth'
    : 'Hypergrowth'

  const roeLabel =
    f.roe == null    ? '—'
    : f.roe < 0      ? 'Destroying value'
    : f.roe < 0.10   ? 'Below average'
    : f.roe < 0.15   ? 'Average'
    : f.roe < 0.25   ? 'Good efficiency'
    : 'Excellent ROE'

  const divLabel =
    f.dividendYield == null || f.dividendYield === 0 ? 'No dividend'
    : f.dividendYield < 0.02 ? 'Minimal yield'
    : f.dividendYield < 0.04 ? 'Modest income'
    : 'High yield'

  return (
    <div className={styles.strip}>
      <div className={styles.sectionLabel}>Fundamentals</div>
      <div className={styles.items}>
        <div className={styles.item}>
          <span className={styles.label}>P/E</span>
          <span className={styles.val}>{mult(f.pe)}</span>
          <span className={styles.sub}>{peLabel}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>Fwd P/E</span>
          <span className={styles.val}>{mult(f.forwardPE)}</span>
          <span className={styles.sub}>{f.forwardPE != null ? fwdPeLabel : 'No estimate'}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>P/B</span>
          <span className={styles.val}>{mult(f.pb)}</span>
          <span className={styles.sub}>{pbLabel}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>Debt / Eq</span>
          <span className={`${styles.val} ${f.debtToEquity != null && f.debtToEquity > 200 ? styles.warn : ''}`}>
            {f.debtToEquity != null ? `${f.debtToEquity.toFixed(0)}%` : '—'}
          </span>
          <span className={styles.sub}>{debtLabel}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>Net Margin</span>
          <span className={`${styles.val} ${f.netMargin != null ? (f.netMargin >= 0 ? styles.pos : styles.neg) : ''}`}>
            {pct(f.netMargin)}
          </span>
          <span className={styles.sub}>{marginLabel}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>Rev Growth</span>
          <span className={`${styles.val} ${f.revenueGrowth != null ? (f.revenueGrowth >= 0 ? styles.pos : styles.neg) : ''}`}>
            {pct(f.revenueGrowth)}
          </span>
          <span className={styles.sub}>{growthLabel}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>ROE</span>
          <span className={`${styles.val} ${f.roe != null ? (f.roe >= 0.15 ? styles.pos : '') : ''}`}>
            {pct(f.roe)}
          </span>
          <span className={styles.sub}>{roeLabel}</span>
        </div>
        <div className={styles.item}>
          <span className={styles.label}>Div Yield</span>
          <span className={styles.val}>
            {f.dividendYield != null && f.dividendYield > 0 ? pct(f.dividendYield) : '—'}
          </span>
          <span className={styles.sub}>{divLabel}</span>
        </div>
      </div>
    </div>
  )
}
