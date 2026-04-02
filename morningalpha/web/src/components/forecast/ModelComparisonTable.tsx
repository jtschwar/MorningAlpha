import type { TickerEntry } from '../../hooks/useTickerIndex'
import type { ForecastCalibration } from '../../hooks/useForecastCalibration'
import styles from './ModelComparisonTable.module.css'

const SERIES_COLORS = ['#3B82F6', '#F59E0B', '#22C55E', '#A78BFA', '#EF4444']

interface Props {
  tickers: TickerEntry[]
  calibration: ForecastCalibration | null
}

function scoreClass(v: number | null): string {
  if (v === null) return styles.muted
  if (v >= 70) return styles.pos
  if (v >= 40) return styles.amber
  return styles.neg
}

function signalLabel(score: number | null): { text: string; cls: string } {
  if (score === null) return { text: 'N/A', cls: styles.muted }
  if (score >= 70) return { text: 'Bullish', cls: styles.bullish }
  if (score >= 40) return { text: 'Neutral', cls: styles.neutral }
  return { text: 'Bearish', cls: styles.bearish }
}

function getExpectedReturn(ticker: TickerEntry, cal: ForecastCalibration | null): string {
  if (!cal) return '—'
  const deciles = cal.horizons['63']
  if (!deciles || deciles.length === 0) return '—'
  const decileIdx = Math.min(9, Math.max(0, Math.floor((ticker.mlScore ?? 50) / 10)))
  const stats = deciles[Math.min(decileIdx, deciles.length - 1)]
  if (!stats) return '—'
  return (stats.period_return_mean * 100).toFixed(1) + '%'
}

export default function ModelComparisonTable({ tickers, calibration }: Props) {
  if (tickers.length === 0) {
    return (
      <div className={styles.wrap}>
        <div className={styles.title}>Model Comparison</div>
        <div style={{ padding: '20px', color: 'var(--text-muted)', fontSize: '0.82rem', fontFamily: "'IBM Plex Mono', monospace" }}>
          Select stocks to compare
        </div>
      </div>
    )
  }

  const rows: { label: string; getValue: (t: TickerEntry) => { text: string; cls: string } }[] = [
    {
      label: 'Consensus Score',
      getValue: t => {
        const v = t.mlScore
        return { text: v !== null ? String(Math.round(v)) : '—', cls: scoreClass(v) }
      },
    },
    {
      label: 'Breakout',
      getValue: t => {
        const v = t.mlScore_breakout
        return { text: v !== null ? String(Math.round(v)) : '—', cls: scoreClass(v) }
      },
    },
    {
      label: 'Composite',
      getValue: t => {
        const v = t.mlScore_composite
        return { text: v !== null ? String(Math.round(v)) : '—', cls: scoreClass(v) }
      },
    },
    {
      label: 'Set Transformer',
      getValue: t => {
        const v = t.mlScore_st
        return { text: v !== null ? String(Math.round(v)) : '—', cls: scoreClass(v) }
      },
    },
    {
      label: 'Expected Ret (63d)',
      getValue: t => {
        const text = getExpectedReturn(t, calibration)
        const isPos = text !== '—' && !text.startsWith('-')
        return { text, cls: text === '—' ? styles.muted : isPos ? styles.pos : styles.neg }
      },
    },
    {
      label: 'Signal',
      getValue: t => signalLabel(t.mlScore),
    },
  ]

  return (
    <div className={styles.wrap}>
      <div className={styles.title}>Model Comparison</div>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Metric</th>
              {tickers.map((t, i) => (
                <th key={t.ticker}>
                  <div className={styles.tickerHeader}>
                    <span className={styles.dot} style={{ background: SERIES_COLORS[i % SERIES_COLORS.length] }} />
                    {t.ticker}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(row => (
              <tr key={row.label}>
                <td>{row.label}</td>
                {tickers.map(t => {
                  const { text, cls } = row.getValue(t)
                  return (
                    <td key={t.ticker} className={cls}>{text}</td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
