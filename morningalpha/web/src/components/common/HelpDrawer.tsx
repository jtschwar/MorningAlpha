import { useState } from 'react'
import type { ReactNode } from 'react'
import styles from './HelpDrawer.module.css'

interface Section {
  id: string
  title: string
  content: ReactNode
}

const SECTIONS: Section[] = [
  {
    id: 'returns',
    title: 'Returns & Periods',
    content: (
      <div>
        <p>Return % is the price change over the selected lookback window (2W / 1M / 3M / 6M).</p>
        <p><strong>2W window caveat:</strong> Sharpe ratio with ~10 trading days is statistically unreliable (marked ⚠). RSI(14) requires 14+ days and shows N/A. Entry Score and Quality Score remain valid.</p>
      </div>
    ),
  },
  {
    id: 'risk',
    title: 'Risk Metrics',
    content: (
      <div>
        <p><strong>Sharpe Ratio</strong> = (Return − Risk-free rate) / Std Dev of returns. Higher is better; &gt;1 is good, &gt;2 is excellent.</p>
        <p><strong>Sortino Ratio</strong> = Like Sharpe but only penalizes downside volatility.</p>
        <p><strong>Max Drawdown</strong> = Largest peak-to-trough decline. E.g. −25% means the stock fell 25% from its peak before recovering.</p>
        <p><strong>Annualized Volatility</strong> = σ_daily × √252 × 100, expressed as a %. A 20% vol stock moves ~20% annually on average.</p>
      </div>
    ),
  },
  {
    id: 'scores',
    title: 'Scoring',
    content: (
      <div>
        <p><strong>Investment Score (0–100)</strong> = weighted composite: Return 30% + Quality 25% + Sharpe 20% + Consistency 15% + Drawdown 10%.</p>
        <p><strong>Quality Score (0–100)</strong> = measures consistency, low drawdown, and positive momentum. Higher = more reliable historical performance.</p>
        <p><strong>Entry Score (0–100)</strong> = short-term timing signal combining RSI, momentum acceleration, price vs 20-day high, and volume surge.</p>
        <p><strong>Risk/Reward Ratio</strong> = |Return| / |Max Drawdown|. A ratio &gt;1 means you earned more than your worst loss.</p>
      </div>
    ),
  },
  {
    id: 'technicals',
    title: 'Technical Indicators',
    content: (
      <div>
        <p><strong>RSI (Relative Strength Index)</strong> — 14-period. &gt;70 = overbought (potential pullback), &lt;30 = oversold (potential bounce). 30–70 = neutral.</p>
        <p><strong>EMA Signal</strong> — Compares 20-day vs 50-day EMA. EMA20 &gt; EMA50 = short-term bullish momentum. Note: a true Golden Cross uses 50/200 EMA, not 20/50.</p>
        <p><strong>Momentum Accel</strong> = rate of change of returns. Positive = accelerating upward momentum.</p>
        <p><strong>Volume Surge</strong> = recent volume vs 20-day average. &gt;1.5 = elevated volume, often confirms price moves.</p>
      </div>
    ),
  },
  {
    id: 'navigation',
    title: 'Keyboard Shortcuts',
    content: (
      <div>
        <p><kbd>J</kbd> / <kbd>↓</kbd> — move focus down in table</p>
        <p><kbd>K</kbd> / <kbd>↑</kbd> — move focus up in table</p>
        <p><kbd>Enter</kbd> — open focused stock detail</p>
        <p><kbd>?</kbd> — open this help drawer</p>
      </div>
    ),
  },
]

interface Props {
  initialSection?: string
}

export default function HelpDrawer({ initialSection }: Props) {
  const [open, setOpen] = useState(false)
  const [expanded, setExpanded] = useState<string>(initialSection ?? '')

  function toggle(id: string) {
    setExpanded(prev => (prev === id ? '' : id))
  }

  // Open drawer and jump to section via deep-link
  function openSection(id: string) {
    setOpen(true)
    setExpanded(id)
  }

  // Expose openSection globally so column headers can trigger it
  if (typeof window !== 'undefined') {
    (window as unknown as Record<string, unknown>).__helpDrawer = { openSection }
  }

  return (
    <>
      {/* FAB button */}
      <button
        className={styles.fab}
        onClick={() => setOpen(o => !o)}
        aria-label="Help"
        title="Help"
      >
        ?
      </button>

      {/* Backdrop */}
      {open && <div className={styles.backdrop} onClick={() => setOpen(false)} />}

      {/* Drawer */}
      <aside className={`${styles.drawer} ${open ? styles.open : ''}`}>
        <div className={styles.drawerHeader}>
          <span className={styles.drawerTitle}>Help & Metrics Guide</span>
          <button className={styles.closeBtn} onClick={() => setOpen(false)}>✕</button>
        </div>

        <div className={styles.sections}>
          {SECTIONS.map(s => (
            <div key={s.id} className={styles.section}>
              <button
                className={`${styles.sectionBtn} ${expanded === s.id ? styles.sectionOpen : ''}`}
                onClick={() => toggle(s.id)}
              >
                <span>{s.title}</span>
                <span className={styles.chevron}>{expanded === s.id ? '▲' : '▼'}</span>
              </button>
              {expanded === s.id && (
                <div className={styles.sectionContent}>{s.content}</div>
              )}
            </div>
          ))}
        </div>
      </aside>
    </>
  )
}
