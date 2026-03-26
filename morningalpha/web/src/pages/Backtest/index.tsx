import { useEffect, useState } from 'react'
import Plot from 'react-plotly.js'
import AppShell from '../../components/layout/AppShell'
import HelpDrawer from '../../components/common/HelpDrawer'
import styles from './Backtest.module.css'

// ---------------------------------------------------------------------------
// Types matching the JSON files written by `alpha ml backtest`
// ---------------------------------------------------------------------------

interface LeaderboardEntry {
  model_id: string
  model_type: string
  n_features: number
  test_period: { start: string; end: string }
  ic_mean: number
  ic_std: number
  icir: number
  ic_hit_rate: number
  ic_tstat: number
  n_months: number
  ls_sharpe: number
  ls_ann_return: number
  ls_max_drawdown: number
  n_periods: number
  persistence_ic: number
  n_test_rows: number
  n_test_dates: number
}

interface IcEntry { month: string; ic: number }
interface CumIcEntry { month: string; cumulative_ic: number }
interface EquityEntry { date: string; cumulative_return: number; underwater: number }
interface DecileEntry { decile: number; ann_return: number }
interface FeatureEntry { feature: string; importance: number; category: string }

interface ModelDetail {
  icOverTime: IcEntry[]
  cumIc: CumIcEntry[]
  equityCurve: EquityEntry[]
  decileReturns: DecileEntry[]
  featureImportance: FeatureEntry[]
}

type Tab = 'leaderboard' | 'ic' | 'equity' | 'features'

// ---------------------------------------------------------------------------
// Shared Plotly dark-theme layout
// ---------------------------------------------------------------------------

const PLOT_LAYOUT_BASE: Partial<Plotly.Layout> = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { family: "'JetBrains Mono', monospace", color: '#94a3b8', size: 11 },
  margin: { t: 20, r: 20, b: 50, l: 55 },
  xaxis: {
    gridcolor: '#1e2d45',
    linecolor: '#1e2d45',
    tickcolor: '#1e2d45',
    zerolinecolor: '#1e2d45',
  },
  yaxis: {
    gridcolor: '#1e2d45',
    linecolor: '#1e2d45',
    tickcolor: '#1e2d45',
    zerolinecolor: '#64748b',
    zerolinewidth: 1,
  },
}

const PLOT_CONFIG: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const BASE = './data/backtest'

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json() as Promise<T>
}

function fmt(val: number, decimals = 3) {
  return val.toFixed(decimals)
}

function pct(val: number, decimals = 1) {
  return `${(val * 100).toFixed(decimals)}%`
}

function colorByIC(ic: number) {
  return ic >= 0 ? '#10b981' : '#ef4444'
}

const CATEGORY_COLOR: Record<string, string> = {
  technical: '#3b82f6',
  market_context: '#8b5cf6',
  fundamental: '#f59e0b',
  derived: '#10b981',
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function LoadingPanel() {
  return <div className={styles.loadingPanel}>Loading...</div>
}

function EmptyPanel({ label }: { label: string }) {
  return <div className={styles.emptyPanel}>{label}</div>
}

function KpiStrip({ e }: { e: LeaderboardEntry }) {
  const stats = [
    { label: 'IC Mean', value: fmt(e.ic_mean), color: e.ic_mean > 0 ? 'pos' : 'neg' },
    { label: 'IC Std', value: fmt(e.ic_std), color: 'neutral' },
    { label: 'ICIR', value: fmt(e.icir, 2), color: e.icir > 0.5 ? 'pos' : 'neutral' },
    { label: 'IC Hit Rate', value: pct(e.ic_hit_rate, 1), color: e.ic_hit_rate > 0.5 ? 'pos' : 'neutral' },
    { label: 'IC t-stat', value: fmt(e.ic_tstat, 2), color: e.ic_tstat > 2 ? 'pos' : 'neutral' },
    { label: 'L/S Sharpe', value: fmt(e.ls_sharpe, 2), color: e.ls_sharpe > 1 ? 'pos' : 'neutral' },
    { label: 'L/S Ann. Ret', value: pct(e.ls_ann_return), color: e.ls_ann_return > 0 ? 'pos' : 'neg' },
    { label: 'Max Drawdown', value: pct(e.ls_max_drawdown), color: 'neg' },
    { label: 'Persistence IC', value: fmt(e.persistence_ic), color: 'neutral' },
  ]
  return (
    <div className={styles.kpiStrip}>
      {stats.map(s => (
        <div key={s.label} className={styles.kpi}>
          <span className={styles.kpiLabel}>{s.label}</span>
          <span className={`${styles.kpiVal} ${styles[s.color as keyof typeof styles] ?? ''}`}>{s.value}</span>
        </div>
      ))}
    </div>
  )
}

function LeaderboardTab({
  entries,
  selectedId,
  onSelect,
}: {
  entries: LeaderboardEntry[]
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  const cols = [
    { key: 'model_id', label: 'Model' },
    { key: 'ic_mean', label: 'IC Mean', fmt: (v: number) => fmt(v) },
    { key: 'icir', label: 'ICIR', fmt: (v: number) => fmt(v, 2) },
    { key: 'ic_hit_rate', label: 'Hit Rate', fmt: (v: number) => pct(v, 1) },
    { key: 'ic_tstat', label: 't-stat', fmt: (v: number) => fmt(v, 2) },
    { key: 'ls_sharpe', label: 'L/S Sharpe', fmt: (v: number) => fmt(v, 2) },
    { key: 'ls_ann_return', label: 'L/S Ann. Ret', fmt: (v: number) => pct(v) },
    { key: 'ls_max_drawdown', label: 'Max DD', fmt: (v: number) => pct(v) },
    { key: 'n_features', label: 'Features', fmt: (v: number) => String(v) },
    { key: 'n_test_dates', label: 'Test Dates', fmt: (v: number) => String(v) },
  ]

  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            {cols.map(c => (
              <th key={c.key}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {entries.map(e => (
            <tr
              key={e.model_id}
              className={`${styles.tableRow} ${e.model_id === selectedId ? styles.tableRowSelected : ''}`}
              onClick={() => onSelect(e.model_id)}
              title="Click to inspect this model"
            >
              {cols.map(c => {
                const raw = e[c.key as keyof LeaderboardEntry]
                const display = c.fmt ? c.fmt(raw as number) : String(raw)
                let colorClass = ''
                if (c.key === 'ic_mean' || c.key === 'icir') colorClass = (raw as number) > 0 ? styles.pos : styles.neg
                if (c.key === 'ls_ann_return') colorClass = (raw as number) > 0 ? styles.pos : styles.neg
                if (c.key === 'ls_max_drawdown') colorClass = styles.neg
                return (
                  <td key={c.key} className={colorClass}>
                    {c.key === 'model_id' ? <span className={styles.mono}>{display}</span> : display}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className={styles.tableHint}>Click a row to inspect that model's charts →</div>
    </div>
  )
}

function IcTab({ data }: { data: ModelDetail }) {
  const { icOverTime, cumIc } = data
  const months = icOverTime.map(d => d.month)
  const ics = icOverTime.map(d => d.ic)
  const barColors = ics.map(colorByIC)

  const cumDates = cumIc.map(d => d.month)
  const cumVals = cumIc.map(d => d.cumulative_ic)

  return (
    <div className={styles.chartGrid}>
      {/* Monthly IC bars */}
      <div className={styles.chartCard}>
        <div className={styles.chartTitle}>Monthly IC</div>
        <div className={styles.chartDesc}>
          Average rank correlation between the model's predicted scores and actual forward returns for each month.
          Green bars mean the model correctly ranked outperformers — aim for consistently positive values above 0.05.
          A single bad month is fine; persistent red bars would indicate the signal has broken down.
        </div>
        <Plot
          data={[{
            type: 'bar',
            x: months,
            y: ics,
            marker: { color: barColors },
            hovertemplate: '%{x}: %{y:.4f}<extra></extra>',
          }]}
          layout={{
            ...PLOT_LAYOUT_BASE,
            height: 260,
            yaxis: {
              ...PLOT_LAYOUT_BASE.yaxis,
              title: { text: 'Rank IC', standoff: 8 },
              tickformat: '.2f',
            },
            shapes: [{
              type: 'line',
              x0: months[0],
              x1: months[months.length - 1],
              y0: 0,
              y1: 0,
              line: { color: '#64748b', width: 1, dash: 'dot' },
            }],
          }}
          config={PLOT_CONFIG}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>

      {/* Cumulative IC line */}
      <div className={styles.chartCard}>
        <div className={styles.chartTitle}>Cumulative IC</div>
        <div className={styles.chartDesc}>
          Running sum of monthly IC over the test period. A steadily rising line means the model generates
          consistent signal every month, not just a lucky streak in one period. Plateaus or dips show months
          where predictive power weakened — useful for spotting regime sensitivity.
        </div>
        <Plot
          data={[{
            type: 'scatter',
            mode: 'lines',
            x: cumDates,
            y: cumVals,
            line: { color: '#3b82f6', width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(59,130,246,0.12)',
            hovertemplate: '%{x}: %{y:.4f}<extra></extra>',
          }]}
          layout={{
            ...PLOT_LAYOUT_BASE,
            height: 260,
            yaxis: {
              ...PLOT_LAYOUT_BASE.yaxis,
              title: { text: 'Cumulative IC', standoff: 8 },
              tickformat: '.2f',
            },
          }}
          config={PLOT_CONFIG}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  )
}

function EquityTab({ data }: { data: ModelDetail }) {
  const { equityCurve, decileReturns } = data
  const dates = equityCurve.map(d => d.date)
  const equity = equityCurve.map(d => d.cumulative_return)
  const underwater = equityCurve.map(d => d.underwater)

  const deciles = decileReturns.map(d => `D${d.decile}`)
  const annRets = decileReturns.map(d => d.ann_return)
  const decileColors = annRets.map(v => (v >= 0 ? '#10b981' : '#ef4444'))

  return (
    <div className={styles.chartGrid}>
      {/* L/S equity curve */}
      <div className={styles.chartCard}>
        <div className={styles.chartTitle}>L/S Portfolio — Cumulative Return</div>
        <div className={styles.chartDesc}>
          Simulated portfolio that goes long the top-ranked 10% of stocks and short the bottom 10%,
          rebalanced at each non-overlapping snapshot. Starts at 1.0×. A value of 2.0× means the strategy
          doubled over the test period. 10 bps round-trip transaction cost applied per period.
        </div>
        <Plot
          data={[{
            type: 'scatter',
            mode: 'lines',
            x: dates,
            y: equity,
            line: { color: '#10b981', width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(16,185,129,0.10)',
            hovertemplate: '%{x}: %{y:.2f}x<extra></extra>',
          }]}
          layout={{
            ...PLOT_LAYOUT_BASE,
            height: 240,
            yaxis: {
              ...PLOT_LAYOUT_BASE.yaxis,
              title: { text: 'Cumulative Return (×)', standoff: 8 },
              tickformat: '.2f',
            },
          }}
          config={PLOT_CONFIG}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>

      {/* Underwater / drawdown */}
      <div className={styles.chartCard}>
        <div className={styles.chartTitle}>Drawdown (Underwater)</div>
        <div className={styles.chartDesc}>
          How far the L/S portfolio sits below its all-time high at each point in time. −15% means the
          portfolio is 15% below its peak. The curve recovering back to 0% means it has fully recovered.
          Shallow, short drawdowns indicate the model holds up well across different market conditions.
        </div>
        <Plot
          data={[{
            type: 'scatter',
            mode: 'lines',
            x: dates,
            y: underwater,
            line: { color: '#ef4444', width: 1.5 },
            fill: 'tozeroy',
            fillcolor: 'rgba(239,68,68,0.15)',
            hovertemplate: '%{x}: %{y:.1%}<extra></extra>',
          }]}
          layout={{
            ...PLOT_LAYOUT_BASE,
            height: 200,
            yaxis: {
              ...PLOT_LAYOUT_BASE.yaxis,
              title: { text: 'Drawdown', standoff: 8 },
              tickformat: '.0%',
            },
          }}
          config={PLOT_CONFIG}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>

      {/* Decile returns */}
      <div className={styles.chartCard}>
        <div className={styles.chartTitle}>Decile Returns (Ann.) — D1=short, D10=long</div>
        <div className={styles.chartDesc}>
          Stocks ranked into 10 equal buckets by predicted score, D1 lowest to D10 highest.
          A working model shows a monotonically rising staircase — the model separates winners from losers
          across the full distribution, not just at the extremes. The gap between D10 and D1 is the
          gross long-short spread before transaction costs.
        </div>
        <Plot
          data={[{
            type: 'bar',
            x: deciles,
            y: annRets,
            marker: { color: decileColors },
            hovertemplate: '%{x}: %{y:.1%}<extra></extra>',
          }]}
          layout={{
            ...PLOT_LAYOUT_BASE,
            height: 240,
            xaxis: {
              ...PLOT_LAYOUT_BASE.xaxis,
              type: 'category',
            },
            yaxis: {
              ...PLOT_LAYOUT_BASE.yaxis,
              title: { text: 'Ann. Return', standoff: 8 },
              tickformat: '.0%',
            },
          }}
          config={PLOT_CONFIG}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  )
}

function FeaturesTab({ data }: { data: ModelDetail }) {
  const top = data.featureImportance.slice(0, 25)
  const sorted = [...top].sort((a, b) => a.importance - b.importance) // ascending for horizontal bar

  const featureNames = sorted.map(f => f.feature)
  const rawImportances = sorted.map(f => f.importance)
  const maxImp = Math.max(...rawImportances, 1e-9)
  const importances = rawImportances.map(v => v / maxImp)
  const colors = sorted.map(f => CATEGORY_COLOR[f.category] ?? '#64748b')

  const categories = [...new Set(top.map(f => f.category))]

  return (
    <div className={styles.featuresWrap}>
      <div className={styles.legendRow}>
        {categories.map(cat => (
          <div key={cat} className={styles.legendItem}>
            <span className={styles.legendDot} style={{ background: CATEGORY_COLOR[cat] ?? '#64748b' }} />
            <span className={styles.legendLabel}>{cat.replace('_', ' ')}</span>
          </div>
        ))}
      </div>
      <div className={styles.chartCard}>
        <div className={styles.chartTitle}>Top 25 Features — Mean |SHAP| (normalized)</div>
        <Plot
          data={[{
            type: 'bar',
            orientation: 'h',
            x: importances,
            y: featureNames,
            marker: { color: colors },
            hovertemplate: '<b>%{y}</b><br>Relative importance: %{x:.2f}<extra></extra>',
          }]}
          layout={{
            ...PLOT_LAYOUT_BASE,
            height: Math.max(380, sorted.length * 22),
            margin: { t: 20, r: 30, b: 40, l: 160 },
            yaxis: {
              ...PLOT_LAYOUT_BASE.yaxis,
              gridcolor: '#1e2d45',
              tickfont: { family: "'JetBrains Mono', monospace", size: 10 },
            },
            xaxis: {
              ...PLOT_LAYOUT_BASE.xaxis,
              title: { text: 'Relative Importance (normalized to top feature)', standoff: 8 },
              range: [0, 1.05],
            },
          }}
          config={PLOT_CONFIG}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  )
}

function EmptyState() {
  const steps = [
    'alpha ml dataset — build historical feature dataset (7-year OHLCV)',
    'alpha ml train — train LightGBM model with Optuna tuning',
    'alpha ml backtest — compute IC, equity curve, and feature importance',
  ]
  return (
    <div className={styles.emptyAll}>
      <div className={styles.emptyTitle}>No Backtest Data Available</div>
      <div className={styles.emptySteps}>
        {steps.map((s, i) => (
          <div key={i} className={styles.step}>
            <span className={styles.stepNum}>{i + 1}</span>
            <code className={styles.stepCode}>{s}</code>
          </div>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Help drawer content
// ---------------------------------------------------------------------------

const ML_HELP_SECTIONS = [
  {
    id: 'ic',
    title: 'Information Coefficient (IC)',
    content: (
      <div>
        <p><strong>Rank IC</strong> is the Spearman rank correlation between predicted scores and actual 10-day forward returns — computed cross-sectionally (stock vs. stock) on each snapshot date. It ranges from −1 to +1.</p>
        <p><strong>IC = 0.10</strong> is considered good in academic literature. Above 0.15 is strong. The model currently sits at ~0.16 mean monthly IC on the test set.</p>
        <p><strong>ICIR</strong> (IC Information Ratio) = IC Mean / IC Std. It measures consistency. An ICIR above 0.5 suggests a real signal; above 1.0 is considered excellent. Higher ICIR is more important than raw IC, because it means the model doesn't blow up in bad months.</p>
        <p><strong>IC Hit Rate</strong> = fraction of months with positive IC. A random model would hit ~50%. The target is above 60%.</p>
        <p><strong>IC t-stat</strong> = (IC Mean / IC Std) × √n_months. Above 2.0 = statistically significant at 95% confidence.</p>
        <p><strong>Cumulative IC</strong> = running sum of monthly ICs over time. A steadily rising line means the model is consistently predictive, not just lucky in one or two months.</p>
        <p><strong>Persistence IC</strong> = rank IC of lagged return_pct (pure momentum) against forward returns. It serves as a naive baseline — the model should beat it.</p>
      </div>
    ),
  },
  {
    id: 'features',
    title: 'Feature Categories',
    content: (
      <div>
        <p>The model uses <strong>58 features</strong> across four categories, shown in the Feature Importance chart by color.</p>
        <p><strong>Technical (blue)</strong> — OHLCV-derived, computed point-in-time from price history: return_pct, Sharpe, Sortino, RSI(7/14/21), MACD, Bollinger %B, Stochastic, ROC(5/10/21), ATR(14), SMA ratios, drawdown metrics, volume trend.</p>
        <p><strong>Market Context (purple)</strong> — same value for every stock on a given date. SPY return (10d/21d), SPY volatility, SPY RSI, SPY above 200-day SMA, and the <em>momentum regime signal</em> (SPY 10-day Sharpe-like ratio). These features are winsorized but not rank-normalized, since rank normalization would zero them out.</p>
        <p><strong>Fundamental (amber)</strong> — quarterly earnings yield, book-to-market, sales-to-price, ROE, debt/equity, revenue growth, profit margin, FCF yield, current ratio, short % float. Loaded from the fundamentals CSV (~1,150 tickers). Missing values are imputed with the cross-sectional median.</p>
        <p><strong>Derived (green)</strong> — cross-sectional alpha features: <em>return_vs_sector</em> (return_pct minus sector median, isolates alpha from beta) and <em>return_pct × spy_momentum_regime</em> (interaction term — momentum signal conditioned on whether the market is in a trend vs. reversal regime).</p>
        <p><strong>SHAP values</strong> (SHapley Additive exPlanations) measure each feature's average contribution to the model's output. Higher = more impact. Computed on a 2,000-row random sample from the test set.</p>
      </div>
    ),
  },
  {
    id: 'equity',
    title: 'Equity & Risk Charts',
    content: (
      <div>
        <p><strong>Long-Short (L/S) Portfolio</strong> — on each 10-day snapshot, go long the top decile (D10) and short the bottom decile (D1), equal-weighted within each leg. A 10 bps round-trip transaction cost is applied per period. This is a paper portfolio — it does not account for borrow cost on shorts, slippage, or liquidity constraints.</p>
        <p><strong>Cumulative Return (×)</strong> — portfolio value starting at 1.0. The curve reflects compounded 10-day returns. Note: the annualized return shown in the KPI strip uses raw forward returns winsorized at the 1st/99th percentile to reduce impact of extreme outliers.</p>
        <p><strong>Drawdown (Underwater)</strong> — the percentage decline from the most recent equity peak. A value of −20% means the portfolio is currently 20% below its all-time high. Recovery means the underwater curve returns to zero.</p>
        <p><strong>Decile Returns</strong> — stocks ranked into 10 equal buckets by predicted score. D1 = lowest predicted (short leg), D10 = highest predicted (long leg). A working model shows a monotonically increasing spread from D1 through D10. Flat or noisy deciles suggest weak signal.</p>
        <p><strong>L/S Sharpe</strong> = annualized Sharpe of the L/S strategy. Computed as (mean 10-day L/S return / std) × √(252/10). Above 1.0 is good; above 2.0 is excellent for a pure quant strategy.</p>
        <p><strong>Max Drawdown</strong> = the largest peak-to-trough decline in the equity curve over the entire test period.</p>
      </div>
    ),
  },
  {
    id: 'leaderboard',
    title: 'Model Leaderboard',
    content: (
      <div>
        <p>Each row is a trained model checkpoint. Click a row to load its charts in the other tabs.</p>
        <p><strong>IC Mean</strong> — average monthly rank IC on the test set. Higher is better; aim for &gt;0.05.</p>
        <p><strong>ICIR</strong> — IC divided by its standard deviation. Measures consistency. Aim for &gt;0.5.</p>
        <p><strong>Hit Rate</strong> — % of months with positive IC. &gt;60% suggests a reliable signal.</p>
        <p><strong>t-stat</strong> — statistical significance of the IC. &gt;2.0 = significant at 95% confidence.</p>
        <p><strong>L/S Sharpe</strong> — annualized Sharpe of the long-short paper portfolio.</p>
        <p><strong>L/S Ann. Ret</strong> — annualized return of the L/S portfolio after 10 bps transaction costs.</p>
        <p><strong>Max DD</strong> — maximum drawdown of the L/S equity curve.</p>
        <p><strong>Features</strong> — number of input features used by the model.</p>
        <p><strong>Test Dates</strong> — number of 10-day snapshot dates in the test split.</p>
      </div>
    ),
  },
  {
    id: 'methodology',
    title: 'Methodology & Caveats',
    content: (
      <div>
        <p><strong>Train / Val / Test split</strong> — time-based, no shuffling. Train: earliest 70%, Val: next 15%, Test: most recent 15%. This prevents look-ahead bias. The model never sees future data during training.</p>
        <p><strong>Cross-sectional rank normalization</strong> — on each date, each feature is ranked across all stocks and rescaled to (−1, +1). This removes absolute price-level effects and makes features comparable across regimes. Market context features (same value for all stocks) are excluded from this step.</p>
        <p><strong>Forward label</strong> = forward_10d_rank: the 10-day forward return, rank-normalized cross-sectionally on each date. Predicting ranks rather than raw returns reduces sensitivity to outliers.</p>
        <p><strong>Survivorship bias</strong> — the dataset only contains tickers present in the current NASDAQ/NYSE/S&P 500 universe. Stocks that were delisted (often due to failure) are missing from earlier dates. This inflates historical performance estimates.</p>
        <p><strong>Regime coverage</strong> — the 7-year lookback (2018–2025) covers four distinct regimes: 2018 rate hike selloff, 2020 COVID crash/recovery, 2021–2022 bull/bear cycle, 2023–2025 recovery. More regime diversity = better generalization.</p>
        <p><strong>L/S portfolio is a paper backtest</strong> — actual execution would incur higher costs (short borrow, market impact, rebalancing friction). Treat Sharpe and return figures as upper bounds.</p>
      </div>
    ),
  },
]

// ---------------------------------------------------------------------------
// Page root
// ---------------------------------------------------------------------------

export default function BacktestPage() {
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<ModelDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<Tab>('leaderboard')

  useEffect(() => {
    fetchJson<LeaderboardEntry[]>(`${BASE}/leaderboard.json`)
      .then(data => {
        setLeaderboard(data)
        if (data.length > 0) setSelectedId(data[0].model_id)
      })
      .catch(() => setLeaderboard([]))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!selectedId) return
    setDetailLoading(true)
    const base = `${BASE}/${selectedId}`
    Promise.all([
      fetchJson<IcEntry[]>(`${base}/ic_over_time.json`),
      fetchJson<CumIcEntry[]>(`${base}/cumulative_ic.json`),
      fetchJson<EquityEntry[]>(`${base}/equity_curve.json`),
      fetchJson<DecileEntry[]>(`${base}/decile_returns.json`),
      fetchJson<FeatureEntry[]>(`${base}/feature_importance.json`),
    ])
      .then(([icOverTime, cumIc, equityCurve, decileReturns, featureImportance]) =>
        setDetail({ icOverTime, cumIc, equityCurve, decileReturns, featureImportance })
      )
      .catch(() => setDetail(null))
      .finally(() => setDetailLoading(false))
  }, [selectedId])

  const selectedEntry = leaderboard.find(e => e.model_id === selectedId) ?? null

  const TABS: { id: Tab; label: string }[] = [
    { id: 'leaderboard', label: 'Leaderboard' },
    { id: 'ic', label: 'IC Over Time' },
    { id: 'equity', label: 'Equity & Risk' },
    { id: 'features', label: 'Feature Importance' },
  ]

  if (loading) {
    return (
      <AppShell showSidebar={false}>
        <div className={styles.page}><LoadingPanel /></div>
        <HelpDrawer sections={ML_HELP_SECTIONS} />
      </AppShell>
    )
  }

  if (leaderboard.length === 0) {
    return (
      <AppShell showSidebar={false}>
        <div className={styles.page}><EmptyState /></div>
        <HelpDrawer sections={ML_HELP_SECTIONS} />
      </AppShell>
    )
  }

  return (
    <AppShell showSidebar={false}>
      <div className={styles.page}>
        <div className={styles.pageHeader}>
          <h1 className={styles.pageTitle}>ML Model Evaluation</h1>
          {selectedEntry && (
            <span className={styles.modelBadge}>{selectedEntry.model_id}</span>
          )}
          {selectedEntry && (
            <span className={styles.periodBadge}>
              Test period: {selectedEntry.test_period.start} → {selectedEntry.test_period.end}
              &nbsp;·&nbsp;{selectedEntry.n_test_dates} dates
              &nbsp;·&nbsp;{selectedEntry.n_test_rows.toLocaleString()} rows
            </span>
          )}
        </div>

        {selectedEntry && <KpiStrip e={selectedEntry} />}

        <div className={styles.tabBar}>
          {TABS.map(({ id, label }) => (
            <button
              key={id}
              className={`${styles.tab} ${activeTab === id ? styles.tabActive : ''}`}
              onClick={() => setActiveTab(id)}
            >
              {label}
            </button>
          ))}
        </div>

        <div className={styles.tabContent}>
          {activeTab === 'leaderboard' && (
            <LeaderboardTab
              entries={leaderboard}
              selectedId={selectedId}
              onSelect={id => { setSelectedId(id); setActiveTab('ic') }}
            />
          )}
          {activeTab === 'ic' && (
            detailLoading ? <LoadingPanel /> :
            detail ? <IcTab data={detail} /> :
            <EmptyPanel label="No IC data available" />
          )}
          {activeTab === 'equity' && (
            detailLoading ? <LoadingPanel /> :
            detail ? <EquityTab data={detail} /> :
            <EmptyPanel label="No equity data available" />
          )}
          {activeTab === 'features' && (
            detailLoading ? <LoadingPanel /> :
            detail ? <FeaturesTab data={detail} /> :
            <EmptyPanel label="No feature importance data available" />
          )}
        </div>
      </div>
      <HelpDrawer sections={ML_HELP_SECTIONS} />
    </AppShell>
  )
}
