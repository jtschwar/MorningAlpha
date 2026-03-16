import { useEffect, useState } from 'react'
import AppShell from '../../components/layout/AppShell'
import styles from './Backtest.module.css'

interface BacktestData {
  filters: Record<string, unknown>
  lookback: string
  forwardWindow: string
  matchCount: number
  avgReturn: number
  medianReturn: number
  winRate: number
  avgSharpe: number
  alphaVsBenchmark: number
  equityCurve: { date: string; strategy: number; benchmark: number }[]
  monthlyReturns: { year: number; month: number; return: number }[]
  signals: { date: string; ticker: string; entryPrice: number; forwardReturn: number; hit: boolean }[]
}

interface MLResult {
  model: string
  rank_ic: number
  mae_return: number
  rmse_return: number
  hit_rate: number
  directional_accuracy: number
  feature_importance: { feature: string; importance: number; category: string }[]
  quintile_returns: { quintile: number; avg_return: number }[]
  trained_at: string
  status: string
}

export default function BacktestPage() {
  const [backtestData, setBacktestData] = useState<BacktestData | null>(null)
  const [mlResult, setMlResult] = useState<MLResult | null>(null)
  const [loadingBacktest, setLoadingBacktest] = useState(true)
  const [loadingML, setLoadingML] = useState(true)

  useEffect(() => {
    fetch('./data/latest/backtest_results.json')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(setBacktestData)
      .catch(() => setBacktestData(null))
      .finally(() => setLoadingBacktest(false))

    fetch('./data/latest/ml_results.json')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(setMlResult)
      .catch(() => setMlResult(null))
      .finally(() => setLoadingML(false))
  }, [])

  const bothEmpty = !loadingBacktest && !loadingML && !backtestData && !mlResult

  return (
    <AppShell showSidebar={false}>
      <div className={styles.page}>
        <h1 className={styles.pageTitle}>Backtest &amp; ML Evaluation</h1>

        {bothEmpty && (
          <div className={styles.emptyAll}>
            <div className={styles.emptyTitle}>No Data Available</div>
            <div className={styles.emptySteps}>
              <div className={styles.step}>
                <span className={styles.stepNum}>1</span>
                <span><code>alpha build-dataset</code> — build historical feature dataset</span>
              </div>
              <div className={styles.step}>
                <span className={styles.stepNum}>2</span>
                <span><code>alpha train</code> — train LightGBM model</span>
              </div>
              <div className={styles.step}>
                <span className={styles.stepNum}>3</span>
                <span><code>alpha evaluate</code> — evaluate model performance</span>
              </div>
              <div className={styles.step}>
                <span className={styles.stepNum}>4</span>
                <span><code>alpha predict</code> — generate live signals</span>
              </div>
            </div>
          </div>
        )}

        {/* Strategy Backtest Section */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Strategy Backtest</h2>
          {loadingBacktest ? (
            <div className={styles.loading}>Loading...</div>
          ) : !backtestData ? (
            <div className={styles.empty}>
              <p>Run <code>alpha build-dataset</code> to enable strategy backtesting.</p>
              <p className={styles.emptyHint}>
                This builds a historical dataset of technical and fundamental features that can be used to
                test whether screener filter combinations predict forward returns.
              </p>
            </div>
          ) : (
            <BacktestResults data={backtestData} />
          )}
        </section>

        {/* ML Evaluation Section */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>ML Model Evaluation</h2>
          {loadingML ? (
            <div className={styles.loading}>Loading...</div>
          ) : !mlResult ? (
            <div className={styles.empty}>
              <p>No models evaluated yet.</p>
              <p className={styles.emptyHint}>
                Run <code>alpha train</code> then <code>alpha evaluate</code> to evaluate model performance.
              </p>
            </div>
          ) : (
            <MLResults data={mlResult} />
          )}
        </section>
      </div>
    </AppShell>
  )
}

function BacktestResults({ data }: { data: BacktestData }) {
  return (
    <div>
      <div className={styles.statsStrip}>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Matches</span>
          <span className={styles.statVal}>{data.matchCount}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Avg Return</span>
          <span className={`${styles.statVal} ${data.avgReturn >= 0 ? styles.pos : styles.neg}`}>
            {data.avgReturn > 0 ? '+' : ''}{data.avgReturn.toFixed(1)}%
          </span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Median Return</span>
          <span className={`${styles.statVal} ${data.medianReturn >= 0 ? styles.pos : styles.neg}`}>
            {data.medianReturn > 0 ? '+' : ''}{data.medianReturn.toFixed(1)}%
          </span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Win Rate</span>
          <span className={styles.statVal}>{data.winRate.toFixed(1)}%</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Avg Sharpe</span>
          <span className={styles.statVal}>{data.avgSharpe.toFixed(2)}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>vs Benchmark</span>
          <span className={`${styles.statVal} ${data.alphaVsBenchmark >= 0 ? styles.pos : styles.neg}`}>
            {data.alphaVsBenchmark > 0 ? '+' : ''}{data.alphaVsBenchmark.toFixed(1)}%
          </span>
        </div>
      </div>
      {data.signals && data.signals.length > 0 && (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Date</th>
                <th>Ticker</th>
                <th>Entry</th>
                <th>Forward Return</th>
                <th>Hit?</th>
              </tr>
            </thead>
            <tbody>
              {data.signals.slice(0, 50).map((sig, i) => (
                <tr key={i}>
                  <td>{sig.date}</td>
                  <td>{sig.ticker}</td>
                  <td>${sig.entryPrice.toFixed(2)}</td>
                  <td className={sig.forwardReturn >= 0 ? styles.pos : styles.neg}>
                    {sig.forwardReturn > 0 ? '+' : ''}{sig.forwardReturn.toFixed(1)}%
                  </td>
                  <td>{sig.hit ? '✓' : '✗'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function MLResults({ data }: { data: MLResult }) {
  return (
    <div>
      <div className={styles.statsStrip}>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Model</span>
          <span className={styles.statVal}>{data.model}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Rank IC</span>
          <span className={styles.statVal}>{data.rank_ic.toFixed(3)}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>MAE</span>
          <span className={styles.statVal}>{data.mae_return.toFixed(1)}%</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Hit Rate</span>
          <span className={styles.statVal}>{data.hit_rate.toFixed(1)}%</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Dir. Accuracy</span>
          <span className={styles.statVal}>{data.directional_accuracy.toFixed(1)}%</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Status</span>
          <span className={`${styles.statVal} ${data.status === 'live' ? styles.pos : ''}`}>
            {data.status.toUpperCase()}
          </span>
        </div>
      </div>
      {data.feature_importance && data.feature_importance.length > 0 && (
        <div className={styles.featureImportance}>
          <div className={styles.subTitle}>Top Feature Importances</div>
          {data.feature_importance.slice(0, 15).map((f, i) => (
            <div key={i} className={styles.featureRow}>
              <span className={styles.featureName}>{f.feature}</span>
              <div className={styles.featureBar}>
                <div
                  className={styles.featureFill}
                  style={{ width: `${(f.importance / data.feature_importance[0].importance) * 100}%` }}
                />
              </div>
              <span className={styles.featureVal}>{(f.importance * 100).toFixed(1)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
