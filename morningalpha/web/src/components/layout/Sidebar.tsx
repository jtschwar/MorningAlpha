import { useStock } from '../../store/StockContext'
import type { FilterState } from '../../store/types'
import CsvUpload from '../common/CsvUpload'
import FilterPresets from '../common/FilterPresets'
import styles from './Sidebar.module.css'

export default function Sidebar() {
  const { state, dispatch } = useStock()
  const { filters } = state

  function set<K extends keyof FilterState>(key: K, value: FilterState[K]) {
    dispatch({ type: 'SET_FILTER', key, value })
  }

  return (
    <div className={styles.sidebar}>
      <section className={styles.section}>
        <CsvUpload />
      </section>

      <section className={styles.section}>
        <div className={styles.sectionTitle}>Presets</div>
        <FilterPresets />
      </section>

      <section className={styles.section}>
        <div className={styles.sectionTitle}>Filters</div>

        <label className={styles.label}>Exchange</label>
        <select
          className={styles.select}
          value={filters.exchange}
          onChange={e => set('exchange', e.target.value)}
        >
          <option value="">All</option>
          <option value="NASDAQ">NASDAQ</option>
          <option value="NYSE">NYSE</option>
          <option value="S&P500">S&P 500</option>
        </select>

        <label className={styles.label}>Market Cap</label>
        <select
          className={styles.select}
          value={filters.marketCapCategory}
          onChange={e => set('marketCapCategory', e.target.value)}
        >
          <option value="">All</option>
          <option value="Mega">Mega (&gt;$200B)</option>
          <option value="Large">Large ($10–200B)</option>
          <option value="Mid">Mid ($2–10B)</option>
          <option value="Small">Small ($300M–2B)</option>
          <option value="Micro">Micro (&lt;$300M)</option>
        </select>

        <label className={styles.label}>Risk Tolerance</label>
        <select
          className={styles.select}
          value={filters.riskTolerance}
          onChange={e => set('riskTolerance', e.target.value as FilterState['riskTolerance'])}
        >
          <option value="all">All</option>
          <option value="conservative">Conservative</option>
          <option value="moderate">Moderate</option>
          <option value="aggressive">Aggressive</option>
        </select>

        <label className={styles.label}>Min Quality Score</label>
        <select
          className={styles.select}
          value={filters.minQuality}
          onChange={e => set('minQuality', Number(e.target.value))}
        >
          <option value={0}>Any</option>
          <option value={40}>≥ 40</option>
          <option value={60}>≥ 60</option>
          <option value={70}>≥ 70</option>
          <option value={80}>≥ 80</option>
        </select>

        <label className={styles.label}>Max Drawdown</label>
        <select
          className={styles.select}
          value={filters.maxDrawdown}
          onChange={e => set('maxDrawdown', Number(e.target.value))}
        >
          <option value={-100}>Any</option>
          <option value={-10}>≥ -10%</option>
          <option value={-20}>≥ -20%</option>
          <option value={-30}>≥ -30%</option>
          <option value={-40}>≥ -40%</option>
        </select>

        {/* RSI Range */}
        <label className={styles.label}>RSI Range</label>
        <div className={styles.rangeRow}>
          <input
            type="number"
            className={styles.rangeInput}
            min={0} max={100}
            value={filters.rsiMin}
            onChange={e => set('rsiMin', Number(e.target.value))}
            placeholder="Min"
          />
          <span className={styles.rangeSep}>–</span>
          <input
            type="number"
            className={styles.rangeInput}
            min={0} max={100}
            value={filters.rsiMax}
            onChange={e => set('rsiMax', Number(e.target.value))}
            placeholder="Max"
          />
        </div>

        {/* SMA Position */}
        <label className={styles.label}>SMA Position</label>
        <select className={styles.select} value={filters.smaPosition} onChange={e => set('smaPosition', e.target.value)}>
          <option value="">Any</option>
          <option value="above_sma50">Above SMA 50</option>
          <option value="below_sma50">Below SMA 50</option>
          <option value="above_sma200">Above SMA 200</option>
          <option value="below_sma200">Below SMA 200</option>
        </select>

        {/* Stochastic */}
        <label className={styles.label}>Stochastic</label>
        <select className={styles.select} value={filters.stochastic} onChange={e => set('stochastic', e.target.value)}>
          <option value="">Any</option>
          <option value="overbought">Overbought (&gt;80)</option>
          <option value="oversold">Oversold (&lt;20)</option>
          <option value="neutral">Neutral (20–80)</option>
        </select>

        {/* Min Sharpe */}
        <label className={styles.label}>Min Sharpe</label>
        <input
          type="number"
          className={styles.select}
          step="0.1"
          value={filters.minSharpe <= -999 ? '' : filters.minSharpe}
          placeholder="Any"
          onChange={e => set('minSharpe', e.target.value === '' ? -999 : Number(e.target.value))}
        />

        <label className={styles.label}>Sort By</label>
        <select
          className={styles.select}
          value={filters.sortBy}
          onChange={e => set('sortBy', e.target.value as FilterState['sortBy'])}
        >
          <option value="investmentScore">Investment Score</option>
          <option value="return">Return %</option>
          <option value="quality">Quality Score</option>
          <option value="sharpe">Sharpe Ratio</option>
          <option value="riskReward">Risk/Reward</option>
          <option value="marketCap">Market Cap</option>
          <option value="entryScore">Entry Score</option>
          <option value="momentumAccel">Momentum</option>
          <option value="maxDrawdown">Max Drawdown (mildest first)</option>
        </select>

        <button
          className={styles.resetBtn}
          onClick={() => dispatch({ type: 'RESET_FILTERS' })}
        >
          Clear Filters
        </button>
      </section>
    </div>
  )
}
