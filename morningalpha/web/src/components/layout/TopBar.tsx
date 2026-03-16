import { useNavigate, useLocation } from 'react-router-dom'
import { useStock } from '../../store/StockContext'
import { useTheme } from '../../store/ThemeContext'
import type { WindowPeriod } from '../../store/types'
import SearchInput from '../common/SearchInput'
import styles from './TopBar.module.css'

const PERIODS: { key: WindowPeriod; label: string }[] = [
  { key: '2w', label: '2W' },
  { key: '1m', label: '1M' },
  { key: '3m', label: '3M' },
  { key: '6m', label: '6M' },
]

interface Props {
  showHamburger: boolean
  onToggleSidebar: () => void
  sidebarOpen: boolean
}

export default function TopBar({ showHamburger, onToggleSidebar, sidebarOpen }: Props) {
  const { state, dispatch } = useStock()
  const { isDark, toggle } = useTheme()
  const navigate = useNavigate()
  const location = useLocation()
  const light = !isDark
  const { activePeriod, dataSource, metadata } = state
  const meta = metadata[activePeriod]

  const statusText = dataSource
    ? `${meta?.totalAnalyzed ?? 0} stocks · ${meta?.metric ?? activePeriod.toUpperCase()}`
    : 'Load a CSV to begin'

  return (
    <header className={styles.topbar}>
      <div className={styles.left}>
        {showHamburger && (
          <button
            className={styles.hamburger}
            onClick={onToggleSidebar}
            aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          >
            {sidebarOpen ? '✕' : '☰'}
          </button>
        )}
        <span className={styles.logo}>MorningAlpha</span>
        <span className={styles.status}>{statusText}</span>
      </div>

      <div className={styles.center}>
        <SearchInput />
      </div>

      <div className={styles.right}>
        <nav className={styles.navGroup}>
          <button
            className={`${styles.navBtn} ${location.pathname === '/' ? styles.navActive : ''}`}
            onClick={() => navigate('/')}
          >
            Screener
          </button>
          <button
            className={`${styles.navBtn} ${location.pathname === '/backtest' ? styles.navActive : ''}`}
            onClick={() => navigate('/backtest')}
          >
            Backtest
          </button>
        </nav>
        <button className={styles.themeBtn} onClick={toggle} title="Toggle light/dark mode">
          {light ? '☀' : '☾'}
        </button>
        <div className={styles.periodGroup}>
          {PERIODS.map(({ key, label }) => (
            <button
              key={key}
              className={`${styles.periodBtn} ${activePeriod === key ? styles.active : ''}`}
              onClick={() => dispatch({ type: 'SET_PERIOD', period: key })}
              title={
                key === '2w' ? 'Sharpe/RSI unreliable at 2-week lookback' : undefined
              }
            >
              {label}
              {key === '2w' && <span className={styles.warn}>⚠</span>}
            </button>
          ))}
        </div>
      </div>
    </header>
  )
}
