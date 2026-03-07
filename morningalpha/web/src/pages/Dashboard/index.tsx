import { useStock } from '../../store/StockContext'
import AppShell from '../../components/layout/AppShell'
import KpiStrip from '../../components/dashboard/KpiStrip'
import ChartPanel from '../../components/dashboard/ChartPanel'
import StockTable from '../../components/dashboard/StockTable'
import RecommendationCards from '../../components/dashboard/RecommendationCards'
import HelpDrawer from '../../components/common/HelpDrawer'
import styles from './Dashboard.module.css'

export default function Dashboard() {
  const { state, filteredData } = useStock()
  const meta = state.metadata[state.activePeriod]

  return (
    <AppShell showSidebar>
      <div className={styles.content}>
        {state.dataSource ? (
          <>
            <RecommendationCards stocks={filteredData} />
            <hr className={styles.divider} />
            <div className={styles.sectionHeading}>At a Glance</div>
            <KpiStrip stocks={filteredData} />
            <ChartPanel stocks={filteredData} metadata={meta} />
            <StockTable stocks={filteredData} />
          </>
        ) : (
          <div className={styles.empty}>
            <div className={styles.emptyTitle}>MorningAlpha</div>
            <div className={styles.emptyHint}>
              Fetching latest data… or upload a CSV from the sidebar.
            </div>
          </div>
        )}
      </div>
      <HelpDrawer />
    </AppShell>
  )
}
