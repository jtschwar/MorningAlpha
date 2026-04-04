import { useState, useCallback } from 'react'
import AppShell from '../../components/layout/AppShell'
import PortfolioTabs from '../../components/portfolio/PortfolioTabs'
import PortfolioKPIStrip from '../../components/portfolio/PortfolioKPIStrip'
import AddHoldingRow from '../../components/portfolio/AddHoldingRow'
import HoldingsTable from '../../components/portfolio/HoldingsTable'
import SectorAllocation from '../../components/portfolio/SectorAllocation'
import MLScoreDistribution from '../../components/portfolio/MLScoreDistribution'
import { useTickerIndex } from '../../hooks/useTickerIndex'
import {
  loadStore,
  saveStore,
  exportAsJSON,
  exportAsCSV,
} from '../../lib/portfolioStorage'
import type { Holding, PortfolioStore } from '../../lib/portfolioStorage'
import styles from './Portfolio.module.css'

function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export default function PortfolioPage() {
  const { tickers: tickerIndex, loading: tickerLoading } = useTickerIndex()
  const [store, setStore] = useState<PortfolioStore>(() => loadStore())

  const activePortfolio = store.portfolios.find(p => p.id === store.activePortfolioId)
  const holdings = activePortfolio?.holdings ?? []

  const updateStore = useCallback((updater: (s: PortfolioStore) => PortfolioStore) => {
    setStore(prev => {
      const next = updater(prev)
      saveStore(next)
      return next
    })
  }, [])

  function handleAddHolding(holding: Holding) {
    updateStore(s => ({
      ...s,
      portfolios: s.portfolios.map(p =>
        p.id === s.activePortfolioId
          ? { ...p, holdings: [...p.holdings, holding] }
          : p
      ),
    }))
  }

  function handleDeleteHolding(id: string) {
    updateStore(s => ({
      ...s,
      portfolios: s.portfolios.map(p =>
        p.id === s.activePortfolioId
          ? { ...p, holdings: p.holdings.filter(h => h.id !== id) }
          : p
      ),
    }))
  }

  function handleCreatePortfolio(name: string) {
    const id = crypto.randomUUID()
    updateStore(s => ({
      ...s,
      portfolios: [...s.portfolios, { id, name, holdings: [], createdAt: new Date().toISOString() }],
      activePortfolioId: id,
    }))
  }

  function handleRenamePortfolio(id: string, name: string) {
    updateStore(s => ({
      ...s,
      portfolios: s.portfolios.map(p => p.id === id ? { ...p, name } : p),
    }))
  }

  function handleDeletePortfolio(id: string) {
    updateStore(s => {
      const remaining = s.portfolios.filter(p => p.id !== id)
      return {
        ...s,
        portfolios: remaining,
        activePortfolioId: s.activePortfolioId === id ? (remaining[0]?.id ?? null) : s.activePortfolioId,
      }
    })
  }

  function handleSwitchPortfolio(id: string) {
    updateStore(s => ({ ...s, activePortfolioId: id }))
  }

  function handleExportJSON() {
    const json = exportAsJSON(store)
    downloadFile(json, 'morningalpha_portfolio.json', 'application/json')
  }

  function handleExportCSV() {
    if (!activePortfolio) return
    const csv = exportAsCSV(activePortfolio)
    downloadFile(csv, `${activePortfolio.name.replace(/\s+/g, '_')}.csv`, 'text/csv')
  }

  return (
    <AppShell showSidebar={false}>
      <div className={styles.page}>
        {/* Page intro */}
        <div className={styles.pageIntro}>
          <div className={styles.introTop}>
            <p className={styles.introDesc}>
              Track holdings and monitor ML scores across your portfolio. Add positions by ticker
              — scores update automatically from the latest scoring run. Use sector allocation and
              score distribution to spot concentration risk and identify where the models see the
              strongest setups.
            </p>
            <div className={styles.exportGroup}>
              <button className={styles.exportBtn} onClick={handleExportJSON}>
                Export JSON
              </button>
              <button className={styles.exportBtn} onClick={handleExportCSV} disabled={!activePortfolio}>
                Export CSV
              </button>
            </div>
          </div>
        </div>

        {/* List switcher */}
        <PortfolioTabs
          portfolios={store.portfolios}
          activeId={store.activePortfolioId}
          onSwitch={handleSwitchPortfolio}
          onCreate={handleCreatePortfolio}
          onRename={handleRenamePortfolio}
          onDelete={handleDeletePortfolio}
        />

        {/* KPI strip */}
        <PortfolioKPIStrip holdings={holdings} tickerIndex={tickerIndex} />

        {/* Main split layout */}
        <div className={styles.splitRow}>
          <div className={styles.splitLeft}>
            <AddHoldingRow
              onAdd={handleAddHolding}
            />
            <HoldingsTable
              holdings={holdings}
              tickerIndex={tickerIndex}
              onDelete={handleDeleteHolding}
            />
          </div>

          <div className={styles.splitRight}>
            <SectorAllocation holdings={holdings} tickerIndex={tickerIndex} />
            <MLScoreDistribution holdings={holdings} tickerIndex={tickerIndex} />
          </div>
        </div>

        {tickerLoading && (
          <div style={{ color: 'var(--text-muted)', fontFamily: "'IBM Plex Mono', monospace", fontSize: '0.78rem' }}>
            Loading ticker data…
          </div>
        )}
      </div>
    </AppShell>
  )
}
