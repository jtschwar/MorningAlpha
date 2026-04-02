import { useState, useEffect } from 'react'
import { Routes, Route } from 'react-router-dom'
import { StockProvider } from './store/StockContext'
import { ThemeProvider } from './store/ThemeContext'
import Dashboard from './pages/Dashboard'
import StockDetail from './pages/StockDetail'
import BacktestPage from './pages/Backtest'
import ForecastPage from './pages/Forecast'
import PortfolioPage from './pages/Portfolio'
import CommandPalette from './components/layout/CommandPalette'

export default function App() {
  const [paletteOpen, setPaletteOpen] = useState(false)

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setPaletteOpen(o => !o)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  return (
    <ThemeProvider>
      <StockProvider>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/stock/:ticker" element={<StockDetail />} />
          <Route path="/backtest" element={<BacktestPage />} />
          <Route path="/forecast" element={<ForecastPage />} />
          <Route path="/portfolio" element={<PortfolioPage />} />
        </Routes>
        <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
      </StockProvider>
    </ThemeProvider>
  )
}
