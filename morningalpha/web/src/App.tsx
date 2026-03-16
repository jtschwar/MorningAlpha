import { Routes, Route } from 'react-router-dom'
import { StockProvider } from './store/StockContext'
import { ThemeProvider } from './store/ThemeContext'
import Dashboard from './pages/Dashboard'
import StockDetail from './pages/StockDetail'
import BacktestPage from './pages/Backtest'

export default function App() {
  return (
    <ThemeProvider>
      <StockProvider>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/stock/:ticker" element={<StockDetail />} />
          <Route path="/backtest" element={<BacktestPage />} />
        </Routes>
      </StockProvider>
    </ThemeProvider>
  )
}
