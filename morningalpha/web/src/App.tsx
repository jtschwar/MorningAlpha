import { Routes, Route } from 'react-router-dom'
import { StockProvider } from './store/StockContext'
import Dashboard from './pages/Dashboard'
import StockDetail from './pages/StockDetail'

export default function App() {
  return (
    <StockProvider>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/stock/:ticker" element={<StockDetail />} />
      </Routes>
    </StockProvider>
  )
}
