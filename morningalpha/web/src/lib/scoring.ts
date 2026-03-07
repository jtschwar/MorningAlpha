import type { Stock, RiskLevel, FilterState, SortKey } from '../store/types'

// ── Score calculations ──────────────────────────────────────────────────────

export function calculateInvestmentScore(stock: Stock): number | null {
  if (stock.ReturnPct == null) return null

  let score = 0
  let factors = 0

  // Return (30%)
  score += Math.min(30, (stock.ReturnPct / 200) * 30)
  factors += 30

  // Quality (25%)
  if (stock.QualityScore != null && !isNaN(stock.QualityScore)) {
    score += (stock.QualityScore / 100) * 25
    factors += 25
  }

  // Sharpe (20%)
  if (stock.SharpeRatio != null && !isNaN(stock.SharpeRatio)) {
    score += Math.max(0, Math.min(20, ((stock.SharpeRatio + 1) / 4) * 20))
    factors += 20
  }

  // Consistency (15%)
  if (stock.ConsistencyScore != null && !isNaN(stock.ConsistencyScore)) {
    score += (stock.ConsistencyScore / 100) * 15
    factors += 15
  }

  // Drawdown penalty (10%)
  if (stock.MaxDrawdown != null && !isNaN(stock.MaxDrawdown)) {
    score += Math.max(0, Math.min(10, ((stock.MaxDrawdown + 50) / 50) * 10))
    factors += 10
  }

  if (factors === 0) return null
  return Math.round((score / factors) * 1000) / 10
}

export function calculateRiskLevel(stock: Stock): RiskLevel {
  const drawdown = stock.MaxDrawdown ?? 0
  const sharpe = stock.SharpeRatio ?? 0

  if (drawdown < -40 || sharpe < -1) return 'very-high'
  if (drawdown < -30 || sharpe < 0) return 'high'
  if (drawdown < -10 && sharpe > 1) return 'low'
  if (drawdown < -20 || sharpe < 0.5) return 'moderate'
  return 'moderate'
}

export function calculateRiskRewardRatio(stock: Stock): number | null {
  if (stock.ReturnPct == null) return null
  const drawdown = Math.abs(stock.MaxDrawdown ?? 1)
  if (drawdown === 0) return Math.abs(stock.ReturnPct)
  return Math.abs(stock.ReturnPct) / drawdown
}

/** Attach computed fields to every stock. Called once at load time. */
export function computeScores(stocks: Stock[]): Stock[] {
  return stocks.map(s => ({
    ...s,
    investmentScore: calculateInvestmentScore(s),
    riskRewardRatio: calculateRiskRewardRatio(s),
    riskLevel: calculateRiskLevel(s),
  }))
}

// ── Filtering ───────────────────────────────────────────────────────────────

export function applyFilters(stocks: Stock[], filters: FilterState): Stock[] {
  let result = stocks

  if (filters.search) {
    const q = filters.search.toLowerCase()
    result = result.filter(
      s => s.Ticker.toLowerCase().includes(q) || s.Name.toLowerCase().includes(q),
    )
  }

  if (filters.exchange) {
    result = result.filter(s => s.Exchange === filters.exchange)
  }

  if (filters.marketCapCategory) {
    result = result.filter(s => s.MarketCapCategory === filters.marketCapCategory)
  }

  if (filters.minQuality > 0) {
    result = result.filter(s => s.QualityScore != null && s.QualityScore >= filters.minQuality)
  }

  if (filters.maxDrawdown > -100) {
    result = result.filter(s => s.MaxDrawdown == null || s.MaxDrawdown >= filters.maxDrawdown)
  }

  if (filters.riskTolerance !== 'all') {
    result = result.filter(s => {
      switch (filters.riskTolerance) {
        case 'conservative':
          return s.riskLevel === 'low' || s.riskLevel === 'moderate'
        case 'moderate':
          return s.riskLevel !== 'very-high'
        case 'aggressive':
          return true
        default:
          return true
      }
    })
  }

  return sortStocks(result, filters.sortBy)
}

// ── Sorting ─────────────────────────────────────────────────────────────────

export function sortStocks(stocks: Stock[], sortBy: SortKey): Stock[] {
  const sorted = [...stocks]

  const cmp: Record<SortKey, (a: Stock, b: Stock) => number> = {
    investmentScore: (a, b) => (b.investmentScore ?? 0) - (a.investmentScore ?? 0),
    return: (a, b) => b.ReturnPct - a.ReturnPct,
    quality: (a, b) => (b.QualityScore ?? 0) - (a.QualityScore ?? 0),
    sharpe: (a, b) => (b.SharpeRatio ?? -999) - (a.SharpeRatio ?? -999),
    riskReward: (a, b) => (b.riskRewardRatio ?? 0) - (a.riskRewardRatio ?? 0),
    marketCap: (a, b) => (b.MarketCap ?? 0) - (a.MarketCap ?? 0),
    entryScore: (a, b) => (b.EntryScore ?? 0) - (a.EntryScore ?? 0),
    momentumAccel: (a, b) => (b.MomentumAccel ?? 0) - (a.MomentumAccel ?? 0),
  }

  return sorted.sort(cmp[sortBy] ?? cmp.investmentScore)
}
