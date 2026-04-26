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
  return stocks.map(s => {
    const investmentScore = calculateInvestmentScore(s)
    const ml = s.mlScore
    const avgScore =
      investmentScore != null && ml != null ? (investmentScore + ml) / 2
      : investmentScore ?? ml ?? null
    return {
      ...s,
      investmentScore,
      avgScore,
      riskRewardRatio: calculateRiskRewardRatio(s),
      riskLevel: calculateRiskLevel(s),
    }
  })
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

  // Sector filter
  if (filters.sectors.length > 0) {
    result = result.filter(s => s.fundamentals?.sector && filters.sectors.includes(s.fundamentals.sector))
  }

  // Industry filter
  if (filters.industries.length > 0) {
    result = result.filter(s => s.fundamentals?.industry && filters.industries.includes(s.fundamentals.industry))
  }

  // RSI range
  if (filters.rsiMin > 0 || filters.rsiMax < 100) {
    result = result.filter(s => {
      const rsi = s.RSI
      if (rsi == null) return false
      return rsi >= filters.rsiMin && rsi <= filters.rsiMax
    })
  }

  // SMA position
  if (filters.smaPosition) {
    result = result.filter(s => {
      switch (filters.smaPosition) {
        case 'above_sma50': return s.PriceToSMA50Pct != null && s.PriceToSMA50Pct > 0
        case 'below_sma50': return s.PriceToSMA50Pct != null && s.PriceToSMA50Pct < 0
        case 'above_sma200': return s.PriceToSMA200Pct != null && s.PriceToSMA200Pct > 0
        case 'below_sma200': return s.PriceToSMA200Pct != null && s.PriceToSMA200Pct < 0
        default: return true
      }
    })
  }

  // Stochastic
  if (filters.stochastic) {
    result = result.filter(s => {
      const k = s.StochK
      if (k == null) return false
      switch (filters.stochastic) {
        case 'overbought': return k > 80
        case 'oversold': return k < 20
        case 'neutral': return k >= 20 && k <= 80
        default: return true
      }
    })
  }

  // P/E range (from fundamentals)
  if (filters.peMin > 0 || filters.peMax < 999) {
    result = result.filter(s => {
      const pe = s.fundamentals?.pe
      if (pe == null) return false
      return pe >= filters.peMin && pe <= filters.peMax
    })
  }

  // Beta range (from fundamentals)
  if (filters.betaMin > 0 || filters.betaMax < 10) {
    result = result.filter(s => {
      const beta = s.fundamentals?.beta
      if (beta == null) return false
      return beta >= filters.betaMin && beta <= filters.betaMax
    })
  }

  // Dividend
  if (filters.dividend) {
    result = result.filter(s => {
      const div = s.fundamentals?.divYield
      switch (filters.dividend) {
        case 'has_dividend': return div != null && div > 0
        case 'no_dividend': return div == null || div === 0
        case 'yield_2pct': return div != null && div >= 0.02
        case 'yield_4pct': return div != null && div >= 0.04
        default: return true
      }
    })
  }

  // Min Sharpe
  if (filters.minSharpe > -999) {
    result = result.filter(s => s.SharpeRatio != null && s.SharpeRatio >= filters.minSharpe)
  }

  return sortStocks(result, filters.sortBy)
}

// ── Sorting ─────────────────────────────────────────────────────────────────

export function sortStocks(stocks: Stock[], sortBy: SortKey): Stock[] {
  const sorted = [...stocks]

  const cmp: Record<SortKey, (a: Stock, b: Stock) => number> = {
    investmentScore: (a, b) => (b.investmentScore ?? 0) - (a.investmentScore ?? 0),
    avgScore: (a, b) => (b.avgScore ?? 0) - (a.avgScore ?? 0),
    return: (a, b) => b.ReturnPct - a.ReturnPct,
    quality: (a, b) => (b.QualityScore ?? 0) - (a.QualityScore ?? 0),
    sharpe: (a, b) => (b.SharpeRatio ?? -999) - (a.SharpeRatio ?? -999),
    riskReward: (a, b) => (b.riskRewardRatio ?? 0) - (a.riskRewardRatio ?? 0),
    marketCap: (a, b) => (b.MarketCap ?? 0) - (a.MarketCap ?? 0),
    entryScore: (a, b) => (b.EntryScore ?? 0) - (a.EntryScore ?? 0),
    momentumAccel: (a, b) => (b.MomentumAccel ?? 0) - (a.MomentumAccel ?? 0),
    mlScore: (a, b) => (b.mlScore ?? -1) - (a.mlScore ?? -1),
    maxDrawdown: (a, b) => (b.MaxDrawdown ?? -100) - (a.MaxDrawdown ?? -100),
    breakoutProb63d: (a, b) => (b.BreakoutProb63d ?? -1) - (a.BreakoutProb63d ?? -1),
    breakoutProb252d100: (a, b) => (b.BreakoutProb252d100 ?? -1) - (a.BreakoutProb252d100 ?? -1),
  }

  return sorted.sort(cmp[sortBy] ?? cmp.investmentScore)
}
