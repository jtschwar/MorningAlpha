import type { Stock } from '../store/types'
import type { FundamentalsData } from '../hooks/useFundamentals'
import type { StockDetailData } from '../store/types'
import { calculateEMA, calculateRSI } from './technicals'

export type SignalLevel = 'STRONG BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG SELL'

export interface Signal {
  level: SignalLevel
  score: number        // 0–100
  reasons: string[]    // top 3 drivers, positive or negative
}

/**
 * Rule-based composite signal. All inputs are optional — the signal
 * degrades gracefully if only some data is available.
 */
export function computeSignal(
  stock: Stock | null,
  fundamentals: FundamentalsData | null,
  ohlcv: StockDetailData | null,
): Signal {
  let points = 0
  let maxPoints = 0
  const pos: string[] = []
  const neg: string[] = []

  // ── CSV metrics ──────────────────────────────────────────────────
  if (stock) {
    // Investment score (0–100) → contributes up to ±20 pts
    if (stock.investmentScore != null) {
      maxPoints += 20
      const invNorm = (stock.investmentScore - 50) / 50
      points += invNorm * 20
      if (invNorm > 0.4) pos.push(`Investment score ${stock.investmentScore.toFixed(0)}`)
      else if (invNorm < -0.2) neg.push(`Weak investment score (${stock.investmentScore.toFixed(0)})`)
    }

    // Entry score (0–100) → ±15 pts
    if (stock.EntryScore != null) {
      maxPoints += 15
      const entNorm = (stock.EntryScore - 50) / 50
      points += entNorm * 15
      if (entNorm > 0.4) pos.push(`Entry score ${stock.EntryScore.toFixed(0)} — good timing`)
      else if (entNorm < -0.2) neg.push(`Poor entry timing (score ${stock.EntryScore.toFixed(0)})`)
    }

    // Quality score → ±10 pts
    if (stock.QualityScore != null) {
      maxPoints += 10
      const qualNorm = (stock.QualityScore - 50) / 50
      points += qualNorm * 10
      if (qualNorm > 0.4) pos.push(`High quality score (${stock.QualityScore.toFixed(0)})`)
    }

    // Sharpe ratio → ±10 pts
    if (stock.SharpeRatio != null) {
      maxPoints += 10
      const sharpeScore = Math.min(Math.max((stock.SharpeRatio - 0.5) / 1.5, -1), 1)
      points += sharpeScore * 10
      if (stock.SharpeRatio >= 1.5) pos.push(`Strong Sharpe (${stock.SharpeRatio.toFixed(2)})`)
      else if (stock.SharpeRatio < 0.3) neg.push(`Poor Sharpe ratio (${stock.SharpeRatio.toFixed(2)})`)
    }

    // Momentum → ±5 pts
    if (stock.MomentumAccel != null) {
      maxPoints += 5
      const momScore = Math.min(Math.max(stock.MomentumAccel / 2, -1), 1)
      points += momScore * 5
      if (stock.MomentumAccel > 1) pos.push('Momentum accelerating')
      else if (stock.MomentumAccel < -1) neg.push('Momentum decelerating')
    }
  }

  // ── Live technical indicators ─────────────────────────────────────
  if (ohlcv) {
    const prices = ohlcv.close
    const ema20 = calculateEMA(prices, 20).filter(v => v != null) as number[]
    const ema50 = calculateEMA(prices, 50).filter(v => v != null) as number[]
    const lastEma20 = ema20.at(-1) ?? null
    const lastEma50 = ema50.at(-1) ?? null
    const rsi = calculateRSI(prices, 14)

    // EMA crossover → ±10 pts
    if (lastEma20 != null && lastEma50 != null) {
      maxPoints += 10
      if (lastEma20 > lastEma50) {
        points += 10
        pos.push('EMA20 > EMA50 — short-term bullish')
      } else {
        points -= 10
        neg.push('EMA20 < EMA50 — short-term bearish')
      }
    }

    // RSI → ±8 pts
    if (rsi != null) {
      maxPoints += 8
      if (rsi < 30) {
        points += 8
        pos.push(`RSI ${rsi.toFixed(0)} — oversold, potential bounce`)
      } else if (rsi > 70) {
        points -= 8
        neg.push(`RSI ${rsi.toFixed(0)} — overbought, caution`)
      } else if (rsi < 45) {
        points -= 3
      } else if (rsi > 55) {
        points += 3
        pos.push(`RSI ${rsi.toFixed(0)} — bullish momentum`)
      }
    }
  }

  // ── Fundamentals ─────────────────────────────────────────────────
  if (fundamentals) {
    // Valuation (P/E) → ±8 pts
    if (fundamentals.pe != null && fundamentals.pe > 0) {
      maxPoints += 8
      if (fundamentals.pe < 15) { points += 8; pos.push(`P/E ${fundamentals.pe.toFixed(1)}x — undervalued`) }
      else if (fundamentals.pe < 25) { points += 3 }
      else if (fundamentals.pe > 50) { points -= 8; neg.push(`P/E ${fundamentals.pe.toFixed(1)}x — expensive`) }
      else if (fundamentals.pe > 35) { points -= 4 }
    }

    // Revenue growth → ±7 pts
    if (fundamentals.revenueGrowth != null) {
      maxPoints += 7
      if (fundamentals.revenueGrowth > 0.15) { points += 7; pos.push(`Revenue growing +${(fundamentals.revenueGrowth * 100).toFixed(0)}%`) }
      else if (fundamentals.revenueGrowth > 0.05) { points += 3 }
      else if (fundamentals.revenueGrowth < -0.05) { points -= 7; neg.push(`Revenue declining ${(fundamentals.revenueGrowth * 100).toFixed(0)}%`) }
    }

    // Debt/Equity → ±5 pts
    if (fundamentals.debtToEquity != null) {
      maxPoints += 5
      if (fundamentals.debtToEquity < 50) { points += 5; pos.push('Low debt load') }
      else if (fundamentals.debtToEquity > 300) { points -= 5; neg.push(`High debt/equity (${fundamentals.debtToEquity.toFixed(0)}%)`) }
      else if (fundamentals.debtToEquity > 150) { points -= 2 }
    }

    // Net margin → ±5 pts
    if (fundamentals.netMargin != null) {
      maxPoints += 5
      if (fundamentals.netMargin > 0.15) { points += 5; pos.push(`${(fundamentals.netMargin * 100).toFixed(0)}% net margin`) }
      else if (fundamentals.netMargin < 0) { points -= 5; neg.push('Loss-making') }
    }
  }

  // Normalise to 0–100
  const score = maxPoints > 0
    ? Math.round(Math.min(Math.max(((points + maxPoints) / (2 * maxPoints)) * 100, 0), 100))
    : 50

  const level: SignalLevel =
    score >= 72 ? 'STRONG BUY'
    : score >= 58 ? 'BUY'
    : score >= 42 ? 'HOLD'
    : score >= 28 ? 'SELL'
    : 'STRONG SELL'

  // Pick top 3 most impactful reasons (positives for buy signals, negatives for sell)
  const isBullish = score >= 50
  const primary = isBullish ? pos : neg
  const secondary = isBullish ? neg : pos
  const reasons = [...primary, ...secondary].slice(0, 3)

  return { level, score, reasons }
}

/**
 * Generate a 1–2 sentence plain-English summary for the top K stocks.
 * Uses only CSV data (no fundamentals/OHLCV needed).
 */
export function summarizeTopPicks(stocks: Stock[]): string {
  if (stocks.length === 0) return ''

  const signals = stocks.map(s => computeSignal(s, null, null))

  // Count signal levels
  const counts: Record<SignalLevel, number> = {
    'STRONG BUY': 0, 'BUY': 0, 'HOLD': 0, 'SELL': 0, 'STRONG SELL': 0,
  }
  for (const sig of signals) counts[sig.level]++

  const top = stocks[0]
  const topSig = signals[0]

  // ── Sentence 1: distribution + lead stock ────────────────────────
  const bullishCount = counts['STRONG BUY'] + counts['BUY']
  const bearishCount = counts['SELL'] + counts['STRONG SELL']

  let s1 = ''
  if (counts['STRONG BUY'] >= Math.ceil(stocks.length / 2)) {
    s1 = `Strong conviction across top ${stocks.length} — ${counts['STRONG BUY']} strong buy${counts['STRONG BUY'] > 1 ? 's' : ''}.`
  } else if (bullishCount >= Math.ceil(stocks.length * 0.6)) {
    s1 = `${bullishCount} of ${stocks.length} picks rated buy or better.`
  } else if (bearishCount >= Math.ceil(stocks.length / 2)) {
    s1 = `Mixed-to-negative signals — consider sizing down.`
  } else {
    s1 = `Mixed signals across top ${stocks.length} picks.`
  }

  // Add lead stock detail
  const returnStr = top.ReturnPct != null
    ? ` (${top.ReturnPct > 0 ? '+' : ''}${top.ReturnPct.toFixed(1)}%)`
    : ''
  s1 += ` ${top.Ticker}${returnStr} leads with a ${topSig.level.toLowerCase()} signal.`

  // ── Sentence 2: notable highlight or warning ─────────────────────
  // Find overbought stocks
  const overbought = stocks.filter(s => s.RSI != null && s.RSI > 70)
  // Find highest Sharpe
  const bestSharpe = [...stocks].sort((a, b) => (b.SharpeRatio ?? 0) - (a.SharpeRatio ?? 0))[0]
  // Find worst drawdown
  const worstDD = [...stocks].sort((a, b) => (a.MaxDrawdown ?? 0) - (b.MaxDrawdown ?? 0))[0]
  // Find accelerating momentum
  const accelerating = stocks.filter(s => s.MomentumAccel != null && s.MomentumAccel > 1)

  let s2 = ''
  if (overbought.length > 0) {
    const names = overbought.map(s => s.Ticker).join(', ')
    s2 = `Watch ${names} — RSI elevated${overbought.length === 1 && overbought[0].RSI != null ? ` (${overbought[0].RSI.toFixed(0)})` : ''}, potential near-term pullback.`
  } else if (accelerating.length >= 2) {
    s2 = `${accelerating.length} picks showing accelerating momentum — trend may continue.`
  } else if (bestSharpe.SharpeRatio != null && bestSharpe.SharpeRatio >= 1.5) {
    s2 = `${bestSharpe.Ticker} stands out for risk-adjusted returns (Sharpe ${bestSharpe.SharpeRatio.toFixed(2)}).`
  } else if (worstDD.MaxDrawdown != null && worstDD.MaxDrawdown < -40) {
    s2 = `${worstDD.Ticker} has seen a steep decline (${worstDD.MaxDrawdown.toFixed(1)}%) — higher risk.`
  }

  return s2 ? `${s1} ${s2}` : s1
}
