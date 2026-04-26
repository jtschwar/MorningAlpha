import { describe, it, expect } from 'vitest'
import {
  calculateInvestmentScore,
  calculateRiskLevel,
  calculateRiskRewardRatio,
  sortStocks,
} from './scoring'
import type { Stock } from '../store/types'

function makeStock(overrides: Partial<Stock> = {}): Stock {
  return {
    Rank: 1,
    Ticker: 'TEST',
    Name: 'Test Corp',
    Exchange: 'NASDAQ',
    ReturnPct: 20,
    SharpeRatio: 1.0,
    SortinoRatio: 1.2,
    MaxDrawdown: -15,
    ConsistencyScore: 60,
    VolumeTrend: null,
    QualityScore: 70,
    RSI: 55,
    MomentumAccel: 0.5,
    PriceVs20dHigh: 0.95,
    VolumeSurge: 1.1,
    EntryScore: 65,
    MarketCap: 1e10,
    MarketCapCategory: 'Large',
    SMA7: null, SMA20: null, SMA50: null, SMA200: null,
    PriceToSMA20Pct: null, PriceToSMA50Pct: null, PriceToSMA200Pct: null,
    EMA7: null, EMA200: null,
    MACD: null, MACDSignal: null, MACDHist: null,
    RSI7: null, RSI21: null,
    StochK: null, StochD: null,
    ROC5: null, ROC10: null, ROC21: null,
    ATR14: null,
    BollingerPctB: null, BollingerBandwidth: null,
    AnnualizedVol: null,
    OBV: null, RelativeVolume: null, VolumeROC: null,
    fundamentals: null,
    mlScore: null,
    BreakoutProb63d: null,
    BreakoutProb252d50: null,
    BreakoutProb252d100: null,
  investmentScore: null,
    avgScore: null,
    riskRewardRatio: null,
    riskLevel: 'unknown',
    ...overrides,
  }
}

describe('calculateInvestmentScore', () => {
  it('returns value in 0–100 range', () => {
    const score = calculateInvestmentScore(makeStock())
    expect(score).not.toBeNull()
    expect(score!).toBeGreaterThanOrEqual(0)
    expect(score!).toBeLessThanOrEqual(100)
  })

  it('returns null when ReturnPct is null', () => {
    const score = calculateInvestmentScore(makeStock({ ReturnPct: null as unknown as number }))
    expect(score).toBeNull()
  })

  it('scores a high-quality stock higher than a low-quality stock', () => {
    const high = calculateInvestmentScore(
      makeStock({ QualityScore: 95, SharpeRatio: 2.5, MaxDrawdown: -5, ReturnPct: 50 }),
    )!
    const low = calculateInvestmentScore(
      makeStock({ QualityScore: 20, SharpeRatio: -0.5, MaxDrawdown: -45, ReturnPct: 5 }),
    )!
    expect(high).toBeGreaterThan(low)
  })
})

describe('calculateRiskLevel', () => {
  it('returns very-high for severe drawdown', () => {
    expect(calculateRiskLevel(makeStock({ MaxDrawdown: -50, SharpeRatio: 0.5 }))).toBe('very-high')
  })

  it('returns very-high for very negative Sharpe', () => {
    expect(calculateRiskLevel(makeStock({ MaxDrawdown: -10, SharpeRatio: -2 }))).toBe('very-high')
  })

  it('returns low for good Sharpe and mild drawdown', () => {
    expect(calculateRiskLevel(makeStock({ MaxDrawdown: -12, SharpeRatio: 1.5 }))).toBe('low')
  })

  it('returns high for drawdown < -30', () => {
    expect(calculateRiskLevel(makeStock({ MaxDrawdown: -35, SharpeRatio: 0.5 }))).toBe('high')
  })
})

describe('calculateRiskRewardRatio', () => {
  it('divides return by abs(drawdown)', () => {
    const ratio = calculateRiskRewardRatio(makeStock({ ReturnPct: 30, MaxDrawdown: -15 }))
    expect(ratio).toBeCloseTo(2.0)
  })

  it('returns null when ReturnPct is null', () => {
    expect(calculateRiskRewardRatio(makeStock({ ReturnPct: null as unknown as number }))).toBeNull()
  })
})

describe('sortStocks', () => {
  const stocks = [
    makeStock({ Ticker: 'A', SharpeRatio: 0.5, riskRewardRatio: 1.0, MarketCap: 1e9, EntryScore: 40, investmentScore: 50 }),
    makeStock({ Ticker: 'B', SharpeRatio: 2.0, riskRewardRatio: 3.0, MarketCap: 5e9, EntryScore: 80, investmentScore: 80 }),
    makeStock({ Ticker: 'C', SharpeRatio: -0.3, riskRewardRatio: 0.5, MarketCap: 2e8, EntryScore: 20, investmentScore: 30 }),
  ]

  it('sorts by sharpe correctly (regression test for sort bug)', () => {
    const sorted = sortStocks(stocks, 'sharpe')
    expect(sorted.map(s => s.Ticker)).toEqual(['B', 'A', 'C'])
  })

  it('sorts by riskReward correctly', () => {
    const sorted = sortStocks(stocks, 'riskReward')
    expect(sorted.map(s => s.Ticker)).toEqual(['B', 'A', 'C'])
  })

  it('sorts by marketCap correctly', () => {
    const sorted = sortStocks(stocks, 'marketCap')
    expect(sorted.map(s => s.Ticker)).toEqual(['B', 'A', 'C'])
  })

  it('sorts by entryScore correctly', () => {
    const sorted = sortStocks(stocks, 'entryScore')
    expect(sorted.map(s => s.Ticker)).toEqual(['B', 'A', 'C'])
  })

  it('sorts by investmentScore correctly', () => {
    const sorted = sortStocks(stocks, 'investmentScore')
    expect(sorted.map(s => s.Ticker)).toEqual(['B', 'A', 'C'])
  })
})
