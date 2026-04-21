import { describe, it, expect } from 'vitest'
import { generateCsvString } from './exportCsv'
import type { Stock } from '../store/types'

function makeStock(overrides: Partial<Stock> = {}): Stock {
  return {
    Rank: 1,
    Ticker: 'AAPL',
    Name: 'Apple Inc',
    Exchange: 'NASDAQ',
    ReturnPct: 25.5,
    SharpeRatio: 1.2,
    SortinoRatio: 1.5,
    MaxDrawdown: -15.3,
    ConsistencyScore: 60,
    VolumeTrend: null,
    QualityScore: 78,
    RSI: 55,
    MomentumAccel: 0.5,
    PriceVs20dHigh: 0.95,
    VolumeSurge: 1.1,
    EntryScore: 65,
    MarketCap: 3e12,
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
  investmentScore: 82,
    riskRewardRatio: 1.67,
    riskLevel: 'low',
    ...overrides,
  }
}

describe('generateCsvString', () => {
  it('returns empty string for empty array', () => {
    expect(generateCsvString([])).toBe('')
  })

  it('produces a header row from stock keys', () => {
    const csv = generateCsvString([makeStock()])
    const header = csv.split('\n')[0]
    expect(header).toContain('Ticker')
    expect(header).toContain('ReturnPct')
    expect(header).toContain('SharpeRatio')
  })

  it('produces one data row per stock', () => {
    const csv = generateCsvString([makeStock(), makeStock({ Ticker: 'MSFT' })])
    const lines = csv.split('\n')
    expect(lines).toHaveLength(3) // header + 2 rows
  })

  it('includes correct values in data row', () => {
    const csv = generateCsvString([makeStock()])
    expect(csv).toContain('AAPL')
    expect(csv).toContain('25.5')
    expect(csv).toContain('NASDAQ')
  })

  it('quotes fields that contain commas', () => {
    const csv = generateCsvString([makeStock({ Name: 'Foo, Inc' })])
    expect(csv).toContain('"Foo, Inc"')
  })

  it('renders null fields as empty strings', () => {
    const csv = generateCsvString([makeStock({ VolumeTrend: null })])
    // VolumeTrend should appear as an empty field (two consecutive commas or trailing comma)
    expect(csv).toMatch(/,,|,$/)
  })

  it('header and data rows have the same number of columns', () => {
    const csv = generateCsvString([makeStock()])
    const [header, row] = csv.split('\n')
    expect(header.split(',').length).toBe(row.split(',').length)
  })
})
