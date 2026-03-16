import { describe, it, expect } from 'vitest'
import { parseFundamentalsCSV } from './fundamentalsParser'

const SAMPLE_CSV = `Ticker,Sector,Industry,MarketCap,PE,ForwardPE,PB,PS,PEG,EPS,RevenueGrowth,EarningsGrowth,ROE,ROA,GrossMargin,OperatingMargin,NetMargin,DebtEquity,CurrentRatio,DivYield,Beta,ShortFloat,InstOwnership
AAPL,Technology,Consumer Electronics,2800000000000,29.5,27.0,45.0,7.5,2.1,6.13,0.08,0.10,1.47,0.28,0.43,0.30,0.25,170,1.5,0.0058,1.22,0.007,0.61
MSFT,Technology,Software,3100000000000,35.2,30.0,12.0,11.0,2.3,11.08,0.16,0.20,0.38,0.22,0.70,0.42,0.36,38,1.8,0.0073,0.90,0.004,0.72
MISSING,,,,,,,,,,,,,,,,,,,,,`

describe('parseFundamentalsCSV', () => {
  it('parses valid CSV', () => {
    const result = parseFundamentalsCSV(SAMPLE_CSV)
    expect(result['AAPL']).toBeDefined()
    expect(result['AAPL'].sector).toBe('Technology')
    expect(result['AAPL'].pe).toBe(29.5)
    expect(result['AAPL'].marketCap).toBe(2800000000000)
  })

  it('handles missing numeric fields as null', () => {
    const result = parseFundamentalsCSV(SAMPLE_CSV)
    expect(result['MISSING']).toBeDefined()
    expect(result['MISSING'].pe).toBeNull()
    expect(result['MISSING'].marketCap).toBeNull()
  })

  it('returns empty record for empty/invalid CSV', () => {
    expect(parseFundamentalsCSV('')).toEqual({})
    expect(parseFundamentalsCSV('Ticker\n')).toEqual({})
  })

  it('keyed by ticker', () => {
    const result = parseFundamentalsCSV(SAMPLE_CSV)
    expect(Object.keys(result)).toContain('AAPL')
    expect(Object.keys(result)).toContain('MSFT')
  })
})
