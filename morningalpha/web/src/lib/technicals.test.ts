import { describe, it, expect } from 'vitest'
import {
  calculateEMA,
  calculateRSI,
  calculateRSISeries,
  calculateAnnualizedVolatility,
} from './technicals'

// 20 steadily rising prices for predictable EMA
const RISING_PRICES = Array.from({ length: 30 }, (_, i) => 100 + i)
// Flat prices
const FLAT_PRICES = new Array(30).fill(100)

describe('calculateEMA', () => {
  it('returns null for first period-1 entries', () => {
    const ema = calculateEMA(RISING_PRICES, 20)
    for (let i = 0; i < 19; i++) expect(ema[i]).toBeNull()
  })

  it('seeds with SMA at index period-1', () => {
    const prices = [1, 2, 3, 4, 5]
    const ema = calculateEMA(prices, 3)
    // SMA of first 3 = (1+2+3)/3 = 2
    expect(ema[2]).toBeCloseTo(2)
  })

  it('returns an array the same length as prices', () => {
    const ema = calculateEMA(RISING_PRICES, 20)
    expect(ema).toHaveLength(RISING_PRICES.length)
  })

  it('returns all nulls when prices shorter than period', () => {
    const ema = calculateEMA([1, 2, 3], 20)
    expect(ema.every(v => v === null)).toBe(true)
  })
})

describe('calculateRSI', () => {
  it('returns null when not enough data', () => {
    expect(calculateRSI([1, 2, 3], 14)).toBeNull()
  })

  it('returns a value in [0, 100]', () => {
    const rsi = calculateRSI(RISING_PRICES, 14)
    expect(rsi).not.toBeNull()
    expect(rsi!).toBeGreaterThanOrEqual(0)
    expect(rsi!).toBeLessThanOrEqual(100)
  })

  it('returns 100 for a uniformly rising series', () => {
    // All gains, no losses → RSI should be 100
    const rsi = calculateRSI(RISING_PRICES, 14)
    expect(rsi).toBe(100)
  })
})

describe('calculateRSISeries', () => {
  it('first `period` entries are null', () => {
    const series = calculateRSISeries(RISING_PRICES, 14)
    for (let i = 0; i < 14; i++) expect(series[i]).toBeNull()
  })

  it('all non-null values are in [0, 100]', () => {
    const series = calculateRSISeries(RISING_PRICES, 14)
    for (const v of series) {
      if (v !== null) {
        expect(v).toBeGreaterThanOrEqual(0)
        expect(v).toBeLessThanOrEqual(100)
      }
    }
  })

  it('returns all nulls when prices shorter than period+1', () => {
    const series = calculateRSISeries([1, 2, 3], 14)
    expect(series.every(v => v === null)).toBe(true)
  })
})

describe('calculateAnnualizedVolatility', () => {
  it('returns 0 for a flat price series (regression test for old $ bug)', () => {
    expect(calculateAnnualizedVolatility(FLAT_PRICES)).toBe(0)
  })

  it('returns a percentage, not a dollar amount', () => {
    // $100 stock, ~1% daily moves → annualized vol should be ~16%, not ~$1
    const prices = Array.from({ length: 252 }, () =>
      100 * Math.exp(0.01 * (Math.random() - 0.5)),
    )
    const vol = calculateAnnualizedVolatility(prices)
    // Should be in reasonable % range (1–100), not in dollar terms
    expect(vol).toBeGreaterThan(0)
    expect(vol).toBeLessThan(200)
  })

  it('returns 0 for fewer than 2 prices', () => {
    expect(calculateAnnualizedVolatility([100])).toBe(0)
    expect(calculateAnnualizedVolatility([])).toBe(0)
  })
})
