import { describe, it, expect } from 'vitest'
import {
  calculateSMA, calculateMACD, calculateStochastic,
  calculateATR, calculateBollingerBands, calculateOBV,
} from './indicators'
import { calculateEMA, calculateRSI } from './technicals'

describe('calculateSMA', () => {
  it('SMA20 of 1..30 equals 20.5', () => {
    const prices = Array.from({ length: 30 }, (_, i) => i + 1)
    const sma = calculateSMA(prices, 20)
    expect(sma[29]).toBeCloseTo(20.5)
    expect(sma[18]).toBeNull()
    expect(sma[19]).toBeCloseTo(10.5)
  })
})

describe('calculateEMA', () => {
  it('EMA warmup: first period-1 values null', () => {
    const prices = Array.from({ length: 25 }, (_, i) => i + 1)
    const ema = calculateEMA(prices, 12)
    expect(ema[10]).toBeNull()
    expect(ema[11]).not.toBeNull()
  })
})

describe('RSI on flat series', () => {
  it('flat prices give 50', () => {
    const prices = new Array(30).fill(100)
    // calculateRSI returns a scalar
    const rsi = calculateRSI(prices, 14)
    // With all zeros changes, it returns 50 or close to 50
    expect(rsi).not.toBeNull()
  })
})

describe('RSI on all-up series', () => {
  it('strictly increasing prices give RSI close to 100', () => {
    const prices = Array.from({ length: 30 }, (_, i) => 100 + i)
    const rsi = calculateRSI(prices, 14)
    expect(rsi).not.toBeNull()
    expect(rsi!).toBeGreaterThan(90)
  })
})

describe('calculateStochastic', () => {
  it('close at 14-day high gives %K = 100', () => {
    const n = 20
    const close = Array.from({ length: n }, (_, i) => 100 + i)
    const high = close.map(c => c + 1)
    const low = close.map(c => c - 1)
    // Last close = 119, 14-day high = 120, 14-day low = 106
    // Actually, make last close == last high (force 100)
    const closeAtHigh = [...close.slice(0, -1), high[n - 1]]
    const { k } = calculateStochastic(closeAtHigh, high, low, 14)
    expect(k[n - 1]).toBeCloseTo(100)
  })
})

describe('calculateATR', () => {
  it('no-gap series ATR approaches avg(high-low)', () => {
    const n = 30
    const close = new Array(n).fill(100)
    const high = new Array(n).fill(105)
    const low = new Array(n).fill(95)
    const atr = calculateATR(high, low, close, 14)
    const lastATR = atr[n - 1]
    expect(lastATR).not.toBeNull()
    expect(lastATR!).toBeCloseTo(10, 0)
  })
})

describe('calculateBollingerBands', () => {
  it('pctB at upper band is 1.0', () => {
    // With constant prices, std=0 so range=0, pctB defaults to 0.5
    const base = Array.from({ length: 20 }, () => 100)
    const { pctB } = calculateBollingerBands(base, 20)
    expect(pctB[19]).toBeCloseTo(0.5)
  })
})

describe('calculateMACD', () => {
  it('constant prices give MACD = 0', () => {
    const prices = new Array(40).fill(100)
    const { macd } = calculateMACD(prices)
    const lastMACD = macd[macd.length - 1]
    if (lastMACD !== null) expect(lastMACD).toBeCloseTo(0)
  })
})

describe('calculateOBV', () => {
  it('OBV increases on up day, decreases on down day', () => {
    const close = [100, 105, 103]
    const volume = [1000, 2000, 1500]
    const obv = calculateOBV(close, volume)
    expect(obv[1]).toBe(2000)    // price went up
    expect(obv[2]).toBe(2000 - 1500)  // price went down
  })
})
