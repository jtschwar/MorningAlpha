import { calculateEMA, calculateRSISeries } from './technicals'

// SMA
export function calculateSMA(prices: number[], period: number): (number | null)[] {
  const result: (number | null)[] = new Array(prices.length).fill(null)
  for (let i = period - 1; i < prices.length; i++) {
    let sum = 0
    for (let j = i - period + 1; j <= i; j++) sum += prices[j]
    result[i] = sum / period
  }
  return result
}

// MACD: returns series arrays
export function calculateMACD(prices: number[]): {
  macd: (number | null)[]
  signal: (number | null)[]
  histogram: (number | null)[]
} {
  const ema12 = calculateEMA(prices, 12)
  const ema26 = calculateEMA(prices, 26)

  const macd: (number | null)[] = prices.map((_, i) => {
    if (ema12[i] == null || ema26[i] == null) return null
    return ema12[i]! - ema26[i]!
  })

  // EMA9 of MACD line — only where macd is non-null
  const signal: (number | null)[] = new Array(prices.length).fill(null)

  // Find first non-null macd index
  const firstValid = macd.findIndex(v => v !== null)
  if (firstValid >= 0) {
    const macdFiltered = macd.slice(firstValid).map(v => v!)
    const ema9 = calculateEMA(macdFiltered, 9)
    for (let i = 0; i < ema9.length; i++) {
      signal[firstValid + i] = ema9[i]
    }
  }

  const histogram: (number | null)[] = prices.map((_, i) => {
    if (macd[i] == null || signal[i] == null) return null
    return macd[i]! - signal[i]!
  })

  return { macd, signal, histogram }
}

// Stochastic
export function calculateStochastic(
  close: number[],
  high: number[],
  low: number[],
  period = 14
): { k: (number | null)[]; d: (number | null)[] } {
  const k: (number | null)[] = new Array(close.length).fill(null)

  for (let i = period - 1; i < close.length; i++) {
    const highestHigh = Math.max(...high.slice(i - period + 1, i + 1))
    const lowestLow = Math.min(...low.slice(i - period + 1, i + 1))
    const range = highestHigh - lowestLow
    k[i] = range === 0 ? 50 : ((close[i] - lowestLow) / range) * 100
  }

  // %D = SMA3 of %K
  const d: (number | null)[] = new Array(close.length).fill(null)
  for (let i = period + 1; i < close.length; i++) {
    const k0 = k[i - 2]
    const k1 = k[i - 1]
    const k2 = k[i]
    if (k0 != null && k1 != null && k2 != null) {
      d[i] = (k0 + k1 + k2) / 3
    }
  }

  return { k, d }
}

// ROC
export function calculateROC(prices: number[], period: number): (number | null)[] {
  const result: (number | null)[] = new Array(prices.length).fill(null)
  for (let i = period; i < prices.length; i++) {
    if (prices[i - period] !== 0) {
      result[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
    }
  }
  return result
}

// ATR
export function calculateATR(
  high: number[],
  low: number[],
  close: number[],
  period = 14
): (number | null)[] {
  const tr: number[] = [high[0] - low[0]]
  for (let i = 1; i < close.length; i++) {
    tr.push(Math.max(
      high[i] - low[i],
      Math.abs(high[i] - close[i - 1]),
      Math.abs(low[i] - close[i - 1])
    ))
  }

  // EMA of TR
  const trEMA = calculateEMA(tr, period)
  return trEMA
}

// Bollinger Bands
export function calculateBollingerBands(
  prices: number[],
  period = 20,
  stdDevMult = 2
): {
  upper: (number | null)[]
  lower: (number | null)[]
  mid: (number | null)[]
  pctB: (number | null)[]
  bandwidth: (number | null)[]
} {
  const upper: (number | null)[] = new Array(prices.length).fill(null)
  const lower: (number | null)[] = new Array(prices.length).fill(null)
  const mid: (number | null)[] = new Array(prices.length).fill(null)
  const pctB: (number | null)[] = new Array(prices.length).fill(null)
  const bandwidth: (number | null)[] = new Array(prices.length).fill(null)

  for (let i = period - 1; i < prices.length; i++) {
    const slice = prices.slice(i - period + 1, i + 1)
    const mean = slice.reduce((a, b) => a + b, 0) / period
    const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / period
    const sd = Math.sqrt(variance)

    mid[i] = mean
    upper[i] = mean + stdDevMult * sd
    lower[i] = mean - stdDevMult * sd

    const range = upper[i]! - lower[i]!
    pctB[i] = range === 0 ? 0.5 : (prices[i] - lower[i]!) / range
    bandwidth[i] = mean === 0 ? 0 : (range / mean) * 100
  }

  return { upper, lower, mid, pctB, bandwidth }
}

// OBV
export function calculateOBV(close: number[], volume: number[]): number[] {
  const obv: number[] = [0]
  for (let i = 1; i < close.length; i++) {
    if (close[i] > close[i - 1]) obv.push(obv[i - 1] + volume[i])
    else if (close[i] < close[i - 1]) obv.push(obv[i - 1] - volume[i])
    else obv.push(obv[i - 1])
  }
  return obv
}

// Relative Volume (current / SMA20 of volume)
export function calculateRelativeVolume(volume: number[], period = 20): (number | null)[] {
  const result: (number | null)[] = new Array(volume.length).fill(null)
  for (let i = period - 1; i < volume.length; i++) {
    const slice = volume.slice(i - period + 1, i + 1)
    const avg = slice.reduce((a, b) => a + b, 0) / period
    result[i] = avg === 0 ? null : volume[i] / avg
  }
  return result
}

// Re-export from technicals for convenience
export { calculateEMA, calculateRSISeries }
