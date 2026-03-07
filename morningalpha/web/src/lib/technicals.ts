// ── EMA ─────────────────────────────────────────────────────────────────────

export function calculateEMA(prices: number[], period: number): (number | null)[] {
  const ema: (number | null)[] = []
  const multiplier = 2 / (period + 1)

  // Fill nulls until we have enough data, compute SMA as seed
  let sum = 0
  for (let i = 0; i < Math.min(period, prices.length); i++) {
    sum += prices[i]
    ema.push(null)
  }

  if (prices.length >= period) {
    ema[period - 1] = sum / period
    for (let i = period; i < prices.length; i++) {
      ema[i] = (prices[i] - ema[i - 1]!) * multiplier + ema[i - 1]!
    }
  }

  return ema
}

// ── RSI ──────────────────────────────────────────────────────────────────────

/** Final RSI value for a price series. Returns null if insufficient data. */
export function calculateRSI(prices: number[], period = 14): number | null {
  if (prices.length < period + 1) return null

  const changes: number[] = []
  for (let i = 1; i < prices.length; i++) changes.push(prices[i] - prices[i - 1])

  let avgGain = 0
  let avgLoss = 0
  for (let i = 0; i < period; i++) {
    if (changes[i] > 0) avgGain += changes[i]
    else avgLoss += Math.abs(changes[i])
  }
  avgGain /= period
  avgLoss /= period

  for (let i = period; i < changes.length; i++) {
    const c = changes[i]
    if (c > 0) {
      avgGain = (avgGain * (period - 1) + c) / period
      avgLoss = (avgLoss * (period - 1)) / period
    } else {
      avgGain = (avgGain * (period - 1)) / period
      avgLoss = (avgLoss * (period - 1) + Math.abs(c)) / period
    }
  }

  if (avgLoss === 0) return 100
  return 100 - 100 / (1 + avgGain / avgLoss)
}

/** Rolling RSI series — null for first `period` entries. Used for RSI chart. */
export function calculateRSISeries(prices: number[], period = 14): (number | null)[] {
  const result: (number | null)[] = new Array(prices.length).fill(null)
  if (prices.length < period + 1) return result

  const changes: number[] = []
  for (let i = 1; i < prices.length; i++) changes.push(prices[i] - prices[i - 1])

  let avgGain = 0
  let avgLoss = 0
  for (let i = 0; i < period; i++) {
    if (changes[i] > 0) avgGain += changes[i]
    else avgLoss += Math.abs(changes[i])
  }
  avgGain /= period
  avgLoss /= period

  result[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss)

  for (let i = period; i < changes.length; i++) {
    const c = changes[i]
    if (c > 0) {
      avgGain = (avgGain * (period - 1) + c) / period
      avgLoss = (avgLoss * (period - 1)) / period
    } else {
      avgGain = (avgGain * (period - 1)) / period
      avgLoss = (avgLoss * (period - 1) + Math.abs(c)) / period
    }
    result[i + 1] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss)
  }

  return result
}

// ── Volatility ───────────────────────────────────────────────────────────────

/**
 * Annualized volatility as a percentage.
 * Computed as: stddev(daily log returns) * sqrt(252) * 100
 * Fixed from old bug (which used price std-dev in dollars).
 */
export function calculateAnnualizedVolatility(prices: number[]): number {
  if (prices.length < 2) return 0

  const returns: number[] = []
  for (let i = 1; i < prices.length; i++) {
    if (prices[i - 1] > 0) returns.push(Math.log(prices[i] / prices[i - 1]))
  }

  if (returns.length === 0) return 0

  const mean = returns.reduce((a, b) => a + b, 0) / returns.length
  const variance = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length
  return Math.sqrt(variance) * Math.sqrt(252) * 100
}

// ── Max Drawdown ─────────────────────────────────────────────────────────────

/** Returns max drawdown as a negative percentage (e.g. -23.4). */
export function calculateMaxDrawdown(prices: number[]): number {
  if (prices.length < 2) return 0
  let peak = prices[0]
  let maxDD = 0
  for (const p of prices) {
    if (p > peak) peak = p
    const dd = (p - peak) / peak
    if (dd < maxDD) maxDD = dd
  }
  return maxDD * 100
}
