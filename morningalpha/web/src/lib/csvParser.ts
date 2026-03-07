import type { Stock, Metadata } from '../store/types'
import { computeScores } from './scoring'

export interface ParseResult {
  data: Stock[]
  metadata: Metadata
}

export function parseCSV(csv: string): ParseResult {
  const lines = csv.trim().split('\n')
  const headers = lines[0].split(',').map(h => h.trim())

  const returnCol = headers.find(h => h.startsWith('Return_'))
  const metric = returnCol ? returnCol.split('_')[1] : '3M'

  const col = (name: string) => headers.findIndex(h => h.toLowerCase() === name.toLowerCase())

  const rankIdx = col('Rank')
  const tickerIdx = col('Ticker')
  const nameIdx = col('Name')
  const exchangeIdx = col('Exchange')
  const returnIdx = headers.findIndex(h => h.startsWith('Return_'))
  const sharpeIdx = col('SharpeRatio')
  const sortinoIdx = col('SortinoRatio')
  const maxDrawdownIdx = col('MaxDrawdown')
  const consistencyIdx = col('ConsistencyScore')
  const volumeTrendIdx = col('VolumeTrend')
  const qualityIdx = col('QualityScore')
  const marketCapIdx = col('MarketCap')
  const marketCapCategoryIdx = col('MarketCapCategory')
  const rsiIdx = col('RSI')
  const momentumAccelIdx = col('MomentumAccel')
  const priceVsHighIdx = col('PriceVs20dHigh')
  const volumeSurgeIdx = col('VolumeSurge')
  const entryScoreIdx = col('EntryScore')

  const num = (values: string[], idx: number): number | null => {
    if (idx < 0 || !values[idx]) return null
    const n = parseFloat(values[idx])
    return isNaN(n) ? null : n
  }
  const str = (values: string[], idx: number, fallback = ''): string =>
    idx >= 0 && values[idx] ? values[idx].trim() : fallback

  const rows: Stock[] = []
  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i])
    if (values.length < 5) continue

    rows.push({
      Rank: rankIdx >= 0 ? parseInt(values[rankIdx]) || i : i,
      Ticker: str(values, tickerIdx) || values[1]?.trim() || '',
      Name: str(values, nameIdx) || values[2]?.trim() || '',
      Exchange: str(values, exchangeIdx) || values[3]?.trim() || '',
      ReturnPct: returnIdx >= 0 ? parseFloat(values[returnIdx]) || 0 : parseFloat(values[4]) || 0,
      SharpeRatio: num(values, sharpeIdx),
      SortinoRatio: num(values, sortinoIdx),
      MaxDrawdown: num(values, maxDrawdownIdx),
      ConsistencyScore: num(values, consistencyIdx),
      VolumeTrend: num(values, volumeTrendIdx),
      QualityScore: num(values, qualityIdx),
      RSI: num(values, rsiIdx),
      MomentumAccel: num(values, momentumAccelIdx),
      PriceVs20dHigh: num(values, priceVsHighIdx),
      VolumeSurge: num(values, volumeSurgeIdx),
      EntryScore: num(values, entryScoreIdx),
      MarketCap: num(values, marketCapIdx),
      MarketCapCategory: marketCapCategoryIdx >= 0 ? str(values, marketCapCategoryIdx) || null : null,
      // Will be filled by computeScores
      investmentScore: null,
      riskRewardRatio: null,
      riskLevel: 'unknown',
    })
  }

  return {
    data: computeScores(rows),
    metadata: { metric, totalAnalyzed: rows.length },
  }
}

export function parseCSVLine(line: string): string[] {
  const result: string[] = []
  let current = ''
  let inQuotes = false

  for (let i = 0; i < line.length; i++) {
    const char = line[i]
    if (char === '"') {
      inQuotes = !inQuotes
    } else if (char === ',' && !inQuotes) {
      result.push(current)
      current = ''
    } else {
      current += char
    }
  }
  result.push(current)
  return result
}
