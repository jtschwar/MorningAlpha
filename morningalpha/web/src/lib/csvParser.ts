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

  // New technical indicator columns
  const sma7Idx = col('SMA7')
  const sma20ColIdx = col('SMA20')
  const sma50ColIdx = col('SMA50')
  const sma200ColIdx = col('SMA200')
  const priceToSMA20PctIdx = col('PriceToSMA20Pct')
  const priceToSMA50PctIdx = col('PriceToSMA50Pct')
  const priceToSMA200PctIdx = col('PriceToSMA200Pct')
  const ema7Idx = col('EMA7')
  const ema200Idx = col('EMA200')
  const macdIdx = col('MACD')
  const macdSignalIdx = col('MACDSignal')
  const macdHistIdx = col('MACDHist')
  const rsi7Idx = col('RSI7')
  const rsi21Idx = col('RSI21')
  const stochKIdx = col('StochK')
  const stochDIdx = col('StochD')
  const roc5Idx = col('ROC5')
  const roc10Idx = col('ROC10')
  const roc21Idx = col('ROC21')
  const atr14Idx = col('ATR14')
  const bollingerPctBIdx = col('BollingerPctB')
  const bollingerBandwidthIdx = col('BollingerBandwidth')
  const annualizedVolIdx = col('AnnualizedVol')
  const obvIdx = col('OBV')
  const relativeVolumeIdx = col('RelativeVolume')
  const volumeROCIdx = col('VolumeROC')

  // Fundamental columns (now embedded in period CSVs)
  const sectorIdx = col('Sector')
  const industryIdx = col('Industry')
  const peIdx = col('PE')
  const forwardPeIdx = col('ForwardPE')
  const pbIdx = col('PB')
  const psIdx = col('PS')
  const pegIdx = col('PEG')
  const epsIdx = col('EPS')
  const revenueGrowthIdx = col('RevenueGrowth')
  const earningsGrowthIdx = col('EarningsGrowth')
  const roeIdx = col('ROE')
  const roaIdx = col('ROA')
  const grossMarginIdx = col('GrossMargin')
  const operatingMarginIdx = col('OperatingMargin')
  const netMarginIdx = col('NetMargin')
  const debtEquityIdx = col('DebtEquity')
  const currentRatioIdx = col('CurrentRatio')
  const divYieldIdx = col('DivYield')
  const betaIdx = col('Beta')
  const shortFloatIdx = col('ShortFloat')
  const instOwnershipIdx = col('InstOwnership')

  const hasFundamentals = sectorIdx >= 0

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
      SMA7: num(values, sma7Idx),
      SMA20: num(values, sma20ColIdx),
      SMA50: num(values, sma50ColIdx),
      SMA200: num(values, sma200ColIdx),
      PriceToSMA20Pct: num(values, priceToSMA20PctIdx),
      PriceToSMA50Pct: num(values, priceToSMA50PctIdx),
      PriceToSMA200Pct: num(values, priceToSMA200PctIdx),
      EMA7: num(values, ema7Idx),
      EMA200: num(values, ema200Idx),
      MACD: num(values, macdIdx),
      MACDSignal: num(values, macdSignalIdx),
      MACDHist: num(values, macdHistIdx),
      RSI7: num(values, rsi7Idx),
      RSI21: num(values, rsi21Idx),
      StochK: num(values, stochKIdx),
      StochD: num(values, stochDIdx),
      ROC5: num(values, roc5Idx),
      ROC10: num(values, roc10Idx),
      ROC21: num(values, roc21Idx),
      ATR14: num(values, atr14Idx),
      BollingerPctB: num(values, bollingerPctBIdx),
      BollingerBandwidth: num(values, bollingerBandwidthIdx),
      AnnualizedVol: num(values, annualizedVolIdx),
      OBV: num(values, obvIdx),
      RelativeVolume: num(values, relativeVolumeIdx),
      VolumeROC: num(values, volumeROCIdx),
      fundamentals: hasFundamentals ? {
        sector: str(values, sectorIdx) || '',
        industry: str(values, industryIdx) || '',
        marketCap: num(values, marketCapIdx),
        pe: num(values, peIdx),
        forwardPe: num(values, forwardPeIdx),
        pb: num(values, pbIdx),
        ps: num(values, psIdx),
        peg: num(values, pegIdx),
        eps: num(values, epsIdx),
        revenueGrowth: num(values, revenueGrowthIdx),
        earningsGrowth: num(values, earningsGrowthIdx),
        roe: num(values, roeIdx),
        roa: num(values, roaIdx),
        grossMargin: num(values, grossMarginIdx),
        operatingMargin: num(values, operatingMarginIdx),
        netMargin: num(values, netMarginIdx),
        debtEquity: num(values, debtEquityIdx),
        currentRatio: num(values, currentRatioIdx),
        divYield: num(values, divYieldIdx),
        beta: num(values, betaIdx),
        shortFloat: num(values, shortFloatIdx),
        instOwnership: num(values, instOwnershipIdx),
      } : null,
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
