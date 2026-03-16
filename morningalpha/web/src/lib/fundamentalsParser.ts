import type { FundamentalData } from '../store/types'

function parseNum(val: string | undefined): number | null {
  if (!val || val.trim() === '' || val.trim() === '—' || val.trim() === 'nan' || val.trim() === 'NaN') return null
  const n = parseFloat(val.trim())
  return isNaN(n) ? null : n
}

function parseStr(val: string | undefined, fallback = ''): string {
  if (!val || val.trim() === '' || val.trim() === 'nan') return fallback
  return val.trim()
}

export function parseFundamentalsCSV(csv: string): Record<string, FundamentalData> {
  const lines = csv.trim().split('\n')
  if (lines.length < 2) return {}

  const headers = lines[0].split(',').map(h => h.trim())
  const col = (name: string) => headers.findIndex(h => h.toLowerCase() === name.toLowerCase())

  const tickerIdx = col('Ticker')
  const sectorIdx = col('Sector')
  const industryIdx = col('Industry')
  const marketCapIdx = col('MarketCap')
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

  const result: Record<string, FundamentalData> = {}

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => v.trim())
    if (values.length < 3) continue
    const ticker = values[tickerIdx]?.trim()
    if (!ticker) continue

    result[ticker] = {
      sector: parseStr(values[sectorIdx]),
      industry: parseStr(values[industryIdx]),
      marketCap: parseNum(values[marketCapIdx]),
      pe: parseNum(values[peIdx]),
      forwardPe: parseNum(values[forwardPeIdx]),
      pb: parseNum(values[pbIdx]),
      ps: parseNum(values[psIdx]),
      peg: parseNum(values[pegIdx]),
      eps: parseNum(values[epsIdx]),
      revenueGrowth: parseNum(values[revenueGrowthIdx]),
      earningsGrowth: parseNum(values[earningsGrowthIdx]),
      roe: parseNum(values[roeIdx]),
      roa: parseNum(values[roaIdx]),
      grossMargin: parseNum(values[grossMarginIdx]),
      operatingMargin: parseNum(values[operatingMarginIdx]),
      netMargin: parseNum(values[netMarginIdx]),
      debtEquity: parseNum(values[debtEquityIdx]),
      currentRatio: parseNum(values[currentRatioIdx]),
      divYield: parseNum(values[divYieldIdx]),
      beta: parseNum(values[betaIdx]),
      shortFloat: parseNum(values[shortFloatIdx]),
      instOwnership: parseNum(values[instOwnershipIdx]),
    }
  }

  return result
}
