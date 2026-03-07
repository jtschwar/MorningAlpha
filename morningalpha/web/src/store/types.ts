export type WindowPeriod = '2w' | '1m' | '3m' | '6m'

export type SortKey =
  | 'investmentScore'
  | 'return'
  | 'quality'
  | 'sharpe'
  | 'riskReward'
  | 'marketCap'
  | 'entryScore'
  | 'momentumAccel'

export type RiskLevel = 'low' | 'moderate' | 'high' | 'very-high' | 'unknown'

export interface Stock {
  Rank: number
  Ticker: string
  Name: string
  Exchange: string
  ReturnPct: number
  SharpeRatio: number | null
  SortinoRatio: number | null
  MaxDrawdown: number | null
  ConsistencyScore: number | null
  VolumeTrend: number | null
  QualityScore: number | null
  RSI: number | null
  MomentumAccel: number | null
  PriceVs20dHigh: number | null
  VolumeSurge: number | null
  EntryScore: number | null
  MarketCap: number | null
  MarketCapCategory: string | null
  // Computed at load time by computeScores()
  investmentScore: number | null
  riskRewardRatio: number | null
  riskLevel: RiskLevel
}

export interface Metadata {
  metric: string
  totalAnalyzed: number
}

export interface FilterState {
  search: string
  exchange: string
  marketCapCategory: string
  riskTolerance: 'all' | 'conservative' | 'moderate' | 'aggressive'
  minQuality: number
  maxDrawdown: number
  sortBy: SortKey
}

export interface StockDetailData {
  timestamps: number[]
  open: number[]
  high: number[]
  low: number[]
  close: number[]
  volume: number[]
  period: string
}

export interface AppState {
  windowData: Record<WindowPeriod, Stock[]>
  metadata: Record<WindowPeriod, Metadata | null>
  activePeriod: WindowPeriod
  dataSource: 'auto' | 'upload' | null
  filters: FilterState
  apiCache: Map<string, StockDetailData>
}

export const DEFAULT_FILTERS: FilterState = {
  search: '',
  exchange: '',
  marketCapCategory: '',
  riskTolerance: 'all',
  minQuality: 0,
  maxDrawdown: -100,
  sortBy: 'investmentScore',
}
