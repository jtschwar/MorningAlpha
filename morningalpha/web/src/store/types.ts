export type WindowPeriod = '2w' | '1m' | '3m' | '6m'

export type SortKey =
  | 'investmentScore'
  | 'mlScore'
  | 'return'
  | 'quality'
  | 'sharpe'
  | 'riskReward'
  | 'marketCap'
  | 'entryScore'
  | 'momentumAccel'
  | 'maxDrawdown'

export type RiskLevel = 'low' | 'moderate' | 'high' | 'very-high' | 'unknown'

export interface FundamentalData {
  sector: string
  industry: string
  marketCap: number | null
  pe: number | null
  forwardPe: number | null
  pb: number | null
  ps: number | null
  peg: number | null
  eps: number | null
  revenueGrowth: number | null
  earningsGrowth: number | null
  roe: number | null
  roa: number | null
  grossMargin: number | null
  operatingMargin: number | null
  netMargin: number | null
  debtEquity: number | null
  currentRatio: number | null
  divYield: number | null
  beta: number | null
  shortFloat: number | null
  instOwnership: number | null
}

export interface ColumnConfig {
  visibleColumns: string[]
}

export interface BacktestResult {
  filters: Record<string, unknown>
  lookback: string
  forwardWindow: string
  matchCount: number
  avgReturn: number
  medianReturn: number
  winRate: number
  avgSharpe: number
  alphaVsBenchmark: number
  equityCurve: { date: string; strategy: number; benchmark: number }[]
  monthlyReturns: { year: number; month: number; return: number }[]
  signals: { date: string; ticker: string; entryPrice: number; forwardReturn: number; hit: boolean }[]
}

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
  // New technical indicator columns (from CSV)
  SMA7: number | null
  SMA20: number | null
  SMA50: number | null
  SMA200: number | null
  PriceToSMA20Pct: number | null
  PriceToSMA50Pct: number | null
  PriceToSMA200Pct: number | null
  EMA7: number | null
  EMA200: number | null
  MACD: number | null
  MACDSignal: number | null
  MACDHist: number | null
  RSI7: number | null
  RSI21: number | null
  StochK: number | null
  StochD: number | null
  ROC5: number | null
  ROC10: number | null
  ROC21: number | null
  ATR14: number | null
  BollingerPctB: number | null
  BollingerBandwidth: number | null
  AnnualizedVol: number | null
  OBV: number | null
  RelativeVolume: number | null
  VolumeROC: number | null
  // Merged from fundamentals.csv at load time
  fundamentals?: FundamentalData | null
  // ML model predicted score (0–100 percentile rank, written by alpha spread)
  mlScore: number | null
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
  // New filters
  sectors: string[]
  industries: string[]
  rsiMin: number
  rsiMax: number
  smaPosition: string  // '' | 'above_sma50' | 'below_sma50' | 'above_sma200' | 'below_sma200'
  stochastic: string   // '' | 'overbought' | 'oversold' | 'neutral'
  peMin: number
  peMax: number
  betaMin: number
  betaMax: number
  dividend: string     // '' | 'has_dividend' | 'no_dividend' | 'yield_2pct' | 'yield_4pct'
  minSharpe: number
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
  columnConfig: ColumnConfig
  backtestResults: BacktestResult | null
}

export const DEFAULT_VISIBLE_COLUMNS = [
  'Rank', 'Ticker', 'Name', 'Exchange', 'Return %', 'Sharpe', 'Quality', 'Entry', 'Max DD', 'RSI', 'Score', 'ML Score'
]

export const DEFAULT_FILTERS: FilterState = {
  search: '',
  exchange: '',
  marketCapCategory: '',
  riskTolerance: 'all',
  minQuality: 0,
  maxDrawdown: -100,
  sortBy: 'investmentScore',
  sectors: [],
  industries: [],
  rsiMin: 0,
  rsiMax: 100,
  smaPosition: '',
  stochastic: '',
  peMin: 0,
  peMax: 999,
  betaMin: 0,
  betaMax: 10,
  dividend: '',
  minSharpe: -999,
}
