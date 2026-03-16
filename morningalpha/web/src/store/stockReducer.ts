import type { AppState, FilterState, Stock, StockDetailData, WindowPeriod, Metadata, FundamentalData, ColumnConfig, BacktestResult } from './types'
import { DEFAULT_FILTERS, DEFAULT_VISIBLE_COLUMNS } from './types'

// ── localStorage helpers ───────────────────────────────────────────────────

const LS_COLUMNS_KEY = 'ma-visible-columns'

function loadColumnConfig(): ColumnConfig {
  try {
    const stored = localStorage.getItem(LS_COLUMNS_KEY)
    if (stored) {
      const parsed = JSON.parse(stored)
      if (Array.isArray(parsed)) return { visibleColumns: parsed }
    }
  } catch {}
  return { visibleColumns: DEFAULT_VISIBLE_COLUMNS }
}

// ── Actions ──────────────────────────────────────────────────────────────────

export type Action =
  | { type: 'SET_WINDOW_DATA'; period: WindowPeriod; data: Stock[]; metadata: Metadata }
  | { type: 'SET_PERIOD'; period: WindowPeriod }
  | { type: 'SET_DATA_SOURCE'; source: 'auto' | 'upload' }
  | { type: 'SET_FILTER'; key: keyof FilterState; value: FilterState[keyof FilterState] }
  | { type: 'SET_FILTERS'; filters: Partial<FilterState> }
  | { type: 'RESET_FILTERS' }
  | { type: 'CACHE_API_RESPONSE'; key: string; data: StockDetailData }
  | { type: 'SET_FUNDAMENTALS'; data: Record<string, FundamentalData> }
  | { type: 'SET_COLUMN_CONFIG'; columns: string[] }
  | { type: 'RESET_COLUMN_CONFIG' }
  | { type: 'SET_BACKTEST_RESULTS'; data: BacktestResult }

// ── Initial state ─────────────────────────────────────────────────────────────

const emptyWindow = { '2w': [], '1m': [], '3m': [], '6m': [] }
const emptyMeta = { '2w': null, '1m': null, '3m': null, '6m': null }

export const initialState: AppState = {
  windowData: emptyWindow,
  metadata: emptyMeta,
  activePeriod: '3m',
  dataSource: null,
  filters: DEFAULT_FILTERS,
  apiCache: new Map(),
  fundamentals: null,
  columnConfig: loadColumnConfig(),
  backtestResults: null,
}

// ── Reducer ───────────────────────────────────────────────────────────────────

export function stockReducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_WINDOW_DATA':
      return {
        ...state,
        windowData: { ...state.windowData, [action.period]: action.data },
        metadata: { ...state.metadata, [action.period]: action.metadata },
      }

    case 'SET_PERIOD':
      return { ...state, activePeriod: action.period }

    case 'SET_DATA_SOURCE':
      return { ...state, dataSource: action.source }

    case 'SET_FILTER':
      return {
        ...state,
        filters: { ...state.filters, [action.key]: action.value },
      }

    case 'SET_FILTERS':
      return {
        ...state,
        filters: { ...state.filters, ...action.filters },
      }

    case 'RESET_FILTERS':
      return { ...state, filters: DEFAULT_FILTERS }

    case 'CACHE_API_RESPONSE': {
      const newCache = new Map(state.apiCache)
      newCache.set(action.key, action.data)
      return { ...state, apiCache: newCache }
    }

    case 'SET_FUNDAMENTALS':
      return { ...state, fundamentals: action.data }

    case 'SET_COLUMN_CONFIG': {
      try { localStorage.setItem(LS_COLUMNS_KEY, JSON.stringify(action.columns)) } catch {}
      return { ...state, columnConfig: { visibleColumns: action.columns } }
    }

    case 'RESET_COLUMN_CONFIG': {
      try { localStorage.removeItem(LS_COLUMNS_KEY) } catch {}
      return { ...state, columnConfig: { visibleColumns: DEFAULT_VISIBLE_COLUMNS } }
    }

    case 'SET_BACKTEST_RESULTS':
      return { ...state, backtestResults: action.data }

    default:
      return state
  }
}
