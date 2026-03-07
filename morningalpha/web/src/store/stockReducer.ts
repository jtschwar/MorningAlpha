import type { AppState, FilterState, Stock, StockDetailData, WindowPeriod, Metadata } from './types'
import { DEFAULT_FILTERS } from './types'

// ── Actions ──────────────────────────────────────────────────────────────────

export type Action =
  | { type: 'SET_WINDOW_DATA'; period: WindowPeriod; data: Stock[]; metadata: Metadata }
  | { type: 'SET_PERIOD'; period: WindowPeriod }
  | { type: 'SET_DATA_SOURCE'; source: 'auto' | 'upload' }
  | { type: 'SET_FILTER'; key: keyof FilterState; value: FilterState[keyof FilterState] }
  | { type: 'SET_FILTERS'; filters: Partial<FilterState> }
  | { type: 'RESET_FILTERS' }
  | { type: 'CACHE_API_RESPONSE'; key: string; data: StockDetailData }

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

    default:
      return state
  }
}
