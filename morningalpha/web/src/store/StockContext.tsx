import { createContext, useContext, useReducer, useMemo } from 'react'
import type { ReactNode, Dispatch as ReactDispatch } from 'react'
import type { AppState, Stock } from './types'
import { stockReducer, initialState } from './stockReducer'
import type { Action } from './stockReducer'
import { applyFilters } from '../lib/scoring'

interface StockContextValue {
  state: AppState
  dispatch: ReactDispatch<Action>
  rawData: Stock[]
  filteredData: Stock[]
}

const StockContext = createContext<StockContextValue | null>(null)

export function StockProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(stockReducer, initialState)

  const rawData = state.windowData[state.activePeriod]

  // Merge fundamentals onto each stock so applyFilters can access them
  const mergedData = useMemo(() => {
    if (!state.fundamentals) return rawData
    return rawData.map(stock => ({
      ...stock,
      fundamentals: state.fundamentals![stock.Ticker] ?? null,
    }))
  }, [rawData, state.fundamentals])

  const filteredData = useMemo(
    () => applyFilters(mergedData, state.filters),
    [mergedData, state.filters],
  )

  const value = useMemo(
    () => ({ state, dispatch, rawData: mergedData, filteredData }),
    [state, dispatch, mergedData, filteredData],
  )

  return <StockContext.Provider value={value}>{children}</StockContext.Provider>
}

export function useStock(): StockContextValue {
  const ctx = useContext(StockContext)
  if (!ctx) throw new Error('useStock must be used within StockProvider')
  return ctx
}
