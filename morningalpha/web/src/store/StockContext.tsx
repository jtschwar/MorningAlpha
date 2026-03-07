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

  const filteredData = useMemo(
    () => applyFilters(rawData, state.filters),
    [rawData, state.filters],
  )

  const value = useMemo(
    () => ({ state, dispatch, rawData, filteredData }),
    [state, dispatch, rawData, filteredData],
  )

  return <StockContext.Provider value={value}>{children}</StockContext.Provider>
}

export function useStock(): StockContextValue {
  const ctx = useContext(StockContext)
  if (!ctx) throw new Error('useStock must be used within StockProvider')
  return ctx
}
