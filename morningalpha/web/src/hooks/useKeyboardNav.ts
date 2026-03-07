import { useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Stock } from '../store/types'

/**
 * Enables J/K/Enter keyboard navigation over the stock table.
 * `focusedIndex` and `setFocusedIndex` are managed by the caller.
 */
export function useKeyboardNav(
  stocks: Stock[],
  focusedIndex: number,
  setFocusedIndex: (i: number) => void,
  enabled = true,
) {
  const navigate = useNavigate()

  const handleKey = useCallback(
    (e: KeyboardEvent) => {
      if (!enabled || stocks.length === 0) return
      // Don't fire when typing in an input
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) return

      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault()
        setFocusedIndex(Math.min(focusedIndex + 1, stocks.length - 1))
      } else if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault()
        setFocusedIndex(Math.max(focusedIndex - 1, 0))
      } else if (e.key === 'Enter' && focusedIndex >= 0) {
        const ticker = stocks[focusedIndex]?.Ticker
        if (ticker) navigate(`/stock/${ticker}`)
      }
    },
    [enabled, stocks, focusedIndex, setFocusedIndex, navigate],
  )

  useEffect(() => {
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [handleKey])
}
