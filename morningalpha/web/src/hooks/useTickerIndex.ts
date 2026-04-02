import { useEffect, useState } from 'react'

export interface TickerEntry {
  ticker: string
  name: string
  sector: string | null
  mlScore: number | null
  mlScore_breakout: number | null
  mlScore_composite: number | null
  mlScore_st: number | null
  investmentScore: number | null
}

interface UseTickerIndexResult {
  tickers: TickerEntry[]
  loading: boolean
  error: string | null
}

export function useTickerIndex(): UseTickerIndexResult {
  const [tickers, setTickers] = useState<TickerEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('./data/latest/ticker_index.json')
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json() as Promise<TickerEntry[]>
      })
      .then(data => setTickers(data))
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  return { tickers, loading, error }
}
