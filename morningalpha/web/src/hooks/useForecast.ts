import { useEffect, useState, useRef } from 'react'

export interface ForecastData {
  ticker: string
  horizons: number[]       // [1, 5, 10, 21, 63]
  last_price: number | null
  paths: number[][]        // [n_paths × n_horizons] — cumulative log-returns
  generated_at: string
}

interface UseForecastResult {
  data: ForecastData | null
  loading: boolean
  error: string | null
}

const baseUrl = (import.meta.env.VITE_PROXY_URL as string) ?? ''

export function useForecast(
  ticker: string | undefined,
  enabled: boolean = true,
): UseForecastResult {
  const [data, setData]       = useState<ForecastData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)
  const abortRef              = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!ticker || !enabled) {
      setData(null)
      setLoading(false)
      setError(null)
      return
    }

    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl

    setLoading(true)
    setError(null)

    fetch(`${baseUrl}/api/forecast/${ticker}`, { signal: ctrl.signal })
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json() as Promise<ForecastData & { error?: string }>
      })
      .then(json => {
        if (json.error) throw new Error(json.error)
        setData(json)
      })
      .catch(e => {
        if (e.name !== 'AbortError') setError(String(e))
      })
      .finally(() => setLoading(false))

    return () => ctrl.abort()
  }, [ticker, enabled])

  return { data, loading, error }
}
