import { useState, useEffect } from 'react'
import { useStock } from '../store/StockContext'
import type { StockDetailData } from '../store/types'

interface Result {
  data: StockDetailData | null
  loading: boolean
  error: string | null
}

export function useStockData(ticker: string | undefined, period: string): Result {
  const { state, dispatch } = useStock()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const cacheKey = `${ticker}_${period}`
  const cached = ticker ? state.apiCache.get(cacheKey) ?? null : null

  useEffect(() => {
    if (!ticker || cached) return

    let cancelled = false
    setLoading(true)
    setError(null)

    const baseUrl = import.meta.env.VITE_PROXY_URL as string
    const url = `${baseUrl}/api/stock/${ticker}?period=${period}`

    fetch(url)
      .then(async res => {
        // Parse JSON safely — a non-JSON body (e.g. HTML error page from Vite when
        // the proxy isn't running) would otherwise throw a cryptic browser-specific
        // error ("The string did not match the expected pattern." in Safari).
        let json: Record<string, unknown>
        try {
          json = await res.json()
        } catch {
          throw new Error('proxy-unreachable')
        }
        if (!res.ok || json.error) {
          if (res.status === 429) throw new Error('API rate limit reached — try again later')
          if (res.status === 404) throw new Error('No data available for this ticker')
          throw new Error((json.error as string) ?? `HTTP ${res.status}`)
        }
        return json as unknown as StockDetailData
      })
      .then(data => {
        if (cancelled) return
        dispatch({ type: 'CACHE_API_RESPONSE', key: cacheKey, data })
      })
      .catch(err => {
        if (cancelled) return
        const msg: string = err.message ?? String(err)
        if (msg === 'proxy-unreachable' || msg.includes('fetch') || msg.includes('network') || msg.includes('Failed')) {
          setError('Could not reach proxy server. Start it with `alpha launch`.')
        } else {
          setError(msg)
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [ticker, period, cached, cacheKey, dispatch])

  return { data: cached, loading, error }
}
