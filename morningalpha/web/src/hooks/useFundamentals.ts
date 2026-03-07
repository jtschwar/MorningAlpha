import { useState, useEffect } from 'react'

export interface FundamentalsData {
  pe:            number | null
  forwardPE:     number | null
  pb:            number | null
  debtToEquity:  number | null  // already a percentage, e.g. 45.2 means 45.2%
  netMargin:     number | null  // decimal, e.g. 0.15 = 15%
  revenueGrowth: number | null  // decimal, e.g. 0.08 = 8%
  roe:           number | null  // decimal, e.g. 0.18 = 18%
  dividendYield: number | null  // decimal, e.g. 0.02 = 2%
}

interface Result {
  data: FundamentalsData | null
  loading: boolean
}

export function useFundamentals(ticker: string | undefined): Result {
  const [data, setData] = useState<FundamentalsData | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!ticker) return
    let cancelled = false
    setLoading(true)
    setData(null)

    const baseUrl = import.meta.env.VITE_PROXY_URL as string
    fetch(`${baseUrl}/api/fundamentals/${ticker}`)
      .then(res => res.json())
      .then((json: FundamentalsData & { error?: string }) => {
        if (cancelled || json.error) return
        setData(json)
      })
      .catch(() => { /* silent — fundamentals are best-effort */ })
      .finally(() => { if (!cancelled) setLoading(false) })

    return () => { cancelled = true }
  }, [ticker])

  return { data, loading }
}
