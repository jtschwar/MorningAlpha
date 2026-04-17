import { useEffect, useRef } from 'react'
import { useStock } from '../../store/StockContext'
import { parseCSV } from '../../lib/csvParser'
import type { WindowPeriod } from '../../store/types'

const PERIOD_PATHS: { period: WindowPeriod; path: string }[] = [
  { period: '3m', path: './data/latest/stocks_3m.csv' },
  { period: '2w', path: './data/latest/stocks_2w.csv' },
  { period: '1m', path: './data/latest/stocks_1m.csv' },
  { period: '6m', path: './data/latest/stocks_6m.csv' },
]

export default function DataLoader() {
  const { state, dispatch } = useStock()
  const started = useRef(false)

  useEffect(() => {
    if (started.current || state.dataSource !== null) return
    started.current = true

    dispatch({ type: 'SET_AUTO_LOAD_STATUS', status: 'loading' })

    const loadPeriod = async ({ period, path }: { period: WindowPeriod; path: string }) => {
      const res = await fetch(path)
      if (!res.ok) throw new Error(`${res.status}`)
      const text = await res.text()
      const { data, metadata } = parseCSV(text)
      dispatch({ type: 'SET_WINDOW_DATA', period, data, metadata })
      dispatch({ type: 'SET_DATA_SOURCE', source: 'auto' })
    }

    loadPeriod(PERIOD_PATHS[0])
      .then(async () => {
        let generatedAt = null
        try {
          const res = await fetch('./data/latest/_generated.json')
          if (res.ok) {
            const { generated_at } = await res.json()
            const d = new Date(generated_at)
            generatedAt = {
              date: d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
              time: d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true }),
            }
          }
        } catch { /* no timestamp */ }
        dispatch({ type: 'SET_AUTO_LOAD_STATUS', status: 'loaded', generatedAt })
        for (const p of PERIOD_PATHS.slice(1)) {
          loadPeriod(p).catch(() => {})
        }
      })
      .catch(() => dispatch({ type: 'SET_AUTO_LOAD_STATUS', status: 'failed' }))
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return null
}
