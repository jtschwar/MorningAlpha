import type { Stock } from '../store/types'

/** Pure function — generates CSV string from an array of stocks. */
export function generateCsvString(stocks: Stock[]): string {
  if (stocks.length === 0) return ''

  const keys = Object.keys(stocks[0]) as (keyof Stock)[]
  const header = keys.join(',')

  const rows = stocks.map(stock =>
    keys
      .map(k => {
        const val = stock[k]
        if (val == null) return ''
        const s = String(val)
        return s.includes(',') ? `"${s}"` : s
      })
      .join(','),
  )

  return [header, ...rows].join('\n')
}

/** Triggers a browser file download. Has DOM dependency — not testable in node. */
export function downloadCsv(content: string, filename = 'stock_data.csv'): void {
  const blob = new Blob([content], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function exportStocks(stocks: Stock[], filename = 'stock_data.csv'): void {
  downloadCsv(generateCsvString(stocks), filename)
}
