export interface Holding {
  id: string
  ticker: string
  shares: number
  avgCost: number
  addedAt: string
}

export interface Portfolio {
  id: string
  name: string
  holdings: Holding[]
  createdAt: string
}

export interface PortfolioStore {
  portfolios: Portfolio[]
  activePortfolioId: string | null
  schemaVersion: 1
}

const STORAGE_KEY = 'morningalpha_portfolio_v1'

function makeDefaultStore(): PortfolioStore {
  const portfolioId = crypto.randomUUID()
  return {
    portfolios: [
      {
        id: portfolioId,
        name: 'My Portfolio',
        holdings: [],
        createdAt: new Date().toISOString(),
      },
    ],
    activePortfolioId: portfolioId,
    schemaVersion: 1,
  }
}

export function loadStore(): PortfolioStore {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return makeDefaultStore()
    const parsed = JSON.parse(raw) as PortfolioStore
    // Basic validation
    if (!parsed.portfolios || !Array.isArray(parsed.portfolios)) return makeDefaultStore()
    // Ensure activePortfolioId is set
    if (!parsed.activePortfolioId && parsed.portfolios.length > 0) {
      parsed.activePortfolioId = parsed.portfolios[0].id
    }
    return parsed
  } catch {
    return makeDefaultStore()
  }
}

export function saveStore(store: PortfolioStore): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store))
  } catch {
    // Storage quota exceeded or unavailable — silently ignore
  }
}

export function exportAsJSON(store: PortfolioStore): string {
  return JSON.stringify(store, null, 2)
}

export function exportAsCSV(portfolio: Portfolio): string {
  const header = 'ticker,shares,avgCost,addedAt'
  const rows = portfolio.holdings.map(h =>
    `${h.ticker},${h.shares},${h.avgCost},${h.addedAt}`
  )
  return [header, ...rows].join('\n')
}

export function importFromJSON(json: string): PortfolioStore {
  const parsed = JSON.parse(json) as PortfolioStore
  if (!parsed.portfolios || !Array.isArray(parsed.portfolios)) {
    throw new Error('Invalid portfolio JSON format')
  }
  return parsed
}
