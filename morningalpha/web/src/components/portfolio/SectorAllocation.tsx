import type { Holding } from '../../lib/portfolioStorage'
import type { TickerEntry } from '../../hooks/useTickerIndex'
import styles from './SectorAllocation.module.css'

const SECTOR_COLORS = [
  '#3B82F6',
  '#F59E0B',
  '#22C55E',
  '#A78BFA',
  '#EF4444',
  '#06B6D4',
  '#F97316',
  '#8B95A5',
]

interface Props {
  holdings: Holding[]
  tickerIndex: TickerEntry[]
}

export default function SectorAllocation({ holdings, tickerIndex }: Props) {
  if (holdings.length === 0) {
    return (
      <div className={styles.wrap}>
        <div className={styles.title}>Sector Allocation</div>
        <div className={styles.empty}>No holdings</div>
      </div>
    )
  }

  const sectorMap = new Map<string, number>()
  holdings.forEach(h => {
    const entry = tickerIndex.find(t => t.ticker === h.ticker)
    const sector = entry?.sector ?? 'Unknown'
    sectorMap.set(sector, (sectorMap.get(sector) ?? 0) + 1)
  })

  // Sort by count descending, cap at 7 + "Other"
  const sorted = [...sectorMap.entries()].sort((a, b) => b[1] - a[1])
  const top7 = sorted.slice(0, 7)
  const otherCount = sorted.slice(7).reduce((sum, [, c]) => sum + c, 0)
  if (otherCount > 0) top7.push(['Other', otherCount])

  const maxCount = Math.max(...top7.map(([, c]) => c), 1)

  return (
    <div className={styles.wrap}>
      <div className={styles.title}>Sector Allocation</div>
      {top7.map(([sector, count], i) => (
        <div key={sector} className={styles.row}>
          <span className={styles.sectorLabel} title={sector}>{sector}</span>
          <div className={styles.barTrack}>
            <div
              className={styles.bar}
              style={{
                width: `${(count / maxCount) * 100}%`,
                background: SECTOR_COLORS[i % SECTOR_COLORS.length],
              }}
            />
          </div>
          <span className={styles.countLabel}>{count}</span>
        </div>
      ))}
    </div>
  )
}
