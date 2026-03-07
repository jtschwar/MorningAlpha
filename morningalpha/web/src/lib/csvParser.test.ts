import { describe, it, expect } from 'vitest'
import { parseCSV, parseCSVLine } from './csvParser'

const MINIMAL_CSV = `Rank,Ticker,Name,Exchange,Return_3M_%,SharpeRatio,MaxDrawdown,QualityScore,EntryScore
1,AAPL,Apple Inc,NASDAQ,25.5,1.2,-15.3,78.0,65.0
2,MSFT,Microsoft Corp,NASDAQ,18.2,0.9,-12.1,82.5,70.0`

describe('parseCSVLine', () => {
  it('splits simple fields', () => {
    expect(parseCSVLine('a,b,c')).toEqual(['a', 'b', 'c'])
  })

  it('handles quoted fields with commas inside', () => {
    expect(parseCSVLine('a,"b,c",d')).toEqual(['a', 'b,c', 'd'])
  })

  it('handles empty fields', () => {
    expect(parseCSVLine('a,,c')).toEqual(['a', '', 'c'])
  })
})

describe('parseCSV', () => {
  it('returns correctly typed Stock array', () => {
    const { data, metadata } = parseCSV(MINIMAL_CSV)
    expect(data).toHaveLength(2)
    expect(metadata.metric).toBe('3M')
    expect(metadata.totalAnalyzed).toBe(2)

    const aapl = data[0]
    expect(aapl.Ticker).toBe('AAPL')
    expect(aapl.Name).toBe('Apple Inc')
    expect(aapl.Exchange).toBe('NASDAQ')
    expect(aapl.ReturnPct).toBe(25.5)
    expect(aapl.SharpeRatio).toBe(1.2)
    expect(aapl.MaxDrawdown).toBe(-15.3)
    expect(aapl.QualityScore).toBe(78.0)
  })

  it('attaches computed scores at parse time', () => {
    const { data } = parseCSV(MINIMAL_CSV)
    expect(data[0].investmentScore).not.toBeNull()
    expect(data[0].riskRewardRatio).not.toBeNull()
    expect(data[0].riskLevel).not.toBe('unknown')
  })

  it('handles missing/null numeric fields gracefully', () => {
    const csv = `Rank,Ticker,Name,Exchange,Return_3M_%\n1,XYZ,XYZ Corp,NYSE,10.0`
    const { data } = parseCSV(csv)
    expect(data[0].SharpeRatio).toBeNull()
    expect(data[0].MaxDrawdown).toBeNull()
    expect(data[0].QualityScore).toBeNull()
  })

  it('skips rows with fewer than 5 fields', () => {
    const csv = `Rank,Ticker,Name,Exchange,Return_3M_%\n1,AAPL,Apple,NASDAQ,25.5\nbad`
    const { data } = parseCSV(csv)
    expect(data).toHaveLength(1)
  })

  it('detects return metric from column name', () => {
    const csv = `Rank,Ticker,Name,Exchange,Return_6M_%\n1,AAPL,Apple,NASDAQ,25.5`
    const { metadata } = parseCSV(csv)
    expect(metadata.metric).toBe('6M')
  })
})
