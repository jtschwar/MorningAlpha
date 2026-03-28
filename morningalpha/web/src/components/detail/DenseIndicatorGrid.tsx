import { useMemo } from 'react'
import type { Stock, FundamentalData, StockDetailData } from '../../store/types'
import { calculateEMA, calculateRSISeries, calculateAnnualizedVolatility } from '../../lib/technicals'
import {
  calculateSMA,
  calculateMACD,
  calculateStochastic,
  calculateATR,
  calculateBollingerBands,
  calculateOBV,
  calculateRelativeVolume,
} from '../../lib/indicators'
import styles from './DenseIndicatorGrid.module.css'

type Period = '1M' | '2W' | '3M' | '6M' | '1Y'

interface Props {
  stock: Stock | null
  fundamentals: FundamentalData | null | undefined
  ohlcv: StockDetailData | null
  period: Period
  section?: 'overview' | 'technicals'
}

// ── Formatters ────────────────────────────────────────────────────────────────

function fmt(v: number | null, dec = 2): string {
  return v != null ? v.toFixed(dec) : '—'
}

function fmtPct(v: number | null, dec = 1, alreadyPct = false): string {
  if (v == null) return '—'
  const val = alreadyPct ? v : v * 100
  return `${val > 0 ? '+' : ''}${val.toFixed(dec)}%`
}

function fmtDollar(v: number | null, dec = 2): string {
  return v != null ? `$${v.toFixed(dec)}` : '—'
}

function fmtLarge(v: number | null): string {
  if (v == null) return '—'
  const abs = Math.abs(v)
  if (abs >= 1e12) return `$${(v / 1e12).toFixed(1)}T`
  if (abs >= 1e9) return `$${(v / 1e9).toFixed(1)}B`
  if (abs >= 1e6) return `$${(v / 1e6).toFixed(0)}M`
  return `$${v.toFixed(0)}`
}

// ── Cell helpers ──────────────────────────────────────────────────────────────

type ColorClass = '' | 'pos' | 'neg' | 'neutral' | 'warn'

interface CellDef {
  label: string
  value: string
  color?: ColorClass
}

function Cell({ label, value, color }: CellDef) {
  const colorClass = color ? styles[color] : ''
  return (
    <div className={styles.cell}>
      <span className={styles.cellLabel}>{label}</span>
      <span className={`${styles.cellVal} ${colorClass}`}>{value}</span>
    </div>
  )
}

function Row({ cells }: { cells: CellDef[] }) {
  return (
    <div className={styles.row}>
      {cells.map((c, i) => (
        <Cell key={i} {...c} />
      ))}
    </div>
  )
}

// ── RSI color ─────────────────────────────────────────────────────────────────

function rsiColor(v: number | null): ColorClass {
  if (v == null) return 'neutral'
  if (v > 70) return 'neg'
  if (v < 30) return 'warn'
  if (v >= 40 && v <= 60) return 'pos'
  return ''
}

// ── Main component ────────────────────────────────────────────────────────────

export default function DenseIndicatorGrid({ stock, fundamentals: f, ohlcv, period, section = 'overview' }: Props) {
  const hideSMA200 = period === '1M' || period === '2W'
  const hideRSI21 = period === '1M' || period === '2W'
  const hideROC21 = period === '1M' || period === '2W'
  const fastOnly = period === '2W'

  // ── Technical indicators computed from OHLCV ──────────────────────────────
  const tech = useMemo(() => {
    if (!ohlcv) return null
    const closes = ohlcv.close
    const highs = ohlcv.high
    const lows = ohlcv.low
    const volumes = ohlcv.volume

    // EMAs
    const ema7 = calculateEMA(closes, 7)
    const ema20 = calculateEMA(closes, 20)
    const ema50 = calculateEMA(closes, 50)
    const ema200 = calculateEMA(closes, 200)
    const lastEma7 = ema7.filter(v => v != null).at(-1) ?? null
    const lastEma20 = ema20.filter(v => v != null).at(-1) ?? null
    const lastEma50 = ema50.filter(v => v != null).at(-1) ?? null
    const lastEma200 = ema200.filter(v => v != null).at(-1) ?? null

    // SMAs
    const sma20 = calculateSMA(closes, 20)
    const sma50 = calculateSMA(closes, 50)
    const sma200 = calculateSMA(closes, 200)
    const lastSma20 = sma20.filter(v => v != null).at(-1) ?? null
    const lastSma50 = sma50.filter(v => v != null).at(-1) ?? null
    const lastSma200 = sma200.filter(v => v != null).at(-1) ?? null
    const lastClose = closes.at(-1) ?? null
    const sma50DistPct = lastSma50 && lastClose ? (lastClose - lastSma50) / lastSma50 * 100 : null
    const sma200DistPct = lastSma200 && lastClose ? (lastClose - lastSma200) / lastSma200 * 100 : null

    // MACD
    const { macd, signal, histogram } = calculateMACD(closes)
    const lastMACD = macd.filter(v => v != null).at(-1) ?? null
    const lastSignal = signal.filter(v => v != null).at(-1) ?? null
    const lastHistogram = histogram.filter(v => v != null).at(-1) ?? null

    // ATR & Vol
    const atr = calculateATR(highs, lows, closes, 14)
    const lastATR = atr.filter(v => v != null).at(-1) ?? null
    const annVol = calculateAnnualizedVolatility(closes)

    // RSI at multiple periods
    const rsi7Series = calculateRSISeries(closes, 7)
    const rsi14Series = calculateRSISeries(closes, 14)
    const rsi21Series = calculateRSISeries(closes, 21)
    const lastRsi7 = rsi7Series.filter(v => v != null).at(-1) ?? null
    const lastRsi14 = rsi14Series.filter(v => v != null).at(-1) ?? null
    const lastRsi21 = rsi21Series.filter(v => v != null).at(-1) ?? null

    // Stochastic
    const { k: stochK, d: stochD } = calculateStochastic(closes, highs, lows, 14)
    const lastStochK = stochK.filter(v => v != null).at(-1) ?? null
    const lastStochD = stochD.filter(v => v != null).at(-1) ?? null

    // Bollinger Bands
    const { pctB, bandwidth } = calculateBollingerBands(closes, 20, 2)
    const lastPctB = pctB.filter(v => v != null).at(-1) ?? null
    const lastBandwidth = bandwidth.filter(v => v != null).at(-1) ?? null

    // OBV
    const obv = calculateOBV(closes, volumes)
    const lastOBV = obv.at(-1) ?? null

    // Relative Volume
    const relVol = calculateRelativeVolume(volumes, 20)
    const lastRelVol = relVol.filter(v => v != null).at(-1) ?? null

    // Volume ROC
    const volROC =
      volumes.length > 21
        ? ((volumes.at(-1)! - volumes.at(-22)!) / volumes.at(-22)!) * 100
        : null

    // EMA signal
    const bullish =
      lastEma20 != null && lastEma50 != null ? lastEma20 > lastEma50 : null

    return {
      lastEma7, lastEma20, lastEma50, lastEma200, bullish,
      lastSma20, lastSma50, lastSma200, sma50DistPct, sma200DistPct,
      lastMACD, lastSignal, lastHistogram, lastATR, annVol,
      lastRsi7, lastRsi14, lastRsi21,
      lastStochK, lastStochD,
      lastPctB, lastBandwidth, lastOBV, lastRelVol, volROC,
    }
  }, [ohlcv])

  // ── Section 1: Fundamentals ───────────────────────────────────────────────

  const fundRow1: CellDef[] = [
    { label: 'P/E',     value: fmt(f?.pe ?? null),        color: f?.pe != null && f.pe > 0 ? (f.pe < 30 ? 'pos' : f.pe > 60 ? 'neg' : '') : 'neutral' },
    { label: 'FWD P/E', value: fmt(f?.forwardPe ?? null), color: f?.forwardPe != null && f.forwardPe > 0 ? (f.forwardPe < 25 ? 'pos' : f.forwardPe > 50 ? 'neg' : '') : 'neutral' },
    { label: 'P/B',     value: fmt(f?.pb ?? null),        color: f?.pb != null ? (f.pb < 3 ? 'pos' : f.pb > 10 ? 'neg' : '') : 'neutral' },
    { label: 'P/S',     value: fmt(f?.ps ?? null),        color: f?.ps != null ? (f.ps < 5 ? 'pos' : f.ps > 20 ? 'neg' : '') : 'neutral' },
    { label: 'PEG',     value: fmt(f?.peg ?? null),       color: f?.peg != null ? (f.peg > 0 && f.peg < 1.5 ? 'pos' : f.peg > 3 ? 'neg' : '') : 'neutral' },
    { label: 'MKT CAP', value: fmtLarge(f?.marketCap ?? null) },
  ]

  const fundRow2: CellDef[] = [
    { label: 'ROE',      value: fmtPct(f?.roe ?? null),            color: f?.roe != null ? (f.roe > 0.15 ? 'pos' : f.roe < 0 ? 'neg' : '') : 'neutral' },
    { label: 'ROA',      value: fmtPct(f?.roa ?? null),            color: f?.roa != null ? (f.roa > 0.05 ? 'pos' : f.roa < 0 ? 'neg' : '') : 'neutral' },
    { label: 'GROSS MRG',  value: fmtPct(f?.grossMargin ?? null),    color: f?.grossMargin != null ? (f.grossMargin > 0.4 ? 'pos' : f.grossMargin < 0.1 ? 'neg' : '') : 'neutral' },
    { label: 'OP MRG',   value: fmtPct(f?.operatingMargin ?? null), color: f?.operatingMargin != null ? (f.operatingMargin > 0.15 ? 'pos' : f.operatingMargin < 0 ? 'neg' : '') : 'neutral' },
    { label: 'NET MRG',  value: fmtPct(f?.netMargin ?? null),       color: f?.netMargin != null ? (f.netMargin > 0.1 ? 'pos' : f.netMargin < 0 ? 'neg' : '') : 'neutral' },
    { label: 'DEBT/EQ',  value: fmt(f?.debtEquity ?? null),         color: f?.debtEquity != null ? (f.debtEquity < 0.5 ? 'pos' : f.debtEquity > 2 ? 'neg' : '') : 'neutral' },
  ]

  const fundRow3: CellDef[] = [
    { label: 'DIV YIELD',  value: fmtPct(f?.divYield ?? null),      color: f?.divYield != null && f.divYield > 0.02 ? 'pos' : 'neutral' },
    { label: 'BETA',       value: fmt(f?.beta ?? null),              color: f?.beta != null ? (f.beta < 1 ? 'pos' : f.beta > 2 ? 'neg' : '') : 'neutral' },
    { label: 'SHORT FL%',  value: fmtPct(f?.shortFloat ?? null),    color: f?.shortFloat != null ? (f.shortFloat > 0.2 ? 'neg' : '') : 'neutral' },
    { label: 'INST OWN%',  value: fmtPct(f?.instOwnership ?? null), color: f?.instOwnership != null && f.instOwnership > 0.5 ? 'pos' : 'neutral' },
    { label: 'SECTOR',     value: f?.sector ?? '—' },
    { label: 'INDUSTRY',   value: f?.industry ?? '—' },
  ]

  // ── Section 2: Performance ────────────────────────────────────────────────

  const perfRow1: CellDef[] = [
    { label: 'INV. SCORE', value: fmt(stock?.investmentScore ?? null, 1),  color: stock?.investmentScore != null ? (stock.investmentScore >= 70 ? 'pos' : stock.investmentScore < 40 ? 'neg' : '') : 'neutral' },
    { label: 'QUALITY',    value: fmt(stock?.QualityScore ?? null, 1),     color: stock?.QualityScore != null ? (stock.QualityScore >= 70 ? 'pos' : stock.QualityScore < 40 ? 'neg' : '') : 'neutral' },
    { label: 'ENTRY',      value: fmt(stock?.EntryScore ?? null, 1),       color: stock?.EntryScore != null ? (stock.EntryScore >= 70 ? 'pos' : stock.EntryScore < 40 ? 'neg' : '') : 'neutral' },
    { label: 'SHARPE',     value: fmt(stock?.SharpeRatio ?? null),         color: stock?.SharpeRatio != null ? (stock.SharpeRatio >= 1 ? 'pos' : stock.SharpeRatio < 0 ? 'neg' : '') : 'neutral' },
    { label: 'SORTINO',    value: fmt(stock?.SortinoRatio ?? null),        color: stock?.SortinoRatio != null ? (stock.SortinoRatio >= 1 ? 'pos' : stock.SortinoRatio < 0 ? 'neg' : '') : 'neutral' },
    { label: 'MAX DD',     value: fmtPct(stock?.MaxDrawdown ?? null, 1, true), color: stock?.MaxDrawdown != null ? (stock.MaxDrawdown > -10 ? 'pos' : stock.MaxDrawdown < -30 ? 'neg' : '') : 'neutral' },
  ]

  const perfRow2: CellDef[] = [
    { label: 'RSI (CSV)', value: fmt(stock?.RSI ?? null, 1),    color: rsiColor(stock?.RSI ?? null) },
    { label: 'MOMENTUM',  value: fmt(stock?.MomentumAccel ?? null), color: stock?.MomentumAccel != null ? (stock.MomentumAccel > 0 ? 'pos' : stock.MomentumAccel < 0 ? 'neg' : '') : 'neutral' },
    { label: 'ROC 5D',    value: fmtPct(stock?.ROC5 ?? null, 1, true),  color: stock?.ROC5 != null ? (stock.ROC5 > 0 ? 'pos' : stock.ROC5 < 0 ? 'neg' : '') : 'neutral' },
    { label: 'ROC 10D',   value: fmtPct(stock?.ROC10 ?? null, 1, true), color: stock?.ROC10 != null ? (stock.ROC10 > 0 ? 'pos' : stock.ROC10 < 0 ? 'neg' : '') : 'neutral' },
    { label: 'ROC 21D',   value: hideROC21 ? '—' : fmtPct(stock?.ROC21 ?? null, 1, true), color: hideROC21 ? 'neutral' : (stock?.ROC21 != null ? (stock.ROC21 > 0 ? 'pos' : stock.ROC21 < 0 ? 'neg' : '') : 'neutral') },
    { label: 'ML SCORE',  value: fmt(stock?.mlScore ?? null, 1), color: stock?.mlScore != null ? (stock.mlScore >= 70 ? 'pos' : stock.mlScore < 40 ? 'neg' : '') : 'neutral' },
  ]

  // ── Section 3: Technical Indicators ──────────────────────────────────────

  const emaSignalVal = tech?.bullish == null
    ? '—'
    : tech.bullish ? 'Bullish' : 'Bearish'
  const emaSignalColor: ColorClass = tech?.bullish == null
    ? 'neutral'
    : tech.bullish ? 'pos' : 'neg'

  const techRow1: CellDef[] = [
    { label: 'EMA 7',    value: tech ? fmtDollar(tech.lastEma7) : '—',   color: fastOnly && tech?.lastEma7 == null ? 'neutral' : undefined },
    { label: 'EMA 20',   value: fastOnly ? '—' : (tech ? fmtDollar(tech.lastEma20) : '—'), color: fastOnly ? 'neutral' : undefined },
    { label: 'EMA 50',   value: fastOnly ? '—' : (tech ? fmtDollar(tech.lastEma50) : '—'), color: fastOnly ? 'neutral' : undefined },
    { label: 'EMA 200',  value: hideSMA200 ? '—' : (tech ? fmtDollar(tech.lastEma200) : '—'), color: hideSMA200 ? 'neutral' : undefined },
    { label: 'EMA SIGNAL', value: fastOnly ? '—' : emaSignalVal, color: fastOnly ? 'neutral' : emaSignalColor },
  ]

  const techRow2: CellDef[] = [
    { label: 'SMA 20',      value: fastOnly ? '—' : (tech ? fmtDollar(tech.lastSma20) : '—'), color: fastOnly ? 'neutral' : undefined },
    { label: 'SMA 50',      value: fastOnly ? '—' : (tech ? fmtDollar(tech.lastSma50) : '—'), color: fastOnly ? 'neutral' : undefined },
    { label: 'SMA 200',     value: hideSMA200 ? '—' : (tech ? fmtDollar(tech.lastSma200) : '—'), color: hideSMA200 ? 'neutral' : undefined },
    {
      label: 'SMA50 DIST%',
      value: fastOnly ? '—' : (tech ? fmtPct(tech.sma50DistPct, 1, true) : '—'),
      color: fastOnly ? 'neutral' : (tech?.sma50DistPct != null ? (tech.sma50DistPct > 0 ? 'pos' : 'neg') : 'neutral'),
    },
    {
      label: 'SMA200 DIST%',
      value: hideSMA200 ? '—' : (tech ? fmtPct(tech.sma200DistPct, 1, true) : '—'),
      color: hideSMA200 ? 'neutral' : (tech?.sma200DistPct != null ? (tech.sma200DistPct > 0 ? 'pos' : 'neg') : 'neutral'),
    },
  ]

  const techRow3: CellDef[] = [
    { label: 'MACD',     value: fastOnly ? '—' : (tech ? fmt(tech.lastMACD) : '—'),      color: fastOnly ? 'neutral' : (tech?.lastMACD != null ? (tech.lastMACD > 0 ? 'pos' : 'neg') : 'neutral') },
    { label: 'MACD SIG', value: fastOnly ? '—' : (tech ? fmt(tech.lastSignal) : '—'),    color: fastOnly ? 'neutral' : undefined },
    { label: 'MACD HIST',value: fastOnly ? '—' : (tech ? fmt(tech.lastHistogram) : '—'), color: fastOnly ? 'neutral' : (tech?.lastHistogram != null ? (tech.lastHistogram > 0 ? 'pos' : 'neg') : 'neutral') },
    { label: 'ATR 14',   value: tech ? fmtDollar(tech.lastATR) : '—' },
    { label: 'ANN. VOL', value: tech ? `${tech.annVol.toFixed(1)}%` : '—', color: tech?.annVol != null ? (tech.annVol < 20 ? 'pos' : tech.annVol > 50 ? 'neg' : '') : 'neutral' },
  ]

  const techRow4: CellDef[] = [
    { label: 'RSI(7)',   value: tech ? fmt(tech.lastRsi7, 1) : '—',   color: rsiColor(tech?.lastRsi7 ?? null) },
    { label: 'RSI(14)',  value: tech ? fmt(tech.lastRsi14, 1) : '—',  color: rsiColor(tech?.lastRsi14 ?? null) },
    { label: 'RSI(21)',  value: hideRSI21 ? '—' : (tech ? fmt(tech.lastRsi21, 1) : '—'), color: hideRSI21 ? 'neutral' : rsiColor(tech?.lastRsi21 ?? null) },
    {
      label: 'STOCH %K',
      value: fastOnly ? '—' : (tech ? fmt(tech.lastStochK, 1) : '—'),
      color: fastOnly ? 'neutral' : (tech?.lastStochK != null ? (tech.lastStochK > 80 ? 'neg' : tech.lastStochK < 20 ? 'warn' : '') : 'neutral'),
    },
    {
      label: 'STOCH %D',
      value: fastOnly ? '—' : (tech ? fmt(tech.lastStochD, 1) : '—'),
      color: fastOnly ? 'neutral' : (tech?.lastStochD != null ? (tech.lastStochD > 80 ? 'neg' : tech.lastStochD < 20 ? 'warn' : '') : 'neutral'),
    },
  ]

  const techRow5: CellDef[] = [
    {
      label: 'BOLL %B',
      value: fastOnly ? '—' : (tech ? fmt(tech.lastPctB) : '—'),
      color: fastOnly ? 'neutral' : (tech?.lastPctB != null ? (tech.lastPctB > 1 ? 'neg' : tech.lastPctB < 0 ? 'warn' : tech.lastPctB > 0.4 && tech.lastPctB < 0.6 ? 'pos' : '') : 'neutral'),
    },
    { label: 'BOLL BW%', value: fastOnly ? '—' : (tech ? fmt(tech.lastBandwidth, 1) : '—'), color: fastOnly ? 'neutral' : undefined },
    { label: 'REL VOL',  value: tech ? fmt(tech.lastRelVol) : '—', color: tech?.lastRelVol != null ? (tech.lastRelVol > 1.5 ? 'pos' : tech.lastRelVol < 0.5 ? 'neutral' : '') : 'neutral' },
    {
      label: 'VOL ROC%',
      value: fastOnly ? '—' : (tech ? fmtPct(tech.volROC, 1, true) : '—'),
      color: fastOnly ? 'neutral' : (tech?.volROC != null ? (tech.volROC > 0 ? 'pos' : 'neg') : 'neutral'),
    },
    { label: 'OBV', value: tech ? (tech.lastOBV != null ? tech.lastOBV.toLocaleString() : '—') : '—' },
  ]

  return (
    <div className={styles.container}>
      {section === 'overview' && (
        <>
          {/* Section 1: Fundamentals */}
          <div className={styles.section}>
            <div className={styles.sectionTitle}>Fundamentals</div>
            <div className={styles.grid}>
              <Row cells={fundRow1} />
              <Row cells={fundRow2} />
              <Row cells={fundRow3} />
            </div>
          </div>

          {/* Section 2: Performance */}
          <div className={styles.section}>
            <div className={styles.sectionTitle}>Performance</div>
            <div className={styles.grid}>
              <Row cells={perfRow1} />
              <Row cells={perfRow2} />
            </div>
          </div>
        </>
      )}

      {section === 'technicals' && (
        <div className={styles.section}>
          <div className={styles.sectionTitle}>Technical Indicators (Live)</div>
          <div className={styles.grid}>
            <Row cells={techRow1} />
            <Row cells={techRow2} />
            <Row cells={techRow3} />
            <Row cells={techRow4} />
            <Row cells={techRow5} />
          </div>
        </div>
      )}
    </div>
  )
}
