import { useEffect, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { useStock } from '../../store/StockContext'
import { parseCSV } from '../../lib/csvParser'
import { parseFundamentalsCSV } from '../../lib/fundamentalsParser'
import type { WindowPeriod } from '../../store/types'
import styles from './CsvUpload.module.css'

const PERIOD_PATHS: { period: WindowPeriod; path: string }[] = [
  { period: '3m', path: './data/latest/stocks_3m.csv' },
  { period: '2w', path: './data/latest/stocks_2w.csv' },
  { period: '1m', path: './data/latest/stocks_1m.csv' },
  { period: '6m', path: './data/latest/stocks_6m.csv' },
]

export default function CsvUpload() {
  const { dispatch } = useStock()
  const [autoStatus, setAutoStatus] = useState<'idle' | 'loading' | 'loaded' | 'failed'>('idle')
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  // Auto-load all 4 period CSVs from GitHub Pages on mount
  useEffect(() => {
    setAutoStatus('loading')

    // Load default (3m) first, then others in background
    const loadPeriod = async ({ period, path }: { period: WindowPeriod; path: string }) => {
      const res = await fetch(path)
      if (!res.ok) throw new Error(`${res.status}`)
      const text = await res.text()
      const { data, metadata } = parseCSV(text)
      dispatch({ type: 'SET_WINDOW_DATA', period, data, metadata })
      dispatch({ type: 'SET_DATA_SOURCE', source: 'auto' })
    }

    loadPeriod(PERIOD_PATHS[0])
      .then(() => {
        setAutoStatus('loaded')
        // Load remaining periods in the background
        for (const p of PERIOD_PATHS.slice(1)) {
          loadPeriod(p).catch(() => {}) // silent — other windows may not exist
        }
        // Fetch fundamentals silently
        fetch('./data/latest/fundamentals.csv')
          .then(r => r.ok ? r.text() : Promise.reject())
          .then(text => {
            const data = parseFundamentalsCSV(text)
            dispatch({ type: 'SET_FUNDAMENTALS', data })
          })
          .catch(() => {}) // silent — fundamentals are optional
      })
      .catch(() => setAutoStatus('failed'))
  }, [dispatch])

  function loadFile(file: File) {
    const reader = new FileReader()
    reader.onload = e => {
      const text = e.target?.result as string
      const { data, metadata } = parseCSV(text)
      // Detect period from filename
      const match = file.name.match(/_(2w|1m|3m|6m)/i)
      const period: WindowPeriod = match
        ? (match[1].toLowerCase() as WindowPeriod)
        : '3m'
      dispatch({ type: 'SET_WINDOW_DATA', period, data, metadata })
      dispatch({ type: 'SET_DATA_SOURCE', source: 'upload' })
    }
    reader.readAsText(file)
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file?.name.endsWith('.csv')) loadFile(file)
  }

  if (autoStatus === 'loaded') {
    return (
      <div className={styles.loaded}>
        <span className={styles.dot} />
        Auto-loaded
        <button
          className={styles.overrideBtn}
          onClick={() => fileRef.current?.click()}
          title="Override with manual upload"
        >
          upload CSV
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={e => e.target.files?.[0] && loadFile(e.target.files[0])}
        />
      </div>
    )
  }

  return (
    <div
      className={`${styles.dropzone} ${dragging ? styles.dragging : ''}`}
      onDrop={onDrop}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onClick={() => fileRef.current?.click()}
    >
      <input
        ref={fileRef}
        type="file"
        accept=".csv"
        style={{ display: 'none' }}
        onChange={e => e.target.files?.[0] && loadFile(e.target.files[0])}
      />
      {autoStatus === 'loading' ? (
        <span className={styles.hint}>Loading data...</span>
      ) : (
        <>
          <span className={styles.hint}>Drop CSV or click to upload</span>
          <span className={styles.sub}>stocks_3m.csv</span>
        </>
      )}
    </div>
  )
}
