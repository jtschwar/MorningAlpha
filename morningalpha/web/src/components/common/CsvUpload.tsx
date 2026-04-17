import { useRef } from 'react'
import type { DragEvent } from 'react'
import { useStock } from '../../store/StockContext'
import { parseCSV } from '../../lib/csvParser'
import type { WindowPeriod } from '../../store/types'
import styles from './CsvUpload.module.css'

export default function CsvUpload() {
  const { state, dispatch } = useStock()
  const { autoLoadStatus, generatedAt } = state
  const fileRef = useRef<HTMLInputElement>(null)

  function loadFile(file: File) {
    const reader = new FileReader()
    reader.onload = e => {
      const text = e.target?.result as string
      const { data, metadata } = parseCSV(text)
      const match = file.name.match(/_(2w|1m|3m|6m)/i)
      const period: WindowPeriod = match ? (match[1].toLowerCase() as WindowPeriod) : '3m'
      dispatch({ type: 'SET_WINDOW_DATA', period, data, metadata })
      dispatch({ type: 'SET_DATA_SOURCE', source: 'upload' })
      dispatch({ type: 'SET_AUTO_LOAD_STATUS', status: 'loaded', generatedAt: null })
    }
    reader.readAsText(file)
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file?.name.endsWith('.csv')) loadFile(file)
  }

  if (autoLoadStatus === 'loaded') {
    return (
      <div className={styles.loaded}>
        <span className={styles.dot} />
        {generatedAt ? (
          <span className={styles.timestamp}>
            <span className={styles.timestampLabel}>Last Updated</span>
            <span>{generatedAt.date} {generatedAt.time}</span>
          </span>
        ) : 'Auto-loaded'}
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
      className={styles.dropzone}
      onDrop={onDrop}
      onDragOver={e => { e.preventDefault() }}
      onClick={() => fileRef.current?.click()}
    >
      <input
        ref={fileRef}
        type="file"
        accept=".csv"
        style={{ display: 'none' }}
        onChange={e => e.target.files?.[0] && loadFile(e.target.files[0])}
      />
      {autoLoadStatus === 'loading' ? (
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
