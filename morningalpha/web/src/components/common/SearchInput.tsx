import { useState, useEffect } from 'react'
import { useStock } from '../../store/StockContext'
import { useDebounce } from '../../hooks/useDebounce'
import styles from './SearchInput.module.css'

export default function SearchInput() {
  const { dispatch } = useStock()
  const [value, setValue] = useState('')
  const debounced = useDebounce(value, 300)

  useEffect(() => {
    dispatch({ type: 'SET_FILTER', key: 'search', value: debounced })
  }, [debounced, dispatch])

  return (
    <div className={styles.wrapper}>
      <span className={styles.icon}>⌕</span>
      <input
        className={styles.input}
        type="text"
        placeholder="Search ticker or name..."
        value={value}
        onChange={e => setValue(e.target.value)}
        aria-label="Search stocks"
      />
      {value && (
        <button className={styles.clear} onClick={() => setValue('')} aria-label="Clear search">
          ✕
        </button>
      )}
    </div>
  )
}
