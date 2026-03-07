import { useStock } from '../../store/StockContext'
import styles from './EmptyState.module.css'

export default function EmptyState() {
  const { dispatch } = useStock()
  return (
    <div className={styles.empty}>
      <div className={styles.icon}>∅</div>
      <div className={styles.msg}>No stocks match your filters</div>
      <button
        className={styles.btn}
        onClick={() => dispatch({ type: 'RESET_FILTERS' })}
      >
        Clear Filters
      </button>
    </div>
  )
}
