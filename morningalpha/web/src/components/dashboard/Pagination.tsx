import styles from './Pagination.module.css'

interface Props {
  page: number
  totalPages: number
  onPage: (p: number) => void
  totalRows: number
  pageSize: number
}

export default function Pagination({ page, totalPages, onPage, totalRows, pageSize }: Props) {
  if (totalPages <= 1) return null

  const start = (page - 1) * pageSize + 1
  const end = Math.min(page * pageSize, totalRows)

  return (
    <div className={styles.bar}>
      <span className={styles.info}>
        {start}–{end} of {totalRows}
      </span>
      <div className={styles.btns}>
        <button
          className={styles.btn}
          disabled={page === 1}
          onClick={() => onPage(1)}
        >
          «
        </button>
        <button
          className={styles.btn}
          disabled={page === 1}
          onClick={() => onPage(page - 1)}
        >
          ‹
        </button>
        <span className={styles.pageNum}>{page} / {totalPages}</span>
        <button
          className={styles.btn}
          disabled={page === totalPages}
          onClick={() => onPage(page + 1)}
        >
          ›
        </button>
        <button
          className={styles.btn}
          disabled={page === totalPages}
          onClick={() => onPage(totalPages)}
        >
          »
        </button>
      </div>
    </div>
  )
}
