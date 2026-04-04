import { useState, useRef, useEffect } from 'react'
import type { Portfolio } from '../../lib/portfolioStorage'
import styles from './PortfolioTabs.module.css'

interface Props {
  portfolios: Portfolio[]
  activeId: string | null
  onSwitch: (id: string) => void
  onCreate: (name: string) => void
  onRename: (id: string, name: string) => void
  onDelete: (id: string) => void
}

export default function PortfolioTabs({ portfolios, activeId, onSwitch, onCreate, onRename, onDelete }: Props) {
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const editRef = useRef<HTMLInputElement>(null)
  const newRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editingId) editRef.current?.select()
  }, [editingId])

  useEffect(() => {
    if (creating) newRef.current?.focus()
  }, [creating])

  function startEdit(p: Portfolio) {
    setEditingId(p.id)
    setEditValue(p.name)
  }

  function commitEdit() {
    if (editingId && editValue.trim()) {
      onRename(editingId, editValue.trim())
    }
    setEditingId(null)
  }

  function commitCreate() {
    const name = newName.trim()
    if (name) onCreate(name)
    setCreating(false)
    setNewName('')
  }

  function cancelCreate() {
    setCreating(false)
    setNewName('')
  }

  return (
    <div className={styles.bar}>
      <div className={styles.tabs}>
        {portfolios.map(p => {
          const isActive = p.id === activeId
          return (
            <div
              key={p.id}
              className={`${styles.tab} ${isActive ? styles.tabActive : ''}`}
              onClick={() => { if (!isActive) onSwitch(p.id) }}
            >
              {editingId === p.id ? (
                <input
                  ref={editRef}
                  className={styles.editInput}
                  value={editValue}
                  onChange={e => setEditValue(e.target.value)}
                  onBlur={commitEdit}
                  onKeyDown={e => {
                    if (e.key === 'Enter') { e.preventDefault(); commitEdit() }
                    if (e.key === 'Escape') setEditingId(null)
                  }}
                  onClick={e => e.stopPropagation()}
                />
              ) : (
                <span
                  className={styles.tabName}
                  onDoubleClick={e => { e.stopPropagation(); startEdit(p) }}
                  title="Double-click to rename"
                >
                  {p.name}
                </span>
              )}
              {portfolios.length > 1 && (
                <button
                  className={styles.deleteTab}
                  onClick={e => { e.stopPropagation(); onDelete(p.id) }}
                  title={`Delete "${p.name}"`}
                  aria-label={`Delete ${p.name}`}
                >
                  ×
                </button>
              )}
            </div>
          )
        })}

        {creating && (
          <div className={`${styles.tab} ${styles.tabActive}`}>
            <input
              ref={newRef}
              className={styles.editInput}
              placeholder="List name…"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              onBlur={commitCreate}
              onKeyDown={e => {
                if (e.key === 'Enter') { e.preventDefault(); commitCreate() }
                if (e.key === 'Escape') cancelCreate()
              }}
            />
          </div>
        )}
      </div>

      {!creating && (
        <button className={styles.newBtn} onClick={() => setCreating(true)} title="New list">
          + New
        </button>
      )}
    </div>
  )
}
