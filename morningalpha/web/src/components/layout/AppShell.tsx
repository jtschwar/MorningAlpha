import { useState } from 'react'
import type { ReactNode } from 'react'
import TopBar from './TopBar'
import Sidebar from './Sidebar'
import styles from './AppShell.module.css'

interface Props {
  children: ReactNode
  showSidebar?: boolean
}

export default function AppShell({ children, showSidebar = true }: Props) {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div
      className={styles.shell}
      data-sidebar={showSidebar && sidebarOpen ? 'open' : 'closed'}
    >
      <TopBar
        showHamburger={showSidebar}
        onToggleSidebar={() => setSidebarOpen(o => !o)}
        sidebarOpen={sidebarOpen}
      />
      {showSidebar && (
        <aside className={styles.sidebar} aria-hidden={!sidebarOpen}>
          <Sidebar />
        </aside>
      )}
      <main className={styles.main}>{children}</main>
    </div>
  )
}
