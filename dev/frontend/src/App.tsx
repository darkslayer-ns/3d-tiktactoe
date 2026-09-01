import { useEffect, useState } from 'react'
import { Game } from './pages/Game'
import { Admin } from './pages/Admin'

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(
    () =>
      typeof window !== 'undefined' &&
      window.matchMedia('(max-width: 820px), (pointer: coarse)').matches,
  )
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 820px), (pointer: coarse)')
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches)
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [])
  return isMobile
}

export default function App() {
  const [tab, setTab] = useState<'play' | 'admin'>('play')
  const isMobile = useIsMobile()

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-dot" />
          NEON CUBE <span className="brand-sub">3D Tic-Tac-Toe</span>
        </div>
        <nav className="tabs">
          <button className={tab === 'play' ? 'tab active' : 'tab'} onClick={() => setTab('play')}>
            Play
          </button>
          <button className={tab === 'admin' ? 'tab active' : 'tab'} onClick={() => setTab('admin')}>
            Admin
          </button>
        </nav>
      </header>
      <main className="content">{tab === 'play' ? <Game mobile={isMobile} /> : <Admin />}</main>
      <footer className="foot">
        <span className="foot-note">
          {isMobile ? 'drag to rotate · pinch to zoom' : 'drag to rotate · scroll to zoom'}
        </span>
      </footer>
    </div>
  )
}