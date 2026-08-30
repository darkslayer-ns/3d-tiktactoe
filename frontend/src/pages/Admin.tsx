import { useEffect, useState } from 'react'
import { listGames, AdminGame, AiDecision } from '../api'

function cellLabel(index: number, size: number): string {
  const z = Math.floor(index / (size * size))
  const r = index % (size * size)
  const y = Math.floor(r / size)
  const x = r % size
  return `${x},${y},${z}`
}

function Decision({ d, size }: { d: AiDecision | null; size: number }) {
  if (!d) return <div className="muted">no AI decision yet</div>
  const label = d.player === 1 ? 'X' : 'O'
  const pct = Math.round((d.value ?? 0) * 100)
  return (
    <div className="decision">
      <div className="decision-head">
        <span className={`player-badge p${d.player}`}>{label}</span>
        <span className="decision-move">
          → cell <b>{d.chosen}</b> <span className="coord">({d.coord.join(',')})</span>
        </span>
        <span className={`kind-badge ${d.kind}`}>{d.kind}</span>
        <span className="depth-tag">depth {d.depth}</span>
      </div>
      <div className="value-row">
        <span className="value-label">{label} win prob</span>
        <div className="value-bar">
          <span className="value-fill" style={{ width: `${pct}%` }} />
        </div>
        <span className="value-pct">{pct}%</span>
      </div>
      {d.scored.length > 0 && (
        <div className="candidate-list">
          {d.scored.map(([idx, v]) => (
            <div key={idx} className={`candidate ${idx === d.chosen ? 'chosen' : ''}`}>
              <span className="candidate-coord">{cellLabel(idx, size)}</span>
              <div className="candidate-bar">
                <span className="candidate-fill" style={{ width: `${Math.max(3, v * 100)}%` }} />
              </div>
              <span className="candidate-val">{v.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}
      {d.line && d.line.length > 1 && (
        <div className="predicted-line">
          <span className="predicted-label">AI predicts</span>
          {d.line.map((s, i) => (
            <span key={i} className={`line-chip p${s.player}`}>
              {i > 0 && <span className="line-arrow">→</span>}
              {s.player === 1 ? 'X' : 'O'}
              <span className="coord">({s.coord.join(',')})</span>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export function Admin() {
  const [games, setGames] = useState<AdminGame[]>([])

  useEffect(() => {
    let stop = false
    const tick = async () => {
      if (stop) return
      try {
        const g = await listGames()
        if (!stop) setGames(g)
      } catch {
        /* backend briefly down */
      }
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => {
      stop = true
      clearInterval(id)
    }
  }, [])

  return (
    <div className="admin">
      <div className="admin-games">
        <h2>Active games ({games.length})</h2>
        {games.length === 0 && <div className="pred-empty">no active games</div>}
        {games.map((g) => {
          const aiDecision = g.x_decision || g.o_decision
          return (
            <div key={g.id} className="panel admin-game">
              <div className="admin-game-head">
                <span className="admin-game-id">{g.id}</span>
                <span className="admin-game-meta">
                  {g.size}×{g.size}×{g.size} · {g.difficulty} ·{' '}
                  {g.over
                    ? g.winner === 0
                      ? 'draw'
                      : `winner ${g.winner}`
                    : g.thinking
                      ? 'AI thinking…'
                      : `turn ${g.current_player}`}{' '}
                  · {g.move_count} moves
                </span>
              </div>
              <Decision d={aiDecision} size={g.size} />
              {g.move_count > 0 && (
                <div className="admin-moves">
                  <span className="moves-label">moves</span>
                  {g.moves.map((m, i) => (
                    <span key={i} className={`move-chip p${m.player}`}>
                      {m.player === 1 ? 'X' : 'O'} {m.index}
                      <span className="coord">({m.coord.join(',')})</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}