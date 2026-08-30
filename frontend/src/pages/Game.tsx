import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createGame,
  getGame,
  makeMove,
  gameSocket,
  GameSnapshot,
  GameEvent,
} from '../api'
import { Board3D, Cells, axisCross } from '../components/Board3D'

export function Game({ mobile = false }: { mobile?: boolean }) {
  const [snap, setSnap] = useState<GameSnapshot | null>(null)
  const [started, setStarted] = useState(false)
  const [size, setSize] = useState(3)
  const [difficulty, setDifficulty] = useState('hard')
  const [humanSide, setHumanSide] = useState(1)
  const [pending, setPending] = useState<number | null>(null)
  const explode = mobile ? 0.45 : 0
  const [errorMsg, setErrorMsg] = useState('')
  const wsRef = useRef<WebSocket | null>(null)

  const onEvent = (e: GameEvent) => {
    if (e.game) setSnap(e.game)
  }

  const newGame = async (side?: number, diff?: string, sz?: number) => {
    wsRef.current?.close()
    try {
      const id = await createGame({
        size: sz ?? size,
        mode: 'pve',
        difficulty: diff ?? difficulty,
        x_agent: 'model',
        o_agent: 'model',
        human_side: side ?? humanSide,
      })
      setSnap(await getGame(id))
      setSize(sz ?? size)
      setPending(null)
      setErrorMsg('')
      setStarted(true)
      wsRef.current = gameSocket(id, onEvent)
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Could not start game.')
    }
  }

  // clear any pending selection whenever the board changes (a move landed)
  useEffect(() => {
    setPending(null)
  }, [snap?.current_player, snap?.over, snap?.thinking])

  useEffect(() => () => wsRef.current?.close(), [])

  const bailOut = useCallback(() => {
    setPending(null)
  }, [])

  // Escape bails out of the offered row/column
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') bailOut()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [bailOut])

  const isHumanTurn = () =>
    !!snap && !snap.over && !snap.thinking && snap.current_player === humanSide

  // select a cell to see its row/column + odds; confirm with the Place button.
  // No double-click — rotation drags can't place anything (drag guard in Board3D).
  const clickCell = useCallback(
    async (index: number) => {
      if (!snap || snap.over || snap.thinking) return
      if (snap.current_player !== humanSide) return
      const occupied = snap.cells[index] !== 0
      if (!occupied && pending != null && !axisCross(pending, size).has(index)) return
      setPending(index)
    },
    [snap, pending, humanSide, size],
  )

  const placeMove = useCallback(async () => {
    if (!snap || pending == null) return
    setPending(null)
    const updated = await makeMove(snap.id, pending)
    setSnap(updated)
  }, [snap, pending])

  const cells: Cells = snap ? (snap.cells as Cells) : []
  const humanColor = humanSide === 1 ? 'X' : 'O'

  const setupControls = (label: string) => (
    <div className="game-controls">
      <label>
        you play as
        <select value={humanSide} onChange={(e) => setHumanSide(Number(e.target.value))}>
          <option value={1}>X (first)</option>
          <option value={2}>O (second)</option>
        </select>
      </label>
      <label>
        board size
        <select value={size} onChange={(e) => setSize(Number(e.target.value))}>
          <option value={3}>3×3×3</option>
          <option value={4}>4×4×4</option>
          <option value={5}>5×5×5</option>
        </select>
      </label>
      <label>
        difficulty
        <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
          <option value="easy">Easy</option>
          <option value="medium">Medium</option>
          <option value="hard">Hard</option>
        </select>
      </label>
      <button className="primary" onClick={() => newGame()}>
        {label}
      </button>
    </div>
  )

  return (
    <div className="game">
      <div className="game-side">
        {started && !mobile && (
          <div className="panel">
            <h2>Play vs the AI</h2>
            {errorMsg && <div className="model-error">{errorMsg}</div>}
            {setupControls('New game')}
          </div>
        )}

        {snap?.over && (
          <div className="panel">
            <div className="status-line">
              {snap.winner === 0
                ? 'Draw'
                : snap.winner === humanSide
                  ? 'You win!'
                  : 'AI wins'}
            </div>
          </div>
        )}

        </div>

      <div className="game-board">
        <Board3D
          size={size}
          cells={cells}
          onCellClick={clickCell}
          onEmptyClick={bailOut}
          hoverable={(i) =>
            !!snap &&
            !snap.over &&
            !snap.thinking &&
            snap.cells[i] === 0 &&
            (pending == null || axisCross(pending, size).has(i))
          }
          winningLine={snap?.winning_line as [number, number, number][] | null}
          pendingIndex={pending}
          explode={explode}
        />
        {isHumanTurn() && pending == null && (
          <div className="board-hint">
            {mobile
              ? 'Tap a cell to select it · drag to rotate'
              : 'Select a cell · then press Place'}
          </div>
        )}
        {isHumanTurn() && pending != null && (
          <div className="board-actions">
            {snap && snap.cells[pending] === 0 && (
              <button className="primary" onClick={placeMove}>
                Place {humanColor}
              </button>
            )}
            <button className="ghost" onClick={bailOut}>
              Cancel
            </button>
          </div>
        )}
        {snap?.thinking && (
          <div className="thinking-badge">
            <span className="thinking-ring" />
            <span className="thinking-dots">
              AI thinking<span>.</span><span>.</span><span>.</span>
            </span>
          </div>
        )}
        {!started && (
          <div className="setup-overlay">
            <div className="panel">
              <h2>Play vs the AI</h2>
              {errorMsg && <div className="model-error">{errorMsg}</div>}
              {setupControls(snap ? 'New game' : 'Start game')}
            </div>
          </div>
        )}
        {started && mobile && (
          <button className="menu-btn" onClick={() => setStarted(false)}>
            Menu
          </button>
        )}
      </div>
    </div>
  )
}