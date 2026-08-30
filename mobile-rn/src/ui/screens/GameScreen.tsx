/**
 * Main game screen. Owns the Board + AI seam (engine, predictor, mover) and
 * mirrors the backend session flow on-device:
 *
 *   human turn -> tap cell (select) -> Place (confirm) -> board.apply
 *   -> predictor.record -> AI turn runs on a ~350ms timer -> move applied
 *
 * GameState lives in `snap` (the on-device equivalent of the backend
 * snapshot); the live Board instance lives in a ref and is mutated by both
 * sides. `thinking` / `over` refs guard re-entrancy around the AI task.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { Pressable, StyleSheet, Text, View } from 'react-native'
import { Theme, fontSize, radius, spacing } from '../theme'
import { Board3D, axisCross } from '../components/Board3D'
import { MenuSheet } from '../components/MenuSheet'
import { StatusBar } from '../components/StatusBar'
import { EMPTY, P1, P2, type Cell } from '../../game/types'
import { Board } from '../../game/board'
import { createNativeEngine } from '../../ai/engine'
import { OpponentPredictor } from '../../ai/predictor'
import { LookaheadMover } from '../../ai/mover'
import { emptyState, type EvalEngine, type GameConfig, type GameState } from '../../ai/types'

/** Short delay so "AI thinking…" is actually visible before the move lands. */
const AI_DELAY_MS = 350

const ENGINE_UNAVAILABLE_MSG =
  'Model engine not available — build with expo run:android/ios'

export function GameScreen() {
  const [config, setConfig] = useState<GameConfig>({ size: 3, difficulty: 'hard', humanSide: 1 })
  const [menuVisible, setMenuVisible] = useState(true)
  const [snap, setSnap] = useState<GameState>(() => emptyState(3, 1))
  const [pending, setPending] = useState<number | null>(null)
  const [engineError, setEngineError] = useState<string | null>(null)

  const engineRef = useRef<EvalEngine | null>(null)
  const boardRef = useRef<Board | null>(null)
  const predictorRef = useRef<OpponentPredictor | null>(null)
  const moverRef = useRef<LookaheadMover | null>(null)
  const humanSideRef = useRef<Cell>(1)
  const thinkingRef = useRef(false)
  const overRef = useRef(false)
  const movesRef = useRef<number[]>([])
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const configRef = useRef<GameConfig>(config)

  // Create the native engine once; surface load failures gracefully.
  useEffect(() => {
    try {
      engineRef.current = createNativeEngine()
    } catch {
      engineRef.current = null
      setEngineError(ENGINE_UNAVAILABLE_MSG)
    }
  }, [])

  useEffect(
    () => () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    },
    [],
  )

  const runAITurn = useCallback(() => {
    if (!boardRef.current || !moverRef.current || !predictorRef.current) return
    if (overRef.current || thinkingRef.current) return
    const aiSide: Cell = humanSideRef.current === P1 ? P2 : P1
    thinkingRef.current = true
    setSnap((prev) => ({ ...prev, thinking: true }))
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      const board = boardRef.current
      const mover = moverRef.current
      const predictor = predictorRef.current
      if (!board || !mover || !predictor) return
      if (overRef.current) return
      try {
        const move = mover.chooseMove(aiSide)
        board.apply(move, aiSide)
        predictor.record(aiSide, move)
        movesRef.current.push(move)
        const outcome = board.outcome()
        overRef.current = outcome.over
        thinkingRef.current = false
        setSnap((prev) => ({
          ...prev,
          cells: board.cells.slice(),
          currentPlayer: outcome.over ? outcome.winner : humanSideRef.current,
          winner: outcome.winner,
          winningLine: outcome.line,
          over: outcome.over,
          thinking: false,
          movesPlayed: movesRef.current.slice(),
        }))
        setPending(null)
      } catch {
        thinkingRef.current = false
        setSnap((prev) => ({ ...prev, thinking: false }))
      }
    }, AI_DELAY_MS)
  }, [])

  const startGame = useCallback(
    (cfg: GameConfig) => {
      const engine = engineRef.current
      if (!engine) {
        setEngineError(ENGINE_UNAVAILABLE_MSG)
        setMenuVisible(false)
        return
      }
      if (timerRef.current) clearTimeout(timerRef.current)
      const board = new Board(cfg.size)
      const predictor = new OpponentPredictor(board, engine)
      const mover = new LookaheadMover(engine, board, predictor, cfg.difficulty)
      predictor.newGame()
      boardRef.current = board
      predictorRef.current = predictor
      moverRef.current = mover
      humanSideRef.current = cfg.humanSide
      configRef.current = cfg
      overRef.current = false
      thinkingRef.current = false
      movesRef.current = []
      setConfig(cfg)
      setEngineError(null)
      setPending(null)
      setSnap(emptyState(cfg.size, cfg.humanSide))
      setMenuVisible(false)
      if (cfg.humanSide === P2) {
        // the AI opens as X
        runAITurn()
      }
    },
    [runAITurn],
  )

  const clickCell = useCallback(
    (index: number) => {
      const board = boardRef.current
      if (!board) return
      if (thinkingRef.current || overRef.current) return
      if (snap.currentPlayer !== humanSideRef.current) return
      if (board.cells[index] !== EMPTY) return
      if (pending != null && !axisCross(pending, board.n).has(index)) return
      setPending(index)
    },
    [pending, snap.currentPlayer],
  )

  const interactive = useCallback(
    (index: number) => {
      const board = boardRef.current
      if (!board) return false
      if (thinkingRef.current || overRef.current) return false
      if (snap.currentPlayer !== humanSideRef.current) return false
      if (board.cells[index] !== EMPTY) return false
      if (pending != null && !axisCross(pending, board.n).has(index)) return false
      return true
    },
    [pending, snap.currentPlayer],
  )

  const placeMove = useCallback(() => {
    const board = boardRef.current
    const predictor = predictorRef.current
    if (!board || !predictor) return
    if (pending == null) return
    if (thinkingRef.current || overRef.current) return
    if (snap.currentPlayer !== humanSideRef.current) return
    if (board.cells[pending] !== EMPTY) return
    const human = humanSideRef.current
    board.apply(pending, human)
    predictor.record(human, pending)
    movesRef.current.push(pending)
    const outcome = board.outcome()
    setPending(null)
    if (outcome.over) {
      overRef.current = true
      setSnap((prev) => ({
        ...prev,
        cells: board.cells.slice(),
        winner: outcome.winner,
        winningLine: outcome.line,
        over: true,
        thinking: false,
        movesPlayed: movesRef.current.slice(),
      }))
      return
    }
    setSnap((prev) => ({
      ...prev,
      cells: board.cells.slice(),
      currentPlayer: human === P1 ? P2 : P1,
      movesPlayed: movesRef.current.slice(),
    }))
    runAITurn()
  }, [pending, snap.currentPlayer, runAITurn])

  const cancelPending = useCallback(() => setPending(null), [])

  const playAgain = useCallback(() => startGame(configRef.current), [startGame])

  const handleStart = useCallback((cfg: GameConfig) => startGame(cfg), [startGame])

  const openMenu = useCallback(() => setMenuVisible(true), [])

  const isHumanTurn = !snap.over && !snap.thinking && snap.currentPlayer === config.humanSide
  const humanMark = config.humanSide === P1 ? 'X' : 'O'

  return (
    <View style={styles.root}>
      <View style={styles.topBar}>
        <View style={styles.brandRow}>
          <View style={styles.brandDot} />
          <Text style={styles.brandText}>NEON CUBE</Text>
        </View>
        <Pressable onPress={openMenu} style={styles.menuBtn} hitSlop={8}>
          <Text style={styles.menuBtnText}>Menu</Text>
        </Pressable>
      </View>

      <View style={styles.boardArea}>
        <Board3D
          size={snap.size}
          cells={snap.cells}
          onCellClick={clickCell}
          pendingIndex={pending}
          winningLine={snap.winningLine}
          interactive={interactive}
        />

        {isHumanTurn && pending == null && (
          <View style={styles.hint} pointerEvents="none">
            <Text style={styles.hintText}>Tap a cell · drag to rotate · pinch to zoom</Text>
          </View>
        )}

        {engineError != null && (
          <View style={styles.engineError} pointerEvents="none">
            <Text style={styles.engineErrorText}>{engineError}</Text>
          </View>
        )}
      </View>

      <View style={styles.bottom}>
        <StatusBar state={snap} humanSide={config.humanSide} onPlayAgain={playAgain} />

        {pending != null && isHumanTurn && (
          <View style={styles.actionBar}>
            <Pressable onPress={placeMove} style={styles.placeBtn}>
              <Text style={styles.placeBtnText}>Place {humanMark}</Text>
            </Pressable>
            <Pressable onPress={cancelPending} style={styles.cancelBtn}>
              <Text style={styles.cancelBtnText}>Cancel</Text>
            </Pressable>
          </View>
        )}
      </View>

      <MenuSheet visible={menuVisible} onStart={handleStart} />
    </View>
  )
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: Theme.bg,
  },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing(4),
    paddingVertical: spacing(3),
    borderBottomWidth: 1,
    borderBottomColor: Theme.border,
    backgroundColor: 'rgba(2, 6, 23, 0.8)',
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing(2),
  },
  brandDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: Theme.cyan,
    shadowColor: Theme.cyan,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.9,
    shadowRadius: 6,
  },
  brandText: {
    color: Theme.cyan,
    fontSize: fontSize(15),
    fontWeight: '800',
    letterSpacing: 2,
  },
  menuBtn: {
    paddingVertical: spacing(1.5),
    paddingHorizontal: spacing(3),
    borderRadius: radius(2),
    borderWidth: 1,
    borderColor: Theme.border,
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
  },
  menuBtnText: {
    color: Theme.text,
    fontSize: fontSize(13),
    fontWeight: '600',
  },
  boardArea: {
    flex: 1,
    position: 'relative',
    overflow: 'hidden',
    backgroundColor: Theme.bg,
  },
  hint: {
    position: 'absolute',
    bottom: spacing(4),
    alignSelf: 'center',
    paddingVertical: spacing(2),
    paddingHorizontal: spacing(4),
    borderRadius: radius(10),
    borderWidth: 1,
    borderColor: Theme.cyan,
    backgroundColor: 'rgba(2, 6, 23, 0.85)',
  },
  hintText: {
    color: Theme.cyan,
    fontSize: fontSize(12),
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  engineError: {
    position: 'absolute',
    top: spacing(4),
    left: spacing(4),
    right: spacing(4),
    paddingVertical: spacing(3),
    paddingHorizontal: spacing(4),
    borderRadius: radius(2),
    borderWidth: 1,
    borderColor: Theme.danger,
    backgroundColor: 'rgba(2, 6, 23, 0.92)',
  },
  engineErrorText: {
    color: Theme.danger,
    fontSize: fontSize(13),
    fontWeight: '600',
    textAlign: 'center',
  },
  bottom: {
    borderTopWidth: 1,
    borderTopColor: Theme.border,
  },
  actionBar: {
    flexDirection: 'row',
    gap: spacing(3),
    paddingHorizontal: spacing(4),
    paddingVertical: spacing(3),
    backgroundColor: 'rgba(2, 6, 23, 0.85)',
  },
  placeBtn: {
    flex: 1,
    paddingVertical: spacing(3),
    borderRadius: radius(2),
    backgroundColor: Theme.cyan,
    alignItems: 'center',
  },
  placeBtnText: {
    color: Theme.bg,
    fontSize: fontSize(16),
    fontWeight: '800',
    letterSpacing: 1,
  },
  cancelBtn: {
    paddingVertical: spacing(3),
    paddingHorizontal: spacing(5),
    borderRadius: radius(2),
    borderWidth: 1,
    borderColor: Theme.border,
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
    alignItems: 'center',
  },
  cancelBtnText: {
    color: Theme.text,
    fontSize: fontSize(15),
    fontWeight: '600',
  },
})