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
import { WelcomeOverlay } from '../components/WelcomeOverlay'
import { GameOverOverlay } from '../components/GameOverOverlay'
import { EMPTY, P1, P2, type Cell } from '../../game/types'
import { Board } from '../../game/board'
import { createNativeEngine } from '../../ai/engine'
import { OpponentPredictor } from '../../ai/predictor'
import { LookaheadMover } from '../../ai/mover'
import { applyResult, type Affinity } from '../../ai/opponentMemory'
import { loadAffinity, saveAffinity, getWelcomed, setWelcomed, loadProfile, saveProfile, loadStats, saveStats, loadPerception, savePerception } from '../../ai/opponentStorage'
import { analyzeMove, PerceptionProfile, PlayerProfile } from '../../ai/profile'
import { adaptiveLevel, emptyStats, recordResult, type GameStats } from '../../ai/stats'
import { emptyState, type EvalEngine, type GameConfig, type GameState } from '../../ai/types'
import { IS_INTERNAL_BUILD } from '../../dev/internalBuild'
import { ModelKnowledgePanel } from '../../dev/ModelKnowledgePanel'

/** Minimum cube-flash duration so a fast AI move still visibly "thinks". */
const MIN_FLASH_MS = 180

const ENGINE_UNAVAILABLE_MSG =
  'Model engine not available — build with expo run:android/ios'

export function GameScreen() {
  const [config, setConfig] = useState<GameConfig>({ size: 3, difficulty: 'hard', humanSide: 1 })
  const [menuVisible, setMenuVisible] = useState(true)
  const [snap, setSnap] = useState<GameState>(() => emptyState(3, 1))
  const [pending, setPending] = useState<number | null>(null)
  const [engineError, setEngineError] = useState<string | null>(null)
  const [welcomeVisible, setWelcomeVisible] = useState(false)
  const [welcomeMode, setWelcomeMode] = useState<'first' | 'howto'>('first')
  const [roundKey, setRoundKey] = useState(0)
  const [resultVisible, setResultVisible] = useState(false)
  const [knowledgeVisible, setKnowledgeVisible] = useState(false)

  const engineRef = useRef<EvalEngine | null>(null)
  const boardRef = useRef<Board | null>(null)
  const predictorRef = useRef<OpponentPredictor | null>(null)
  const moverRef = useRef<LookaheadMover | null>(null)
  const affinityRef = useRef<Affinity | null>(null)
  const affinityLoadRef = useRef<Promise<Affinity> | null>(null)
  const profileRef = useRef<PlayerProfile | null>(null)
  const perceptionRef = useRef<PerceptionProfile | null>(null)
  const statsRef = useRef<GameStats>(emptyStats())
  const adaptiveRef = useRef(0)
  const humanSideRef = useRef<Cell>(1)
  const thinkingRef = useRef(false)
  const overRef = useRef(false)
  const demoRef = useRef(false)
  const turnRef = useRef<Cell>(P1)
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

  // Load the persistent opponent memory once; games started before it resolves
  // fall back to a fresh map (the load is fast, so this is rare).
  useEffect(() => {
    affinityLoadRef.current = loadAffinity()
    void affinityLoadRef.current.then((aff) => {
      affinityRef.current = aff
    })
  }, [])

  // Load the persistent player-style profile (attacker/defender) once.
  useEffect(() => {
    void loadProfile().then((p) => {
      if (p) profileRef.current = new PlayerProfile(0.95, p)
      else profileRef.current = new PlayerProfile()
    })
  }, [])

  // Load the persistent 3D-perception profile + game stats once.
  useEffect(() => {
    void loadPerception().then((p) => {
      perceptionRef.current = new PerceptionProfile(0.95, p ?? undefined)
    })
    void loadStats().then((s) => {
      if (s) {
        statsRef.current = s
        adaptiveRef.current = adaptiveLevel(s)
      }
    })
  }, [])

  useEffect(
    () => () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    },
    [],
  )

  // Let the winning line flash for ~1.6s, THEN reveal the result overlay.
  useEffect(() => {
    if (snap.over && !snap.demo) {
      const id = setTimeout(() => setResultVisible(true), 1600)
      return () => clearTimeout(id)
    }
    setResultVisible(false)
  }, [snap.over, snap.demo])

  const persistAffinity = useCallback((aff: Affinity) => {
    void saveAffinity(aff).catch(() => {})
  }, [])

  // Reward the winner's cells / penalize the loser's, then save the memory so
  // the AI keeps learning your winning moves across restarts. Also record the
  // game outcome into the stats store to drive adaptive difficulty.
  const endGame = useCallback(
    (winner: Cell) => {
      const s = statsRef.current
      recordResult(s, winner, humanSideRef.current)
      adaptiveRef.current = adaptiveLevel(s)
      void saveStats(s)

      const predictor = predictorRef.current
      const aff = affinityRef.current
      if (predictor && aff) {
        const loser: Cell = winner === P1 ? P2 : P1
        applyResult(aff, winner, loser)
        persistAffinity(aff)
      }
    },
    [persistAffinity],
  )

const runAITurn = useCallback(async () => {
    if (!boardRef.current || !moverRef.current || !predictorRef.current) return
    if (overRef.current || thinkingRef.current) return
    const aiSide: Cell = humanSideRef.current === P1 ? P2 : P1
    thinkingRef.current = true
    // Flash starts immediately (the cube breathes) — the human's mark already
    // rendered, and the inference runs in the background below.
    setSnap((prev) => (prev.thinking ? prev : { ...prev, thinking: true }))
    if (timerRef.current) clearTimeout(timerRef.current)
    const startedAt = Date.now()
    const board = boardRef.current
    const mover = moverRef.current
    const predictor = predictorRef.current
    if (!board || !mover || !predictor) return
    if (overRef.current) return
    try {
      const move = await mover.chooseMove(aiSide)
      // Keep the flash visible for at least MIN_FLASH_MS so the player always
      // sees the move land AFTER the cube breathes (render → flash → resolve).
      const elapsed = Date.now() - startedAt
      if (elapsed < MIN_FLASH_MS) {
        await new Promise<void>((resolve) => setTimeout(resolve, MIN_FLASH_MS - elapsed))
      }
      thinkingRef.current = false
      if (overRef.current) return
      board.apply(move, aiSide)
        predictor.record(aiSide, move)
        movesRef.current.push(move)
        const outcome = board.outcome()
        overRef.current = outcome.over
        thinkingRef.current = false
        if (outcome.over) endGame(outcome.winner)
        setSnap((prev) => ({
          ...prev,
          cells: board.cells.slice(),
          currentPlayer: outcome.over ? outcome.winner : humanSideRef.current,
          winner: outcome.winner,
          winningLine: outcome.line,
          over: outcome.over,
          thinking: false,
          movesPlayed: movesRef.current.slice(),
          lastAiMove: move,
          hintIndex: null,
        }))
        setPending(null)
      } catch {
        thinkingRef.current = false
        setSnap((prev) => ({ ...prev, thinking: false }))
      }
  }, [endGame])

  // AI-vs-AI demo: plays the whole game by itself so the core loop can be
  // recorded/shown. Uses a throwaway predictor (empty memory) so demo games
  // never pollute the persistent opponent memory, and no win/loss reward.
  const runDemoTurn = useCallback(() => {
    const board = boardRef.current
    const mover = moverRef.current
    const predictor = predictorRef.current
    if (!board || !mover || !predictor) return
    if (overRef.current || !demoRef.current) return
    const side = turnRef.current
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(async () => {
      try {
        const move = await mover.chooseMove(side)
        if (overRef.current || !demoRef.current) return
        board.apply(move, side)
        predictor.record(side, move)
        movesRef.current.push(move)
        const outcome = board.outcome()
        if (outcome.over) {
          overRef.current = true
          setSnap((prev) => ({
            ...prev,
            cells: board.cells.slice(),
            winner: outcome.winner,
            winningLine: outcome.line,
            over: true,
            thinking: false,
            demo: true,
            movesPlayed: movesRef.current.slice(),
            lastAiMove: move,
          }))
          setPending(null)
          return
        }
        turnRef.current = side === P1 ? P2 : P1
        setSnap((prev) => ({
          ...prev,
          cells: board.cells.slice(),
          currentPlayer: turnRef.current,
          thinking: false,
          demo: true,
          movesPlayed: movesRef.current.slice(),
          lastAiMove: move,
        }))
        runDemoTurn()
      } catch {
        // engine hiccup — stop the demo rather than spin forever
        demoRef.current = false
      }
    }, 700)
  }, [])

  const startDemo = useCallback(
    (cfg: GameConfig) => {
      const engine = engineRef.current
      if (!engine) {
        setEngineError(ENGINE_UNAVAILABLE_MSG)
        setMenuVisible(false)
        return
      }
      if (timerRef.current) clearTimeout(timerRef.current)
      const board = new Board(cfg.size)
      // Fresh, throwaway memory — demos don't teach the AI.
      const predictor = new OpponentPredictor(board, engine)
      const mover = new LookaheadMover(engine, board, predictor, cfg.difficulty)
      boardRef.current = board
      predictorRef.current = predictor
      moverRef.current = mover
      humanSideRef.current = cfg.humanSide
      configRef.current = cfg
      overRef.current = false
      thinkingRef.current = false
      demoRef.current = true
      turnRef.current = P1
      movesRef.current = []
      setConfig(cfg)
      setEngineError(null)
      setPending(null)
      setSnap({ ...emptyState(cfg.size, cfg.humanSide), demo: true })
      setMenuVisible(false)
      setRoundKey((k) => k + 1)
      runDemoTurn()
    },
    [runDemoTurn],
  )

  const startGame = useCallback(
    async (cfg: GameConfig) => {
      const engine = engineRef.current
      if (!engine) {
        setEngineError(ENGINE_UNAVAILABLE_MSG)
        setMenuVisible(false)
        return
      }
      if (timerRef.current) clearTimeout(timerRef.current)
      if (!affinityRef.current) {
        affinityRef.current = await (affinityLoadRef.current ?? loadAffinity())
      }
      const board = new Board(cfg.size)
      const predictor = new OpponentPredictor(board, engine, 0.9, affinityRef.current)
      const mover = new LookaheadMover(engine, board, predictor, cfg.difficulty)
      mover.setAggression(profileRef.current?.aggression() ?? 0)
      // Hard is intentionally unbeatable; only easy/medium adapt to your results.
      if (cfg.difficulty !== 'hard') mover.setAdaptive(adaptiveRef.current)
      predictor.newGame()
      boardRef.current = board
      predictorRef.current = predictor
      moverRef.current = mover
      humanSideRef.current = cfg.humanSide
      configRef.current = cfg
      overRef.current = false
      thinkingRef.current = false
      demoRef.current = false
      movesRef.current = []
      setConfig(cfg)
      setEngineError(null)
      setPending(null)
      setSnap(emptyState(cfg.size, cfg.humanSide))
      setMenuVisible(false)
      setRoundKey((k) => k + 1)
      if (cfg.humanSide === P2) {
        // the AI opens as X
        runAITurn()
      }
    },
    [runAITurn],
  )

  // First launch: welcome overlay with an AI-vs-AI demo playing behind it.
  useEffect(() => {
    let cancelled = false
    void getWelcomed().then((welcomed) => {
      if (cancelled) return
      if (!welcomed && engineRef.current) {
        setWelcomeVisible(true)
        setMenuVisible(false)
        startDemo({ size: 3, difficulty: 'medium', humanSide: 1 })
      }
    })
    return () => {
      cancelled = true
    }
  }, [startDemo])

  const clickCell = useCallback(
    (index: number) => {
      const board = boardRef.current
      if (!board) return
      if (demoRef.current || thinkingRef.current || overRef.current) return
      if (snap.currentPlayer !== humanSideRef.current) return
      if (board.cells[index] !== EMPTY) return
      if (pending != null && !axisCross(pending, board.n).has(index)) return
      setPending(index)
    },
    [pending, snap.currentPlayer],
  )

  const placeMove = useCallback(() => {
    const board = boardRef.current
    const predictor = predictorRef.current
    if (!board || !predictor) return
    if (pending == null) return
    if (demoRef.current || thinkingRef.current || overRef.current) return
    if (snap.currentPlayer !== humanSideRef.current) return
    if (board.cells[pending] !== EMPTY) return
    const human = humanSideRef.current
    const { style, axis } = analyzeMove(board, human, pending)
    board.apply(pending, human)
    predictor.record(human, pending)
    const prof = profileRef.current
    if (prof) {
      prof.record(style)
      moverRef.current?.setAggression(prof.aggression())
      void saveProfile(prof.toJSON())
    }
    const perc = perceptionRef.current
    if (perc) {
      perc.record(axis)
      void savePerception(perc.toJSON())
    }
    movesRef.current.push(pending)
    const outcome = board.outcome()
    setPending(null)
    if (outcome.over) {
      overRef.current = true
      endGame(outcome.winner)
      setSnap((prev) => ({
        ...prev,
        cells: board.cells.slice(),
        winner: outcome.winner,
        winningLine: outcome.line,
        over: true,
        thinking: false,
        movesPlayed: movesRef.current.slice(),
        lastAiMove: null,
        hintIndex: null,
      }))
      return
    }
    setSnap((prev) => ({
      ...prev,
      cells: board.cells.slice(),
      currentPlayer: human === P1 ? P2 : P1,
      movesPlayed: movesRef.current.slice(),
      lastAiMove: null,
      hintIndex: null,
    }))
    runAITurn()
  }, [pending, snap.currentPlayer, runAITurn, endGame])

  const cancelPending = useCallback(() => setPending(null), [])

  // Take back the last human move (and the AI reply that followed it), so the
  // player can re-think. Works from mid-game and from the finished screen.
  const undoMove = useCallback(() => {
    const board = boardRef.current
    if (!board) return
    if (demoRef.current || thinkingRef.current) return
    const hist = movesRef.current
    if (hist.length === 0) return
    const human = humanSideRef.current
    let lastHuman = -1
    for (let i = hist.length - 1; i >= 0; i--) {
      // plies alternate strictly: ply 0 is P1, ply 1 is P2, …
      const owner: Cell = i % 2 === 0 ? P1 : P2
      if (owner === human) {
        lastHuman = i
        break
      }
    }
    if (lastHuman < 0) return
    for (const m of hist.slice(lastHuman)) board.cells[m] = EMPTY
    movesRef.current = hist.slice(0, lastHuman)
    if (timerRef.current) clearTimeout(timerRef.current)
    overRef.current = false
    thinkingRef.current = false
    setPending(null)
    setSnap((prev) => ({
      ...prev,
      cells: board.cells.slice(),
      currentPlayer: human,
      winner: EMPTY,
      winningLine: null,
      over: false,
      thinking: false,
      movesPlayed: movesRef.current.slice(),
      lastAiMove: null,
      hintIndex: null,
      demo: false,
    }))
  }, [])

  // Recommend the model's best move for the human and pre-select it.
  const showHint = useCallback(async () => {
    const mover = moverRef.current
    if (!mover) return
    if (demoRef.current || thinkingRef.current || overRef.current) return
    const human = humanSideRef.current
    try {
      const hint = await mover.getHint(human)
      if (hint < 0) return
      if (demoRef.current || overRef.current) return
      setPending(hint)
      setSnap((prev) => ({ ...prev, hintIndex: hint }))
    } catch {
      // ignore
    }
  }, [])

  const playAgain = useCallback(() => startGame(configRef.current), [startGame])

  const handleStart = useCallback((cfg: GameConfig) => startGame(cfg), [startGame])

  const openMenu = useCallback(() => setMenuVisible(true), [])

// "How to play" from the menu: run a live AI-vs-AI demo behind the guide
// overlay so the animation literally shows how a game (and a win) works.
const showHowTo = useCallback(
  (cfg: GameConfig) => {
    setWelcomeMode('howto')
    setWelcomeVisible(true)
    startDemo(cfg)
  },
  [startDemo],
)

  const dismissWelcome = useCallback(() => {
    setWelcomeVisible(false)
    void setWelcomed()
    setMenuVisible(true)
  }, [])

  const isHumanTurn =
    !snap.demo && !snap.over && !snap.thinking && snap.currentPlayer === config.humanSide
  const humanMark = config.humanSide === P1 ? 'X' : 'O'

  return (
    <View style={styles.root}>
      <View style={styles.boardArea}>
        <Board3D
          size={snap.size}
          cells={snap.cells}
          onCellClick={clickCell}
          pendingIndex={pending}
          winningLine={snap.winningLine}
          lastAiMove={snap.lastAiMove}
          hintIndex={snap.hintIndex}
          thinking={snap.thinking}
          startKey={roundKey}
        />

        {engineError != null && (
          <View style={styles.engineError} pointerEvents="none">
            <Text style={styles.engineErrorText}>{engineError}</Text>
          </View>
        )}

        {IS_INTERNAL_BUILD && (
          <Pressable
            onPress={() => setKnowledgeVisible(true)}
            style={styles.debugBtn}
            hitSlop={8}
            testID="debug-model-knowledge"
          >
            <Text style={styles.debugBtnText}>AI</Text>
          </Pressable>
        )}
      </View>

      {!snap.demo && (
        <View style={styles.bottom}>
          <StatusBar
            state={snap}
            humanSide={config.humanSide}
            onPlayAgain={playAgain}
            onNewGame={openMenu}
            onHint={showHint}
            onUndo={undoMove}
          />
        </View>
      )}

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

      {snap.over && !snap.demo && resultVisible && (
        <GameOverOverlay
          winner={snap.winner}
          humanSide={config.humanSide}
          onPlayAgain={playAgain}
          onMenu={openMenu}
        />
      )}

      <MenuSheet visible={menuVisible} onStart={handleStart} onHowTo={showHowTo} />

      {welcomeVisible && (
        <WelcomeOverlay
          onStart={dismissWelcome}
          buttonLabel={welcomeMode === 'howto' ? 'Got it' : 'Start playing'}
        />
      )}

      {IS_INTERNAL_BUILD && (
        <ModelKnowledgePanel
          visible={knowledgeVisible}
          onClose={() => setKnowledgeVisible(false)}
          engine={engineRef.current}
          board={boardRef.current}
          humanSide={config.humanSide}
          difficulty={config.difficulty}
          predictor={predictorRef.current}
          mover={moverRef.current}
          profile={profileRef.current}
          perception={perceptionRef.current}
          stats={statsRef.current}
          adaptive={adaptiveRef.current}
        />
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: Theme.bg,
  },
  boardArea: {
    flex: 1,
    position: 'relative',
    overflow: 'hidden',
    backgroundColor: Theme.bg,
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
  debugBtn: {
    position: 'absolute',
    top: spacing(4),
    right: spacing(4),
    width: 34,
    height: 34,
    borderRadius: 17,
    borderWidth: 1.5,
    borderColor: Theme.cyan,
    backgroundColor: 'rgba(2, 6, 23, 0.85)',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: Theme.cyan,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 8,
  },
  debugBtnText: {
    color: Theme.cyan,
    fontSize: fontSize(11),
    fontWeight: '800',
    letterSpacing: 1,
  },
  bottom: {
    // StatusBar draws its own top border.
  },
  actionBar: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: spacing(15),
    flexDirection: 'row',
    gap: spacing(3),
    paddingHorizontal: spacing(4),
    paddingVertical: spacing(3),
    backgroundColor: 'rgba(2, 6, 23, 0.85)',
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: Theme.border,
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