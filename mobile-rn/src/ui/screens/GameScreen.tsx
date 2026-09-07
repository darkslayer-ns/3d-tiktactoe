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
import { AppState, Pressable, StyleSheet, Text, View, type AppStateStatus } from 'react-native'
import { useSafeAreaInsets } from 'react-native-safe-area-context'
import { Theme, fontSize, radius, spacing } from '../theme'
import { Board3D, axisCross } from '../components/Board3D'
import { MenuSheet } from '../components/MenuSheet'
import { StatusBar } from '../components/StatusBar'
import { WelcomeOverlay } from '../components/WelcomeOverlay'
import { GameOverOverlay } from '../components/GameOverOverlay'
import { EMPTY, P1, P2, type Cell } from '../../game/types'
import { Board } from '../../game/board'
import { isAvailable, aiApplyMove, aiChooseMove, aiEndGame, aiHint, aiSetBoard, aiStart, aiState, type NativeAIConfig, type NativeAIState } from '../../native/TfmEngine'
import { type Affinity } from '../../ai/opponentMemory'
import { loadAffinity, saveAffinity, getWelcomed, setWelcomed, loadProfile, saveProfile, loadStats, saveStats, loadPerception, savePerception } from '../../ai/opponentStorage'
import type { ProfileCounts } from '../../ai/profile'
import { emptyStats, type GameStats } from '../../ai/stats'
import { emptyState, type GameConfig, type GameState } from '../../ai/types'
import { IS_INTERNAL_BUILD } from '../../dev/internalBuild'
import { ModelKnowledgePanel } from '../../dev/ModelKnowledgePanel'
import { playSfx, hapticSelection } from '../../audio/SoundManager'

/** Minimum cube-flash duration so a fast AI move still visibly "thinks". */
const MIN_FLASH_MS = 180

const ENGINE_UNAVAILABLE_MSG =
  'Model engine not available — build with expo run:android/ios'

function flattenAffinity(aff: Affinity | null): number[] {
  if (!aff) return []
  const out: number[] = []
  for (const [side, row] of aff) for (const [cell, weight] of row) out.push(side, cell, weight)
  return out
}

function nativeConfig(
  cfg: GameConfig,
  affinity: Affinity | null,
  profile: ProfileCounts,
  perception: { axis: number; face: number; space: number },
  stats: GameStats,
  adaptive: number,
): NativeAIConfig {
  return {
    n: cfg.size,
    humanSide: cfg.humanSide,
    difficulty: cfg.difficulty,
    aggression: profile.attack + profile.defend < 0.5 ? 0 : (profile.attack - profile.defend) / (profile.attack + profile.defend),
    adaptive,
    affinity: flattenAffinity(affinity),
    profile: [profile.attack, profile.defend, profile.neutral],
    perception: [perception.axis, perception.face, perception.space],
    stats: [stats.wins, stats.losses, stats.draws],
  }
}

function applyNativeState(target: NativeAIState, affinityRef: { current: Affinity | null }, profileRef: { current: ProfileCounts }, perceptionRef: { current: { axis: number; face: number; space: number } }, statsRef: { current: GameStats }) {
  const affinity: Affinity = new Map()
  for (let i = 0; i + 2 < target.affinity.length; i += 3) {
    const side = target.affinity[i]
    const cell = target.affinity[i + 1]
    const row = affinity.get(side) ?? new Map<number, number>()
    row.set(cell, target.affinity[i + 2])
    affinity.set(side, row)
  }
  affinityRef.current = affinity
  profileRef.current = { attack: target.profile[0] ?? 0, defend: target.profile[1] ?? 0, neutral: target.profile[2] ?? 0 }
  perceptionRef.current = { axis: target.perception[0] ?? 0, face: target.perception[1] ?? 0, space: target.perception[2] ?? 0 }
  statsRef.current = { wins: target.stats[0] ?? 0, losses: target.stats[1] ?? 0, draws: target.stats[2] ?? 0 }
}

export function GameScreen() {
  const insets = useSafeAreaInsets()
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
  const [nativeSnapshot, setNativeSnapshot] = useState<NativeAIState | null>(null)

  const engineRef = useRef<boolean>(false)
  const boardRef = useRef<Board | null>(null)
  const affinityRef = useRef<Affinity | null>(null)
  const affinityLoadRef = useRef<Promise<Affinity> | null>(null)
  const profileRef = useRef<ProfileCounts>({ attack: 0, defend: 0, neutral: 0 })
  const perceptionRef = useRef<{ axis: number; face: number; space: number }>({ axis: 0, face: 0, space: 0 })
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
  const appStateRef = useRef<AppStateStatus>(AppState.currentState)
  const aiEpochRef = useRef(0)
  const inferenceInFlightRef = useRef(false)
  const resumeAiRef = useRef(false)
  const resumeDemoRef = useRef(false)

  // Create the native engine once; surface load failures gracefully.
  useEffect(() => {
    engineRef.current = isAvailable()
    if (!engineRef.current) setEngineError(ENGINE_UNAVAILABLE_MSG)
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
      if (p) profileRef.current = p
    })
  }, [])

  // Load the persistent 3D-perception profile + game stats once.
  useEffect(() => {
    void loadPerception().then((p) => {
      if (p) perceptionRef.current = p
    })
    void loadStats().then((s) => {
      if (s) {
        statsRef.current = s
        const total = s.wins + s.losses
        const rate = total < 2 ? 0.5 : s.wins / total
        adaptiveRef.current = Math.max(-1, Math.min(1, (0.55 - rate) * 2.5))
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
      const human = humanSideRef.current
      playSfx(winner === EMPTY ? 'draw' : winner === human ? 'win' : 'lose')
      const state = aiEndGame(winner)
      setNativeSnapshot(state)
      applyNativeState(state, affinityRef, profileRef, perceptionRef, statsRef)
      adaptiveRef.current = state.adaptive
      void saveStats(statsRef.current)
      void saveProfile(profileRef.current)
      void savePerception(perceptionRef.current)
      persistAffinity(affinityRef.current ?? new Map())
    },
    [persistAffinity],
  )

  const runAITurn = useCallback(async () => {
    if (appStateRef.current !== 'active') {
      resumeAiRef.current = true
      return
    }
    if (!boardRef.current || !engineRef.current) return
    if (overRef.current || thinkingRef.current) return
    const epoch = aiEpochRef.current
    const aiSide: Cell = humanSideRef.current === P1 ? P2 : P1
    thinkingRef.current = true
    // Flash starts immediately (the cube breathes) — the human's mark already
    // rendered, and the inference runs in the background below.
    setSnap((prev) => (prev.thinking ? prev : { ...prev, thinking: true }))
    if (timerRef.current) clearTimeout(timerRef.current)
    const startedAt = Date.now()
    const board = boardRef.current
    if (!board) return
    if (overRef.current) return
    inferenceInFlightRef.current = true
    try {
      const decision = await aiChooseMove(aiSide)
      const move = decision.move
      if (epoch !== aiEpochRef.current || appStateRef.current !== 'active') {
        inferenceInFlightRef.current = false
        thinkingRef.current = false
        resumeAiRef.current = true
        setSnap((prev) => ({ ...prev, thinking: false }))
        if (appStateRef.current === 'active') {
          resumeAiRef.current = false
          setTimeout(() => void runAITurn(), 0)
        }
        return
      }
      // Keep the flash visible for at least MIN_FLASH_MS so the player always
      // sees the move land AFTER the cube breathes (render → flash → resolve).
      const elapsed = Date.now() - startedAt
      if (elapsed < MIN_FLASH_MS) {
        await new Promise<void>((resolve) => setTimeout(resolve, MIN_FLASH_MS - elapsed))
      }
      if (epoch !== aiEpochRef.current || appStateRef.current !== 'active') {
        inferenceInFlightRef.current = false
        thinkingRef.current = false
        resumeAiRef.current = true
        setSnap((prev) => ({ ...prev, thinking: false }))
        if (appStateRef.current === 'active') {
          resumeAiRef.current = false
          setTimeout(() => void runAITurn(), 0)
        }
        return
      }
      if (overRef.current) {
        inferenceInFlightRef.current = false
        return
      }
      thinkingRef.current = false
      board.apply(move, aiSide)
      aiApplyMove(aiSide, move)
      setNativeSnapshot(aiState())
      playSfx('ai')
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
      inferenceInFlightRef.current = false
    } catch {
      inferenceInFlightRef.current = false
      thinkingRef.current = false
      setSnap((prev) => ({ ...prev, thinking: false }))
    }
  }, [endGame])

  // AI-vs-AI demo: plays the whole game by itself so the core loop can be
  // recorded/shown. Uses a throwaway predictor (empty memory) so demo games
  // never pollute the persistent opponent memory, and no win/loss reward.
  const runDemoTurn = useCallback(() => {
    if (appStateRef.current !== 'active') {
      resumeDemoRef.current = true
      return
    }
    const board = boardRef.current
    if (!board || !engineRef.current) return
    if (overRef.current || !demoRef.current) return
    const side = turnRef.current
    const epoch = aiEpochRef.current
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(async () => {
      try {
        if (appStateRef.current !== 'active' || epoch !== aiEpochRef.current) {
          resumeDemoRef.current = true
          return
        }
        inferenceInFlightRef.current = true
        const decision = await aiChooseMove(side)
        const move = decision.move
        if (
          overRef.current ||
          !demoRef.current ||
          appStateRef.current !== 'active' ||
          epoch !== aiEpochRef.current
        ) {
          inferenceInFlightRef.current = false
          resumeDemoRef.current = true
          if (appStateRef.current === 'active' && !overRef.current && demoRef.current) {
            resumeDemoRef.current = false
            setTimeout(() => runDemoTurn(), 0)
          }
          return
        }
        board.apply(move, side)
        aiApplyMove(side, move)
        setNativeSnapshot(aiState())
        movesRef.current.push(move)
        const outcome = board.outcome()
        if (outcome.over) {
          overRef.current = true
          inferenceInFlightRef.current = false
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
        inferenceInFlightRef.current = false
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
        inferenceInFlightRef.current = false
        demoRef.current = false
      }
    }, 700)
  }, [])

  // Do not spend CPU/native-engine time while the app is backgrounded. Any
  // in-flight result is invalidated; the pending AI/demo turn is resumed once
  // the app is active again instead of applying a stale move on return.
  useEffect(() => {
    const sub = AppState.addEventListener('change', (next) => {
      const wasActive = appStateRef.current === 'active'
      appStateRef.current = next

      if (next !== 'active') {
        if (demoRef.current && !overRef.current) {
          resumeDemoRef.current = true
        } else if (thinkingRef.current || timerRef.current != null) {
          resumeAiRef.current = true
        }
        aiEpochRef.current += 1
        if (timerRef.current) clearTimeout(timerRef.current)
        timerRef.current = null
        if (thinkingRef.current) {
          thinkingRef.current = false
          setSnap((prev) => (prev.thinking ? { ...prev, thinking: false } : prev))
        }
        return
      }

      if (!wasActive && !inferenceInFlightRef.current) {
        if (resumeDemoRef.current) {
          resumeDemoRef.current = false
          timerRef.current = setTimeout(() => {
            timerRef.current = null
            runDemoTurn()
          }, 0)
        } else if (resumeAiRef.current) {
          resumeAiRef.current = false
          timerRef.current = setTimeout(() => {
            timerRef.current = null
            void runAITurn()
          }, 0)
        }
      }
    })
    return () => sub.remove()
  }, [runAITurn, runDemoTurn])

  const startDemo = useCallback(
    (cfg: GameConfig) => {
      if (!engineRef.current) {
        setEngineError(ENGINE_UNAVAILABLE_MSG)
        setMenuVisible(false)
        return
      }
      if (timerRef.current) clearTimeout(timerRef.current)
      const board = new Board(cfg.size)
      // Fresh native session — demos don't teach persistent memory.
      aiStart(nativeConfig(cfg, null, { attack: 0, defend: 0, neutral: 0 }, { axis: 0, face: 0, space: 0 }, emptyStats(), 0))
      setNativeSnapshot(aiState())
      boardRef.current = board
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
      if (!engineRef.current) {
        setEngineError(ENGINE_UNAVAILABLE_MSG)
        setMenuVisible(false)
        return
      }
      if (timerRef.current) clearTimeout(timerRef.current)
      if (!affinityRef.current) {
        affinityRef.current = await (affinityLoadRef.current ?? loadAffinity())
      }
      const board = new Board(cfg.size)
      aiStart(nativeConfig(cfg, affinityRef.current, profileRef.current, perceptionRef.current, statsRef.current, cfg.difficulty === 'hard' ? 0 : adaptiveRef.current))
      setNativeSnapshot(aiState())
      boardRef.current = board
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
    // Screenshot mode (build with EXPO_PUBLIC_SCREENSHOT=1): start an AI-vs-AI
    // demo with no overlays so App Store screenshots capture the game board.
    if (process.env.EXPO_PUBLIC_SCREENSHOT === '1') {
      setMenuVisible(false)
      const t = setInterval(() => {
        if (engineRef.current && !demoRef.current && !overRef.current) {
          clearInterval(t)
          startDemo({ size: 3, difficulty: 'medium', humanSide: 1 })
        }
      }, 120)
      return () => clearInterval(t)
    }
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
      hapticSelection()
    },
    [pending, snap.currentPlayer],
  )

  const placeMove = useCallback(() => {
    const board = boardRef.current
    if (!board || !engineRef.current) return
    if (pending == null) return
    if (demoRef.current || thinkingRef.current || overRef.current) return
    if (snap.currentPlayer !== humanSideRef.current) return
    if (board.cells[pending] !== EMPTY) return
    const human = humanSideRef.current
    board.apply(pending, human)
    aiApplyMove(human, pending)
    setNativeSnapshot(aiState())
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
    // Render the user's mark first, THEN start the AI inference — on larger
    // boards the render (instanced frame loop + pop-in) is the slow part, and
    // the placed mark should visibly land before the cube starts "thinking".
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      runAITurn()
    }, 180)
  }, [pending, snap.currentPlayer, runAITurn, endGame])

  const cancelPending = useCallback(() => {
    setPending(null)
  }, [])

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
    aiSetBoard(board.cells)
    setNativeSnapshot(aiState())
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
    if (!engineRef.current) return
    if (demoRef.current || thinkingRef.current || overRef.current) return
    const human = humanSideRef.current
    try {
      const hint = await aiHint(human)
      if (hint < 0) return
      if (demoRef.current || overRef.current) return
      setPending(hint)
      setSnap((prev) => ({ ...prev, hintIndex: hint }))
    } catch {
      // ignore
    }
  }, [])

  const playAgain = useCallback(() => {
    startGame(configRef.current)
  }, [startGame])

  const handleStart = useCallback((cfg: GameConfig) => {
    startGame(cfg)
  }, [startGame])

  const openMenu = useCallback(() => {
    setMenuVisible(true)
  }, [])

  const onHint = useCallback(() => {
    void showHint()
  }, [showHint])

  const onUndo = useCallback(() => {
    undoMove()
  }, [undoMove])

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
    <View style={[styles.root, { paddingTop: insets.top }]}>
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
          winner={snap.winner}
          over={snap.over}
          humanSide={config.humanSide}
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
        <View style={[styles.bottom, { paddingBottom: insets.bottom }]}>
          <StatusBar
            state={snap}
            humanSide={config.humanSide}
            onPlayAgain={playAgain}
            onHint={onHint}
            onUndo={onUndo}
          />
        </View>
      )}

      {pending != null && isHumanTurn && (
        <View style={[styles.actionBar, { bottom: spacing(15) + insets.bottom }]}>
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
          engine={null}
          board={boardRef.current}
          humanSide={config.humanSide}
          difficulty={config.difficulty}
          predictor={null}
          mover={null}
          profile={null}
          perception={null}
          stats={statsRef.current}
          adaptive={adaptiveRef.current}
          nativeState={nativeSnapshot}
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
