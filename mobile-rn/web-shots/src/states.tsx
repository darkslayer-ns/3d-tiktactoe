import { Image, Pressable, StyleSheet, Text, View } from 'react-native'
import { Theme, fontSize, radius, spacing } from '../../src/ui/theme'
import { Board3D } from '../../src/ui/components/Board3D'
import { MenuSheet } from '../../src/ui/components/MenuSheet'
import { WelcomeOverlay } from '../../src/ui/components/WelcomeOverlay'
import { emptyState, type GameState } from '../../src/ai/types'
import { EMPTY, P1, P2, type Cell, type Coord } from '../../src/game/types'

import podiumWinX from '../../assets/models/podium_win_x.png?url'
import fidgetX from '../../assets/sprites/fidget_x.png?url'
import fidgetO from '../../assets/sprites/fidget_o.png?url'

// Fidget sprite sheet grid (must match TurnFidget.tsx).
const FIDGET_COLS = 12
const FIDGET_ROWS = 10

function boardWith(place: Array<[number, Cell]>, size = 3): Cell[] {
  const cells = new Array<Cell>(size ** 3).fill(EMPTY)
  for (const [i, c] of place) cells[i] = c
  return cells
}

function makeState(partial: Partial<GameState> = {}): GameState {
  return { ...emptyState(3, P1), ...partial }
}

// A plausible mid-game on 3×3×3: X at 0/13/20, O at 2/6/18.
const midCells = boardWith([
  [0, P1],
  [13, P1],
  [20, P1],
  [2, P2],
  [6, P2],
  [18, P2],
])

// X wins along the (1,1,1) space diagonal.
const winCells = boardWith([
  [0, P1],
  [13, P1],
  [26, P1],
  [2, P2],
  [6, P2],
  [18, P2],
])
const winLine: Coord[] = [
  [0, 0, 0],
  [1, 1, 1],
  [2, 2, 2],
]

const noop = () => {}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Theme.bg },
  board: { flex: 1, position: 'relative', backgroundColor: Theme.bg },
  topBar: {
    position: 'absolute',
    top: spacing(15),
    left: spacing(4),
    flexDirection: 'row',
    alignItems: 'center',
  },
  turnRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing(2),
  },
  sheetClip: {
    overflow: 'hidden',
  },
  turnText: {
    fontSize: fontSize(15),
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  bottomBar: {
    borderTopWidth: 1,
    borderTopColor: Theme.border,
    backgroundColor: 'rgba(2, 6, 23, 0.85)',
    paddingHorizontal: spacing(4),
    paddingTop: spacing(2),
    paddingBottom: spacing(6),
  },
  bottomActions: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: spacing(2),
  },
  btn: {
    paddingVertical: spacing(1.5),
    paddingHorizontal: spacing(3),
    borderRadius: radius(2),
    borderWidth: 1,
    borderColor: Theme.border,
    backgroundColor: 'rgba(2, 6, 23, 0.85)',
  },
  btnText: {
    color: Theme.muted,
    fontSize: fontSize(12),
    fontWeight: '700',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  btnHint: {
    borderColor: 'rgba(34, 211, 238, 0.45)',
    backgroundColor: 'rgba(34, 211, 238, 0.1)',
  },
  btnHintText: {
    color: Theme.cyan,
    fontSize: fontSize(12),
    fontWeight: '700',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  // ---- iOS chrome (status bar + home indicator) ----
  frame: { flex: 1, position: 'relative', backgroundColor: Theme.bg },
  statusBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 44,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 28,
  },
  statusTime: { color: '#ffffff', fontSize: 15, fontWeight: '700', letterSpacing: 0.2 },
  statusRight: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  signal: { flexDirection: 'row', alignItems: 'flex-end', gap: 1.5 },
  signalBar: { width: 3, backgroundColor: '#ffffff', borderRadius: 1 },
  battery: {
    width: 25,
    height: 12,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.55)',
    padding: 1.5,
    justifyContent: 'center',
  },
  batteryFill: { width: 15, height: 8, borderRadius: 2, backgroundColor: '#ffffff' },
  homeWrap: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 20,
    alignItems: 'center',
    justifyContent: 'flex-end',
    paddingBottom: 8,
  },
  homePill: { width: 134, height: 5, borderRadius: 2.5, backgroundColor: 'rgba(255,255,255,0.85)' },
  // ---- game-over card ----
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  overlayBackdrop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(2, 6, 23, 0.55)',
  },
  card: {
    alignItems: 'center',
    paddingHorizontal: spacing(10),
    paddingVertical: spacing(7),
    borderRadius: radius(6),
    borderWidth: 1,
    borderColor: Theme.cyan,
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
    maxWidth: '92%',
    shadowColor: Theme.cyan,
    shadowOpacity: 0.5,
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 0 },
  },
  podium: { width: 190, height: 190, marginBottom: spacing(2) },
  title: {
    fontSize: fontSize(30),
    fontWeight: '800',
    letterSpacing: 1,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 14,
  },
  playBtn: {
    marginTop: spacing(4),
    paddingVertical: spacing(3.5),
    paddingHorizontal: spacing(12),
    borderRadius: radius(6),
    borderWidth: 2,
    alignItems: 'center',
  },
  playText: { fontSize: fontSize(15), fontWeight: '800', letterSpacing: 2 },
  menuText: {
    marginTop: spacing(4),
    color: Theme.muted,
    fontSize: fontSize(11),
    fontWeight: '700',
    letterSpacing: 2,
  },
})

function Board({ cells, hint = null, win = null, thinking = false }: { cells: Cell[]; hint?: number | null; win?: Coord[] | null; thinking?: boolean }) {
  return (
    <View style={styles.board}>
      <Board3D
        size={3}
        cells={cells}
        onCellClick={noop}
        pendingIndex={null}
        winningLine={win}
        lastAiMove={null}
        hintIndex={hint}
        thinking={thinking}
        interactive={() => false}
        startKey={0}
      />
    </View>
  )
}

function StatusBarChrome({ compact = false }: { compact?: boolean }) {
  return (
    <View style={[styles.statusBar, compact && { height: 24, paddingHorizontal: 24 }]}>
      <Text style={styles.statusTime}>9:41</Text>
      <View style={styles.statusRight}>
        <View style={styles.signal}>
          <View style={[styles.signalBar, { height: 4 }]} />
          <View style={[styles.signalBar, { height: 6 }]} />
          <View style={[styles.signalBar, { height: 8 }]} />
          <View style={[styles.signalBar, { height: 10 }]} />
        </View>
        <View style={styles.battery}>
          <View style={styles.batteryFill} />
        </View>
      </View>
    </View>
  )
}

function HomeIndicator() {
  return (
    <View style={styles.homeWrap} pointerEvents="none">
      <View style={styles.homePill} />
    </View>
  )
}

function TurnIndicator({ mark }: { mark: 'X' | 'O' }) {
  const accent = mark === 'X' ? Theme.cyan : Theme.pink
  const sheet = mark === 'X' ? fidgetX : fidgetO
  const size = 44
  return (
    <View style={styles.turnRow}>
      <View style={[styles.sheetClip, { width: size, height: size }]}>
        <Image
          source={{ uri: sheet }}
          resizeMode="stretch"
          style={{
            width: FIDGET_COLS * size,
            height: FIDGET_ROWS * size,
            transform: [{ translateX: 0 }, { translateY: 0 }],
          }}
        />
      </View>
      <Text style={[styles.turnText, { color: accent }]}>Your turn</Text>
    </View>
  )
}

function BottomBar({ showHint = false, showUndo = false }: { showHint?: boolean; showUndo?: boolean }) {
  return (
    <View style={styles.bottomBar}>
      <View style={styles.bottomActions}>
        {showHint && (
          <Pressable style={[styles.btn, styles.btnHint]}>
            <Text style={styles.btnHintText}>Hint</Text>
          </Pressable>
        )}
        {showUndo && (
          <Pressable style={styles.btn}>
            <Text style={styles.btnText}>Undo</Text>
          </Pressable>
        )}
        <Pressable style={styles.btn}>
          <Text style={styles.btnText}>New game</Text>
        </Pressable>
      </View>
    </View>
  )
}

function GameOverCard() {
  return (
    <View style={styles.overlay}>
      <View style={styles.overlayBackdrop} />
      <View style={styles.card}>
        <Image source={{ uri: podiumWinX }} style={styles.podium} resizeMode="contain" />
        <Text style={[styles.title, { color: Theme.cyan, textShadowColor: Theme.cyan }]}>YOU WIN!</Text>
        <View style={[styles.playBtn, { borderColor: Theme.cyan }]}>
          <Text style={[styles.playText, { color: Theme.cyan }]}>PLAY AGAIN</Text>
        </View>
        <Text style={styles.menuText}>MENU</Text>
      </View>
    </View>
  )
}

function Chrome({ children, compact = false }: { children: React.ReactNode; compact?: boolean }) {
  return (
    <View style={styles.frame}>
      {children}
      <StatusBarChrome compact={compact} />
      <HomeIndicator />
    </View>
  )
}

const states: Record<string, React.ReactNode> = {
  menu: (
    <Chrome>
      <View style={styles.root}>
        <Board cells={midCells} />
        <MenuSheet visible onStart={noop} onHowTo={noop} />
      </View>
    </Chrome>
  ),

  welcome: (
    <Chrome>
      <View style={styles.root}>
        <Board cells={midCells} />
        <WelcomeOverlay onStart={noop} />
      </View>
    </Chrome>
  ),

  'your-turn': (
    <Chrome>
      <View style={styles.root}>
        <Board cells={midCells} />
        <View style={styles.topBar}>
          <TurnIndicator mark="X" />
        </View>
        <BottomBar showHint showUndo />
      </View>
    </Chrome>
  ),

  opponent: (
    <Chrome>
      <View style={styles.root}>
        <Board cells={midCells} thinking />
        <BottomBar showUndo />
      </View>
    </Chrome>
  ),

  hint: (
    <Chrome>
      <View style={styles.root}>
        <Board cells={midCells} hint={4} />
        <View style={styles.topBar}>
          <TurnIndicator mark="X" />
        </View>
        <BottomBar showHint showUndo />
      </View>
    </Chrome>
  ),

  gameover: (
    <Chrome>
      <View style={styles.root}>
        <Board cells={winCells} win={winLine} />
        <GameOverCard />
      </View>
    </Chrome>
  ),
}

export function States() {
  const params = new URLSearchParams(window.location.search)
  const state = params.get('state') ?? 'menu'
  const compact = params.get('device') === 'ipad'
  return <>{states[state] ?? states.menu}</>
}
