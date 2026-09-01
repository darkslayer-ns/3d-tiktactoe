import { StyleSheet, View } from 'react-native'
import { Theme } from '../../src/ui/theme'
import { Board3D } from '../../src/ui/components/Board3D'
import { MenuSheet } from '../../src/ui/components/MenuSheet'
import { StatusBar } from '../../src/ui/components/StatusBar'
import { WelcomeOverlay } from '../../src/ui/components/WelcomeOverlay'
import { emptyState, type GameState } from '../../src/ai/types'
import { EMPTY, P1, P2, type Cell, type Coord } from '../../src/game/types'

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

const states: Record<string, React.ReactNode> = {
  menu: (
    <View style={styles.root}>
      <Board cells={midCells} />
      <MenuSheet visible onStart={noop} onHowTo={noop} />
    </View>
  ),

  welcome: (
    <View style={styles.root}>
      <Board cells={midCells} />
      <WelcomeOverlay onStart={noop} />
    </View>
  ),

  'your-turn': (
    <View style={styles.root}>
      <Board cells={midCells} />
      <StatusBar state={makeState({ currentPlayer: P1 })} humanSide={P1} onPlayAgain={noop} onNewGame={noop} onHint={noop} onUndo={noop} />
    </View>
  ),

  opponent: (
    <View style={styles.root}>
      <Board cells={midCells} />
      <StatusBar state={makeState({ currentPlayer: P2 })} humanSide={P1} onPlayAgain={noop} onNewGame={noop} onUndo={noop} />
    </View>
  ),

  hint: (
    <View style={styles.root}>
      <Board cells={midCells} hint={13} />
      <StatusBar state={makeState({ currentPlayer: P1 })} humanSide={P1} onPlayAgain={noop} onNewGame={noop} onHint={noop} onUndo={noop} />
    </View>
  ),

  gameover: (
    <View style={styles.root}>
      <Board cells={winCells} win={winLine} />
      <StatusBar state={makeState({ currentPlayer: P1, winner: P1, over: true })} humanSide={P1} onPlayAgain={noop} onUndo={noop} />
    </View>
  ),
}

export function States() {
  const state = new URLSearchParams(window.location.search).get('state') ?? 'menu'
  return <>{states[state] ?? states.menu}</>
}