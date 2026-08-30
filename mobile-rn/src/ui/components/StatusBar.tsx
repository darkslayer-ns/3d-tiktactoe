/**
 * Slim status banner: whose turn it is, "AI thinking…" with animated dots,
 * or the win/draw result with a "Play again" button.
 */

import { useEffect, useState } from 'react'
import { Pressable, StyleSheet, Text, View } from 'react-native'
import { Theme, fontSize, radius, spacing } from '../theme'
import type { GameState } from '../../ai/types'
import { EMPTY, P1, type Cell } from '../../game/types'

interface StatusBarProps {
  state: GameState
  humanSide: Cell
  onPlayAgain: () => void
}

function markOf(player: Cell): string {
  return player === P1 ? 'X' : 'O'
}

export function StatusBar({ state, humanSide, onPlayAgain }: StatusBarProps) {
  const [dots, setDots] = useState(0)

  useEffect(() => {
    if (!state.thinking) return
    const id = setInterval(() => setDots((d) => (d + 1) % 4), 380)
    return () => clearInterval(id)
  }, [state.thinking])

  if (state.over) {
    const result =
      state.winner === EMPTY
        ? 'Draw'
        : state.winner === humanSide
          ? 'You win!'
          : 'AI wins'
    const accent = state.winner === EMPTY ? Theme.muted : Theme.cyan
    return (
      <View style={styles.bar}>
        <View style={[styles.dot, { backgroundColor: accent, shadowColor: accent }]} />
        <Text style={[styles.text, { color: accent }]}>{result}</Text>
        <Pressable onPress={onPlayAgain} style={styles.playAgain}>
          <Text style={styles.playAgainText}>Play again</Text>
        </Pressable>
      </View>
    )
  }

  if (state.thinking) {
    return (
      <View style={styles.bar}>
        <View style={styles.spinner} />
        <Text style={styles.text}>
          AI thinking{'.'.repeat(dots)}
          <Text style={styles.dim}>{'.'.repeat(3 - dots)}</Text>
        </Text>
      </View>
    )
  }

  const player = state.currentPlayer
  const isHumanTurn = player === humanSide
  const accent = player === P1 ? Theme.cyan : Theme.pink
  return (
    <View style={styles.bar}>
      <View style={[styles.dot, { backgroundColor: accent, shadowColor: accent }]} />
      <Text style={[styles.text, { color: accent }]}>
        {isHumanTurn ? 'Your turn' : "AI's turn"} · {markOf(player)}
      </Text>
    </View>
  )
}

const styles = StyleSheet.create({
  bar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing(2),
    minHeight: spacing(9),
    paddingHorizontal: spacing(4),
    paddingVertical: spacing(2),
    backgroundColor: 'rgba(2, 6, 23, 0.85)',
    borderTopWidth: 1,
    borderTopColor: Theme.border,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.9,
    shadowRadius: 6,
  },
  text: {
    color: Theme.text,
    fontSize: fontSize(14),
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  dim: {
    opacity: 0.25,
  },
  spinner: {
    width: 14,
    height: 14,
    borderRadius: 7,
    borderWidth: 2,
    borderColor: 'rgba(34, 211, 238, 0.25)',
    borderTopColor: Theme.cyan,
  },
  playAgain: {
    marginLeft: 'auto',
    paddingVertical: spacing(1.5),
    paddingHorizontal: spacing(3),
    borderRadius: radius(2),
    borderWidth: 1,
    borderColor: Theme.cyan,
    backgroundColor: 'rgba(34, 211, 238, 0.12)',
  },
  playAgainText: {
    color: Theme.cyan,
    fontSize: fontSize(13),
    fontWeight: '700',
  },
})