/**
 * Slim status banner: whose turn it is, "AI thinking…" with animated dots,
 * or the win/draw result with a "Play again" button.
 */

import { useEffect, useRef } from 'react'
import { Animated, Pressable, StyleSheet, Text, View } from 'react-native'
import { Theme, fontSize, radius, spacing } from '../theme'
import type { GameState } from '../../ai/types'
import { EMPTY, P1, type Cell } from '../../game/types'

interface StatusBarProps {
  state: GameState
  humanSide: Cell
  onPlayAgain: () => void
  onNewGame?: () => void
  onHint?: () => void
  onUndo?: () => void
}

function markOf(player: Cell): string {
  return player === P1 ? 'X' : 'O'
}

export function StatusBar({
  state,
  humanSide,
  onPlayAgain,
  onNewGame,
  onHint,
  onUndo,
}: StatusBarProps) {
  const pulse = useRef(new Animated.Value(1)).current

  useEffect(() => {
    if (!state.thinking) return
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 0.35, duration: 380, useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 1, duration: 380, useNativeDriver: true }),
      ]),
    )
    loop.start()
    return () => loop.stop()
  }, [state.thinking, pulse])

  const renderMark = (player: Cell, animating = false) => {
    const accent = player === P1 ? Theme.cyan : Theme.pink
    return (
      <Animated.View style={[styles.badge, { borderColor: accent, opacity: animating ? pulse : 1 }]}>
        <Text style={[styles.badgeText, { color: accent }]}>{markOf(player)}</Text>
      </Animated.View>
    )
  }

  if (state.over) {
    const result =
      state.winner === EMPTY
        ? 'Draw'
        : state.winner === humanSide
          ? 'You win!'
          : 'ISOCUBE wins'
    const accent = state.winner === EMPTY ? Theme.muted : state.winner === humanSide ? Theme.cyan : Theme.pink
    const winnerMark = state.winner === EMPTY ? null : state.winner
    return (
      <View style={styles.bar}>
        {winnerMark ? renderMark(winnerMark) : <View style={[styles.badge, { borderColor: Theme.muted }]}><Text style={[styles.badgeText, { color: Theme.muted }]}>–</Text></View>}
        <Text style={[styles.text, { color: accent }]}>{result}</Text>
        <Pressable onPress={onPlayAgain} style={styles.playAgain}>
          <Text style={styles.playAgainText}>Play again</Text>
        </Pressable>
        {onUndo != null && (
          <Pressable onPress={onUndo} style={styles.newGame}>
            <Text style={styles.newGameText}>Undo</Text>
          </Pressable>
        )}
      </View>
    )
  }

  const player = state.currentPlayer
  const isHumanTurn = player === humanSide
  const accent = player === P1 ? Theme.cyan : Theme.pink
  return (
    <View style={styles.bar}>
      {renderMark(player, state.thinking)}
      <Text style={[styles.text, { color: accent }]}>
        {isHumanTurn ? 'Your turn' : "Opponent's turn"}
      </Text>
      <View style={styles.actions}>
        {isHumanTurn && onHint != null && (
          <Pressable onPress={onHint} style={styles.hint}>
            <Text style={styles.hintText}>Hint</Text>
          </Pressable>
        )}
        {!state.thinking && onUndo != null && (
          <Pressable onPress={onUndo} style={styles.newGame}>
            <Text style={styles.newGameText}>Undo</Text>
          </Pressable>
        )}
        {onNewGame != null && (
          <Pressable onPress={onNewGame} style={styles.newGame}>
            <Text style={styles.newGameText}>New game</Text>
          </Pressable>
        )}
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  bar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing(3),
    height: spacing(13),
    paddingHorizontal: spacing(4),
    backgroundColor: 'rgba(2, 6, 23, 0.85)',
    borderTopWidth: 1,
    borderTopColor: Theme.border,
  },
  badge: {
    width: 26,
    height: 26,
    borderRadius: radius(2),
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
  },
  badgeText: {
    fontSize: fontSize(14),
    fontWeight: '800',
  },
  text: {
    color: Theme.text,
    fontSize: fontSize(14),
    fontWeight: '700',
    letterSpacing: 0.5,
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
  newGame: {
    paddingVertical: spacing(1.5),
    paddingHorizontal: spacing(3),
    borderRadius: radius(2),
    borderWidth: 1,
    borderColor: Theme.border,
  },
  newGameText: {
    color: Theme.muted,
    fontSize: fontSize(12),
    fontWeight: '700',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  hint: {
    paddingVertical: spacing(1.5),
    paddingHorizontal: spacing(3),
    borderRadius: radius(2),
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.45)',
    backgroundColor: 'rgba(34, 211, 238, 0.1)',
  },
  hintText: {
    color: Theme.cyan,
    fontSize: fontSize(12),
    fontWeight: '700',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  actions: {
    marginLeft: 'auto',
    flexDirection: 'row',
    gap: spacing(2),
  },
})