/**
 * Game-over overlay (Stitch "Win Overlay" / "Loss Overlay" design):
 * a frosted-glass card floating center with a neon bloom title, an icon, a
 * pulsing glow, and a glowing PLAY AGAIN button. Animates in with a spring
 * scale + fade.
 */

import { useEffect, useRef } from 'react'
import { Animated, Pressable, StyleSheet, Text, View } from 'react-native'
import { BlurView } from 'expo-blur'
import { Theme, fontSize, radius, spacing } from '../theme'
import { EMPTY, type Cell } from '../../game/types'

interface GameOverOverlayProps {
  winner: Cell
  humanSide: Cell
  onPlayAgain: () => void
  onMenu: () => void
}

export function GameOverOverlay({ winner, humanSide, onPlayAgain, onMenu }: GameOverOverlayProps) {
  const enter = useRef(new Animated.Value(0)).current
  const glow = useRef(new Animated.Value(0)).current

  useEffect(() => {
    Animated.timing(enter, { toValue: 1, duration: 420, useNativeDriver: true }).start()
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(glow, { toValue: 1, duration: 850, useNativeDriver: true }),
        Animated.timing(glow, { toValue: 0, duration: 850, useNativeDriver: true }),
      ]),
    )
    loop.start()
    return () => loop.stop()
  }, [enter, glow])

  const isDraw = winner === EMPTY
  const isWin = !isDraw && winner === humanSide
  const accent = isDraw ? Theme.purple : isWin ? Theme.cyan : Theme.pink
  const title = isDraw ? 'DRAW' : isWin ? 'YOU WIN!' : 'ISOCUBE WINS'
  const icon = isDraw ? '◆' : isWin ? '🏆' : '▣'
  const iconColor = isDraw ? Theme.purple : isWin ? Theme.pink : Theme.cyan

  const glowOpacity = glow.interpolate({ inputRange: [0, 1], outputRange: [0.45, 1] })

  return (
    <View style={styles.overlay}>
      <BlurView intensity={30} tint="dark" style={styles.backdrop} />
      <Animated.View
        style={[
          styles.card,
          {
            borderColor: accent,
            shadowColor: accent,
            opacity: enter,
            transform: [{ scale: enter.interpolate({ inputRange: [0, 1], outputRange: [0.82, 1] }) }],
          },
        ]}
      >
        <Animated.Text style={[styles.icon, { color: iconColor, opacity: glowOpacity }]}>
          {icon}
        </Animated.Text>
        <Animated.Text
          style={[styles.title, { color: accent, textShadowColor: accent, opacity: glowOpacity }]}
        >
          {title}
        </Animated.Text>
        <Pressable onPress={onPlayAgain} style={[styles.playBtn, { borderColor: accent }]}>
          <Animated.Text style={[styles.playText, { color: accent, opacity: glowOpacity }]}>
            PLAY AGAIN
          </Animated.Text>
        </Pressable>
        <Pressable onPress={onMenu} style={styles.menuLink}>
          <Text style={styles.menuText}>MENU</Text>
        </Pressable>
      </Animated.View>
    </View>
  )
}

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  backdrop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  card: {
    alignItems: 'center',
    paddingHorizontal: spacing(10),
    paddingVertical: spacing(9),
    borderRadius: radius(6),
    borderWidth: 1,
    backgroundColor: 'rgba(15, 23, 42, 0.85)',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 24,
  },
  icon: {
    fontSize: fontSize(46),
    marginBottom: spacing(3),
  },
  title: {
    fontSize: fontSize(30),
    fontWeight: '800',
    letterSpacing: 1,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 14,
  },
  playBtn: {
    marginTop: spacing(6),
    paddingVertical: spacing(3.5),
    paddingHorizontal: spacing(12),
    borderRadius: radius(6),
    borderWidth: 2,
    alignItems: 'center',
  },
  playText: {
    fontSize: fontSize(15),
    fontWeight: '800',
    letterSpacing: 2,
  },
  menuLink: {
    marginTop: spacing(4),
    padding: spacing(2),
  },
  menuText: {
    color: Theme.muted,
    fontSize: fontSize(11),
    fontWeight: '700',
    letterSpacing: 2,
  },
})