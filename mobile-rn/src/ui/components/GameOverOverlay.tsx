/**
 * Game-over overlay (Stitch "Win Overlay" / "Loss Overlay" design):
 * a frosted-glass card floating center with a neon bloom title, a pulsing
 * glow, and a glowing PLAY AGAIN button. Wins use a Blender-rendered winner
 * mark on the worn podium; draws keep the gold trophy.
 */

import { useEffect, useRef } from 'react'
import { Animated, Image, Pressable, StyleSheet, Text, View } from 'react-native'
import { BlurView } from 'expo-blur'
import { Theme, fontSize, radius, spacing } from '../theme'
import { EMPTY, P1, type Cell } from '../../game/types'
import { WinnerCelebration } from './WinnerCelebration'

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
    Animated.spring(enter, { toValue: 1, damping: 18, stiffness: 200, useNativeDriver: true }).start()
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
        {isDraw ? (
          <Image
            source={require('../../../assets/models/trophy.png')}
            style={styles.icon}
            resizeMode="contain"
          />
        ) : (
          <WinnerCelebration mark={winner === P1 ? 'X' : 'O'} size={190} />
        )}
        <Animated.Text
          style={[
            styles.title,
            !isDraw && styles.winnerTitle,
            { color: accent, textShadowColor: accent, opacity: glowOpacity },
          ]}
        >
          {title}
        </Animated.Text>
        <Pressable
          onPress={onPlayAgain}
          style={({ pressed }) => [
            styles.playBtn,
            { backgroundColor: accent, shadowColor: accent },
            pressed && styles.playBtnPressed,
          ]}
        >
          <Text style={styles.playText}>PLAY AGAIN</Text>
        </Pressable>
        <Pressable onPress={onMenu} style={({ pressed }) => [styles.menuLink, pressed && { opacity: 0.6 }]}>
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
    paddingVertical: spacing(7),
    borderRadius: radius(6),
    borderWidth: 1,
    backgroundColor: 'rgba(15, 23, 42, 0.85)',
    maxWidth: '92%',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 24,
  },
  icon: {
    width: 116,
    height: 116,
    marginBottom: spacing(2),
  },
  title: {
    fontSize: fontSize(30),
    fontWeight: '800',
    letterSpacing: 1,
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 14,
  },
  winnerTitle: {
    fontSize: fontSize(22),
    letterSpacing: 0.7,
    textShadowRadius: 10,
  },
  playBtn: {
    marginTop: spacing(4),
    paddingVertical: spacing(3.5),
    paddingHorizontal: spacing(12),
    borderRadius: radius(6),
    alignItems: 'center',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.55,
    shadowRadius: 18,
    elevation: 10,
  },
  playBtnPressed: {
    opacity: 0.85,
    transform: [{ scale: 0.97 }],
  },
  playText: {
    color: Theme.bg,
    fontSize: fontSize(15),
    fontWeight: '900',
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
