/** Blender-rendered podium celebration flipbook for X/O wins. */

import { useEffect, useMemo, useRef } from 'react'
import { Animated, StyleSheet, View } from 'react-native'

const COLS = 10
const ROWS = 6
const FRAMES = 60
const FPS = 30

const SHEETS = {
  X: require('../../../assets/models/podium_celebration_x.png'),
  O: require('../../../assets/models/podium_celebration_o.png'),
} as const

export type WinnerMark = keyof typeof SHEETS

interface WinnerCelebrationProps {
  mark: WinnerMark
  size?: number
}

export function WinnerCelebration({ mark, size = 160 }: WinnerCelebrationProps) {
  const frame = useRef(new Animated.Value(0)).current
  const raf = useRef(0)
  const start = useRef<number | null>(null)
  const indices = useMemo(() => Array.from({ length: FRAMES }, (_, i) => i), [])

  useEffect(() => {
    start.current = null
    const tick = (time: number) => {
      if (start.current == null) start.current = time
      frame.setValue(Math.floor(((time - start.current) / 1000) * FPS) % FRAMES)
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [frame])

  const translateX = useMemo(
    () =>
      frame.interpolate({
        inputRange: indices,
        outputRange: indices.map((i) => -(i % COLS) * size),
      }),
    [frame, indices, size],
  )
  const translateY = useMemo(
    () =>
      frame.interpolate({
        inputRange: indices,
        outputRange: indices.map((i) => -Math.floor(i / COLS) * size),
      }),
    [frame, indices, size],
  )

  return (
    <View style={[styles.clip, { width: size, height: size }]} pointerEvents="none">
      <Animated.Image
        source={SHEETS[mark]}
        resizeMode="stretch"
        style={{
          width: COLS * size,
          height: ROWS * size,
          transform: [{ translateX }, { translateY }],
        }}
      />
    </View>
  )
}

const styles = StyleSheet.create({
  clip: {
    overflow: 'hidden',
  },
})
