/**
 * Floating "whose turn" mascot: the current player's mark doing a funny
 * impatient loop, rendered in Blender as a 60fps sprite sheet
 * (assets/sprites/fidget_{x,o}.png, 12x10 grid of 160px cells).
 *
 * Playback is a rAF loop that writes an integer frame index into an
 * Animated.Value; two interpolations map it to the sprite-sheet crop offset.
 * No per-frame React state, so it never re-renders the tree.
 */

import { useEffect, useMemo, useRef } from 'react'
import { Animated, StyleSheet, View } from 'react-native'

export const FIDGET_COLS = 12
export const FIDGET_ROWS = 10
export const FIDGET_FRAMES = FIDGET_COLS * FIDGET_ROWS
export const FIDGET_FPS = 60

const SHEETS = {
  X: require('../../../assets/sprites/fidget_x.png'),
  O: require('../../../assets/sprites/fidget_o.png'),
} as const

export type FidgetMark = keyof typeof SHEETS

interface TurnFidgetProps {
  mark: FidgetMark
  /** rendered cell size in dp */
  size?: number
  testID?: string
}

export function TurnFidget({ mark, size = 72, testID }: TurnFidgetProps) {
  const frame = useRef(new Animated.Value(0)).current
  const raf = useRef(0)
  const start = useRef<number | null>(null)

  useEffect(() => {
    start.current = null
    const tick = (t: number) => {
      if (start.current == null) start.current = t
      frame.setValue(Math.floor(((t - start.current) / 1000) * FIDGET_FPS) % FIDGET_FRAMES)
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [frame])

  const frames = useMemo(() => Array.from({ length: FIDGET_FRAMES }, (_, i) => i), [])
  const tx = useMemo(
    () =>
      frame.interpolate({
        inputRange: frames,
        outputRange: frames.map((i) => -(i % FIDGET_COLS) * size),
      }),
    [frame, frames, size],
  )
  const ty = useMemo(
    () =>
      frame.interpolate({
        inputRange: frames,
        outputRange: frames.map((i) => -Math.floor(i / FIDGET_COLS) * size),
      }),
    [frame, frames, size],
  )

  return (
    <View style={[styles.clip, { width: size, height: size }]} testID={testID} pointerEvents="none">
      <Animated.Image
        source={SHEETS[mark]}
        resizeMode="stretch"
        style={{
          width: FIDGET_COLS * size,
          height: FIDGET_ROWS * size,
          transform: [{ translateX: tx }, { translateY: ty }],
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
