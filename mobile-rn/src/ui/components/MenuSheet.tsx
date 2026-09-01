/**
 * Home / menu sheet for ISOCUBE — matches the Stitch Home design:
 * neon wordmark + logo, tagline, settings panel with cyan-active toggles,
 * glowing outlined Start button.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Animated,
  Image,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native'
import { Theme, fontSize, radius, spacing } from '../theme'
import type { Difficulty, GameConfig } from '../../ai/types'
import type { Cell } from '../../game/types'

export type { GameConfig }

interface MenuSheetProps {
  visible: boolean
  onStart: (config: GameConfig) => void
  onHowTo?: (config: GameConfig) => void
}

interface Option<T> {
  value: T
  label: string
}

function Segmented<T extends string | number>({
  options,
  value,
  onChange,
}: {
  options: Option<T>[]
  value: T
  onChange: (value: T) => void
}) {
  return (
    <View style={styles.segmentRow}>
      {options.map((opt) => {
        const active = opt.value === value
        return (
          <Pressable
            key={String(opt.value)}
            onPress={() => onChange(opt.value)}
            style={[styles.segment, active && styles.segmentActive]}
          >
            <Text style={[styles.segmentText, active && styles.segmentTextActive]}>
              {opt.label}
            </Text>
          </Pressable>
        )
      })}
    </View>
  )
}

const SIDE_OPTIONS: Option<Cell>[] = [
  { value: 1, label: 'X First' },
  { value: 2, label: 'O Second' },
]

const DIFFICULTY_OPTIONS: Option<Difficulty>[] = [
  { value: 'easy', label: 'Easy' },
  { value: 'medium', label: 'Medium' },
  { value: 'hard', label: 'Hard' },
]

const MIN_SIZE = 3
const MAX_SIZE = 6

const THUMB = 24
const TRACK_H = 6

/** Minimal track-and-thumb slider. Uses absolute `pageX` + a measured window
 * origin (NOT child-relative locationX), a native-driven Animated.Value for
 * the thumb/fill, and emits onChange only when the snapped size actually
 * changes — so a drag never re-renders React and tracks the finger at 60fps. */
function SizeSlider({
  value,
  min,
  max,
  onChange,
}: {
  value: number
  min: number
  max: number
  onChange: (v: number) => void
}) {
  const touchRef = useRef<View>(null)
  const trackWRef = useRef(1)
  const originXRef = useRef(0)
  const lastEmittedRef = useRef(value)
  const pos = useRef(new Animated.Value(0, { useNativeDriver: true })).current // native-driven
  const [trackW, setTrackW] = useState(1)

  const fillScale = useMemo(
    () =>
      pos.interpolate({
        inputRange: [0, Math.max(1, trackW - THUMB)],
        outputRange: [0, 1],
        extrapolate: 'clamp',
      }),
    [pos, trackW],
  )

  const clamp01 = (v: number) => Math.min(1, Math.max(0, v))

  const setFromPageX = (pageX: number) => {
    const span = Math.max(1, trackWRef.current - THUMB)
    const tt = clamp01((pageX - originXRef.current - THUMB / 2) / span)
    pos.setValue(tt * span)
    const next = min + Math.round(tt * (max - min))
    if (next !== lastEmittedRef.current) {
      lastEmittedRef.current = next
      onChange(next)
    }
  }

  return (
    <View
      ref={touchRef}
      style={styles.sliderTouch}
      onLayout={(e) => {
        const w = e.nativeEvent.layout.width
        trackWRef.current = w
        setTrackW(w)
        pos.setValue(clamp01((value - min) / (max - min)) * Math.max(1, w - THUMB))
      }}
      onStartShouldSetResponder={() => true}
      onMoveShouldSetResponder={() => true}
      onResponderTerminationRequest={() => false}
      onResponderGrant={(e) => {
        touchRef.current?.measureInWindow((x) => {
          originXRef.current = x
          setFromPageX(e.nativeEvent.pageX)
        })
      }}
      onResponderMove={(e) => setFromPageX(e.nativeEvent.pageX)}
    >
      <View style={styles.sliderTrack}>
        <Animated.View
          pointerEvents="none"
          style={[styles.sliderFill, { transform: [{ scaleX: fillScale }] }]}
        />
        <Animated.View
          pointerEvents="none"
          style={[styles.sliderThumb, { transform: [{ translateX: pos }] }]}
        />
      </View>
    </View>
  )
}

export function MenuSheet({ visible, onStart, onHowTo }: MenuSheetProps) {
  const [humanSide, setHumanSide] = useState<Cell>(1)
  const [difficulty, setDifficulty] = useState<Difficulty>('hard')
  const [size, setSize] = useState(3)

  const translateY = useRef(new Animated.Value(700)).current

  useEffect(() => {
    if (visible) {
      translateY.setValue(700)
      Animated.spring(translateY, {
        toValue: 0,
        damping: 22,
        stiffness: 240,
        useNativeDriver: true,
      }).start()
    }
  }, [visible, translateY])

  return (
    <Modal visible={visible} transparent animationType="none" onRequestClose={() => {}}>
      <View style={styles.backdrop}>
        <Animated.View
          style={[styles.sheet, { transform: [{ translateY }] }]}
        >
          <View style={styles.handle} />

          <View style={styles.brand}>
            <Image
              source={require('../../../assets/isocube_logo.png')}
              style={styles.logo}
              resizeMode="contain"
            />
          </View>

          <View style={styles.panel}>
            <Text style={styles.sectionLabel}>You play</Text>
            <Segmented options={SIDE_OPTIONS} value={humanSide} onChange={setHumanSide} />

            <Text style={styles.sectionLabel}>Difficulty</Text>
            <Segmented options={DIFFICULTY_OPTIONS} value={difficulty} onChange={setDifficulty} />

            <Text style={styles.sectionLabel}>Board size</Text>
            <View style={styles.sizeRow}>
              <SizeSlider value={size} min={MIN_SIZE} max={MAX_SIZE} onChange={setSize} />
              <Text style={styles.sizeValue}>{size}×{size}×{size}</Text>
            </View>
          </View>

          <Pressable
            onPress={() => onStart({ humanSide, difficulty, size })}
            style={({ pressed }) => [styles.startBtn, pressed && styles.startBtnPressed]}
          >
            {({ pressed }) => (
              <Text style={[styles.startBtnText, pressed && styles.startBtnTextPressed]}>
                Start Game
              </Text>
            )}
          </Pressable>

          {onHowTo != null && (
            <Pressable
              onPress={() => onHowTo({ humanSide, difficulty, size })}
              style={({ pressed }) => [styles.howToBtn, pressed && styles.howToBtnPressed]}
            >
              {({ pressed }) => (
                <Text style={[styles.howToText, pressed && styles.howToTextPressed]}>How to play</Text>
              )}
            </Pressable>
          )}
        </Animated.View>
      </View>
    </Modal>
  )
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(2, 6, 23, 0.72)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: Theme.panel,
    borderTopLeftRadius: radius(6),
    borderTopRightRadius: radius(6),
    borderWidth: 1,
    borderColor: Theme.border,
    paddingHorizontal: spacing(5),
    paddingTop: spacing(2),
    paddingBottom: spacing(8),
  },
  handle: {
    alignSelf: 'center',
    width: 44,
    height: 4,
    borderRadius: 2,
    backgroundColor: Theme.border,
    marginBottom: spacing(3),
  },
  brand: {
    alignItems: 'center',
    marginBottom: spacing(4),
  },
  logo: {
    width: 280,
    height: 108,
  },
  panel: {
    backgroundColor: Theme.panel,
    borderRadius: radius(5),
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    paddingHorizontal: spacing(4),
    paddingVertical: spacing(3),
  },
  sectionLabel: {
    color: Theme.muted,
    fontSize: fontSize(11),
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 2,
    marginTop: spacing(3),
    marginBottom: spacing(1.5),
  },
  segmentRow: {
    flexDirection: 'row',
    gap: spacing(1),
    backgroundColor: Theme.bg,
    borderRadius: radius(3),
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    padding: spacing(1),
  },
  segment: {
    flex: 1,
    paddingVertical: spacing(2.5),
    borderRadius: radius(2),
    alignItems: 'center',
  },
  segmentActive: {
    backgroundColor: 'rgba(34, 211, 238, 0.18)',
    shadowColor: Theme.cyan,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
  },
  segmentText: {
    color: Theme.muted,
    fontSize: fontSize(13),
    fontWeight: '600',
  },
  segmentTextActive: {
    color: Theme.cyan,
    fontWeight: '800',
  },
  sizeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing(3),
  },
  sliderTouch: {
    flex: 1,
    justifyContent: 'center',
    paddingVertical: spacing(2),
  },
  sliderTrack: {
    position: 'relative',
    height: TRACK_H,
    borderRadius: TRACK_H / 2,
    backgroundColor: Theme.bg,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
  },
  sliderFill: {
    position: 'absolute',
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
    borderRadius: TRACK_H / 2,
    backgroundColor: 'rgba(34, 211, 238, 0.5)',
    transformOrigin: 'left',
  },
  sliderThumb: {
    position: 'absolute',
    left: 0,
    top: (TRACK_H - THUMB) / 2,
    width: THUMB,
    height: THUMB,
    borderRadius: THUMB / 2,
    backgroundColor: Theme.cyan,
    shadowColor: Theme.cyan,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.6,
    shadowRadius: 6,
  },
  sizeValue: {
    minWidth: spacing(12),
    textAlign: 'right',
    color: Theme.cyan,
    fontSize: fontSize(13),
    fontWeight: '800',
  },
  startBtn: {
    marginTop: spacing(5),
    paddingVertical: spacing(3.5),
    borderRadius: radius(4),
    borderWidth: 2,
    borderColor: Theme.cyan,
    alignItems: 'center',
    shadowColor: Theme.cyan,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
  },
  startBtnPressed: {
    backgroundColor: Theme.cyan,
  },
  startBtnText: {
    color: Theme.cyan,
    fontSize: fontSize(16),
    fontWeight: '800',
    letterSpacing: 3,
    textTransform: 'uppercase',
  },
  startBtnTextPressed: {
    color: Theme.bg,
  },
  howToBtn: {
    marginTop: spacing(3),
    paddingVertical: spacing(1.5),
    alignItems: 'center',
  },
  howToBtnPressed: {
    opacity: 0.6,
  },
  howToText: {
    color: Theme.muted,
    fontSize: fontSize(12),
    fontWeight: '700',
    letterSpacing: 2,
    textTransform: 'uppercase',
  },
  howToTextPressed: {
    color: Theme.cyan,
  },
})