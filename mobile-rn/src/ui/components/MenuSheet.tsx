/**
 * Bottom sheet menu for Neon Cube: pick your side, difficulty and board size,
 * then Start. Slides up from the bottom over a dimmed backdrop.
 */

import { useEffect, useRef, useState } from 'react'
import {
  Animated,
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
  { value: 1, label: 'X · first' },
  { value: 2, label: 'O · second' },
]

const DIFFICULTY_OPTIONS: Option<Difficulty>[] = [
  { value: 'easy', label: 'Easy' },
  { value: 'medium', label: 'Medium' },
  { value: 'hard', label: 'Hard' },
]

const SIZE_OPTIONS: Option<number>[] = [
  { value: 3, label: '3×3×3' },
  { value: 4, label: '4×4×4' },
  { value: 5, label: '5×5×5' },
]

export function MenuSheet({ visible, onStart }: MenuSheetProps) {
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
          <Text style={styles.title}>Neon Cube</Text>
          <Text style={styles.subtitle}>3D tic-tac-toe against the on-device model</Text>

          <Text style={styles.sectionLabel}>You play</Text>
          <Segmented options={SIDE_OPTIONS} value={humanSide} onChange={setHumanSide} />

          <Text style={styles.sectionLabel}>Difficulty</Text>
          <Segmented options={DIFFICULTY_OPTIONS} value={difficulty} onChange={setDifficulty} />

          <Text style={styles.sectionLabel}>Board size</Text>
          <Segmented options={SIZE_OPTIONS} value={size} onChange={setSize} />

          <Pressable
            onPress={() => onStart({ humanSide, difficulty, size })}
            style={styles.startBtn}
          >
            <Text style={styles.startBtnText}>Start game</Text>
          </Pressable>
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
    marginBottom: spacing(4),
  },
  title: {
    color: Theme.cyan,
    fontSize: fontSize(26),
    fontWeight: '800',
    letterSpacing: 2,
    textAlign: 'center',
  },
  subtitle: {
    color: Theme.muted,
    fontSize: fontSize(13),
    textAlign: 'center',
    marginTop: spacing(1),
    marginBottom: spacing(4),
  },
  sectionLabel: {
    color: Theme.muted,
    fontSize: fontSize(11),
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginTop: spacing(3),
    marginBottom: spacing(1.5),
  },
  segmentRow: {
    flexDirection: 'row',
    gap: spacing(1.5),
  },
  segment: {
    flex: 1,
    paddingVertical: spacing(2.5),
    paddingHorizontal: spacing(2),
    borderRadius: radius(2),
    borderWidth: 1,
    borderColor: Theme.border,
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
    alignItems: 'center',
  },
  segmentActive: {
    borderColor: Theme.cyan,
    backgroundColor: Theme.cyan,
  },
  segmentText: {
    color: Theme.text,
    fontSize: fontSize(13),
    fontWeight: '600',
  },
  segmentTextActive: {
    color: Theme.bg,
    fontWeight: '800',
  },
  startBtn: {
    marginTop: spacing(6),
    paddingVertical: spacing(3),
    borderRadius: radius(2),
    backgroundColor: Theme.cyan,
    alignItems: 'center',
  },
  startBtnText: {
    color: Theme.bg,
    fontSize: fontSize(16),
    fontWeight: '800',
    letterSpacing: 1,
  },
})