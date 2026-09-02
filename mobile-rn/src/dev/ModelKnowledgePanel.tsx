/**
 * "What the AI knows about you" — internal-only debug panel.
 *
 * A Neon-Cube-styled bottom sheet showing, for the CURRENT board, what the
 * on-device model has learned about the human: predicted replies, remembered
 * cell affinity, style profile, and recent form. Compiled only into internal
 * builds (see ../dev/internalBuild).
 */

import { useEffect, useMemo, useRef } from 'react'
import {
  Animated,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native'
import { Theme, fontSize, radius, spacing } from '../ui/theme'
import type { Board } from '../game/board'
import type { Cell } from '../game/types'
import type { Difficulty, EvalEngine } from '../ai/types'
import type { OpponentPredictor } from '../ai/predictor'
import type { LookaheadMover } from '../ai/mover'
import type { PerceptionProfile, PlayerProfile } from '../ai/profile'
import type { GameStats } from '../ai/stats'
import { buildKnowledgeSnapshot, type ModelKnowledgeSnapshot } from './knowledge'

interface ModelKnowledgePanelProps {
  visible: boolean
  onClose: () => void
  engine: EvalEngine | null
  board: Board | null
  humanSide: Cell
  difficulty: Difficulty
  predictor: OpponentPredictor | null
  mover: LookaheadMover | null
  profile: PlayerProfile | null
  perception: PerceptionProfile | null
  stats: GameStats
  adaptive: number
}

/** lerp two hex colors; t=0 -> a, t=1 -> b */
function lerpColor(a: string, b: string, t: number): string {
  const pa = parseInt(a.slice(1), 16)
  const pb = parseInt(b.slice(1), 16)
  const ch = (sa: number, sb: number) => {
    const x = Math.round(sa + (sb - sa) * Math.max(0, Math.min(1, t)))
    return x.toString(16).padStart(2, '0')
  }
  return `#${ch((pa >> 16) & 255, (pb >> 16) & 255)}${ch((pa >> 8) & 255, (pb >> 8) & 255)}${ch(pa & 255, pb & 255)}`
}

const fmtPct = (v: number): string => `${Math.round(v * 100)}%`
const fmtW = (v: number): string => (Number.isFinite(v) ? v.toFixed(2) : '0.00')

export function ModelKnowledgePanel(props: ModelKnowledgePanelProps) {
  const { visible, onClose, engine, board, humanSide, difficulty, predictor, mover, profile, perception, stats, adaptive } = props
  const snap = useMemo<ModelKnowledgeSnapshot | null>(
    () =>
      buildKnowledgeSnapshot({
        engine,
        board,
        humanSide,
        difficulty,
        predictor,
        mover,
        profile,
        perception,
        stats,
        adaptive,
      }),
    [engine, board, humanSide, difficulty, predictor, mover, profile, perception, stats, adaptive, visible],
  )

  const translateY = useRef(new Animated.Value(900)).current

  useEffect(() => {
    if (visible) {
      translateY.setValue(900)
      Animated.spring(translateY, { toValue: 0, damping: 24, stiffness: 240, useNativeDriver: true }).start()
    }
  }, [visible, translateY])

  const n = board?.n ?? 3
  const tile = n >= 5 ? 16 : n === 4 ? 20 : 24
  const probByCell = useMemo(() => {
    const map = new Map<number, number>()
    for (const p of snap?.predictions ?? []) map.set(p.index, p.prob)
    return map
  }, [snap])

  if (snap === null) return null

  return (
    <Modal visible={visible} transparent animationType="none" onRequestClose={onClose}>
      <View style={styles.backdrop}>
        <Animated.View style={[styles.sheet, { transform: [{ translateY }] }]}>
          <View style={styles.handle} />

          <View style={styles.header}>
            <View>
              <Text style={styles.title}>MODEL MEMORY</Text>
              <Text style={styles.subtitle}>What the AI knows about you</Text>
            </View>
            <View style={styles.headerRight}>
              <View style={styles.badge}>
                <Text style={styles.badgeText}>INTERNAL BUILD</Text>
              </View>
              <Pressable onPress={onClose} style={styles.closeBtn} hitSlop={8}>
                <Text style={styles.closeText}>✕</Text>
              </Pressable>
            </View>
          </View>

          <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.body}>
            {/* Current position */}
            <View style={styles.card}>
              <Text style={styles.cardLabel}>CURRENT POSITION</Text>
              <View style={styles.row}>
                <View style={styles.metric}>
                  <Text style={styles.metricValueCyan}>{snap.winProbHuman == null ? '–' : snap.winProbHuman.toFixed(3)}</Text>
                  <Text style={styles.metricLabel}>YOUR WIN PROB</Text>
                </View>
                <View style={styles.metric}>
                  <Text style={styles.metricValue}>
                    {snap.bestMoveIndex == null ? '–' : String(snap.bestMoveIndex).padStart(2, '0')}
                  </Text>
                  <Text style={styles.metricLabel}>MODEL BEST MOVE</Text>
                </View>
                <View style={styles.metric}>
                  <Text style={styles.metricValue}>{snap.difficulty.toUpperCase()}</Text>
                  <Text style={styles.metricLabel}>DIFFICULTY</Text>
                </View>
              </View>
              <View style={styles.barTrack}>
                <View
                  style={[
                    styles.barFill,
                    { width: `${Math.max(0, Math.min(100, (snap.winProbHuman ?? 0.5) * 100))}%` },
                  ]}
                />
              </View>
              {snap.lastDecision != null && (
                <Text style={styles.debugLine}>
                  last: cell {String(snap.lastDecision.chosen).padStart(2, '0')} · value{' '}
                  {snap.lastDecision.value == null ? '–' : snap.lastDecision.value.toFixed(3)} · depth{' '}
                  {snap.lastDecision.depth}
                </Text>
              )}
            </View>

            {/* Where you're predicted to play */}
            <View style={styles.card}>
              <Text style={styles.cardLabel}>WHERE YOU'RE PREDICTED TO PLAY</Text>
              {snap.predictions.length === 0 ? (
                <Text style={styles.empty}>no legal moves</Text>
              ) : (
                <>
                  <View style={styles.grid}>
                    {Array.from({ length: n ** 3 }, (_, i) => i).map((i) => {
                      const occupied = board ? board.cells[i] : 0
                      const prob = probByCell.get(i)
                      const bg = occupied !== 0 ? 'transparent' : lerpColor('#f472b6', '#a3e635', prob ?? 0)
                      return (
                        <View
                          key={i}
                          style={[
                            styles.tile,
                            { width: tile, height: tile, backgroundColor: bg, borderColor: occupied !== 0 ? (occupied === humanSide ? Theme.cyan : Theme.pink) : 'rgba(255,255,255,0.08)' },
                          ]}
                        >
                          <Text
                            style={[
                              styles.tileText,
                              { color: occupied !== 0 ? Theme.bg : 'rgba(2,6,23,0.85)', fontSize: n >= 5 ? 7 : 9 },
                            ]}
                          >
                            {occupied !== 0 ? (occupied === humanSide ? 'X' : 'O') : prob == null ? '' : fmtPct(prob)}
                          </Text>
                        </View>
                      )
                    })}
                  </View>
                  {snap.predictions.slice(0, 5).map((p, idx) => (
                    <View key={p.index} style={styles.rankRow}>
                      <Text style={styles.rankIndex}>
                        {idx + 1}. cell {String(p.index).padStart(2, '0')}
                      </Text>
                      <View style={styles.rankTrack}>
                        <View style={[styles.rankFill, { width: `${Math.max(4, p.prob * 100)}%`, backgroundColor: lerpColor('#f472b6', '#a3e635', p.prob) }]} />
                      </View>
                      <Text style={styles.rankPct}>{fmtPct(p.prob)}</Text>
                    </View>
                  ))}
                </>
              )}
            </View>

            {/* Style profile */}
            <View style={styles.card}>
              <Text style={styles.cardLabel}>STYLE PROFILE</Text>
              <View style={styles.meterWrap}>
                <View style={styles.meterTrack}>
                  <Text style={styles.meterEnd}>DEFENDER</Text>
                  <View
                    style={[
                      styles.meterMarker,
                      { left: `${((snap.aggression + 1) / 2) * 100}%`, transform: [{ translateX: -5 }] },
                    ]}
                  />
                  <Text style={styles.meterEnd}>ATTACKER</Text>
                </View>
                <Text style={styles.meterValue}>
                  AGGRESSION {snap.aggression >= 0 ? '+' : ''}
                  {snap.aggression.toFixed(2)}
                </Text>
              </View>
              <View style={styles.percRow}>
                {snap.perceptionBars.map((bar) => {
                  const max = Math.max(1, ...snap.perceptionBars.map((b) => b.value))
                  return (
                    <View key={bar.label} style={styles.percCol}>
                      <View style={styles.percTrack}>
                        <View
                          style={[
                            styles.percFill,
                            { width: `${(bar.value / max) * 100}%`, backgroundColor: Theme.cyan },
                          ]}
                        />
                      </View>
                      <Text style={styles.percLabel}>{bar.label}</Text>
                    </View>
                  )
                })}
                <Text style={styles.percScore}>PERCEPTION {snap.perception.toFixed(2)}</Text>
              </View>
            </View>

            {/* Recent form */}
            <View style={styles.card}>
              <Text style={styles.cardLabel}>RECENT FORM</Text>
              <View style={styles.formRow}>
                <View style={styles.metric}>
                  <Text style={styles.metricValueCyan}>{fmtPct(snap.winRate)}</Text>
                  <Text style={styles.metricLabel}>WIN RATE</Text>
                </View>
                <View style={styles.metric}>
                  <Text
                    style={[
                      styles.metricValue,
                      { color: snap.adaptive > 0.05 ? Theme.pink : snap.adaptive < -0.05 ? Theme.cyan : Theme.text },
                    ]}
                  >
                    {snap.adaptive > 0.05 ? 'EASING' : snap.adaptive < -0.05 ? 'HARDENING' : 'BALANCED'}
                  </Text>
                  <Text style={styles.metricLabel}>ADAPTIVE LEVEL</Text>
                </View>
              </View>
              <View style={styles.chipRow}>
                <View style={[styles.chip, { borderColor: Theme.cyan }]}>
                  <Text style={[styles.chipText, { color: Theme.cyan }]}>W {Math.round(snap.stats.wins)}</Text>
                </View>
                <View style={[styles.chip, { borderColor: Theme.pink }]}>
                  <Text style={[styles.chipText, { color: Theme.pink }]}>L {Math.round(snap.stats.losses)}</Text>
                </View>
                <View style={[styles.chip, { borderColor: Theme.border }]}>
                  <Text style={[styles.chipText, { color: Theme.muted }]}>D {Math.round(snap.stats.draws)}</Text>
                </View>
              </View>
            </View>

            {/* Cell affinity */}
            <View style={styles.card}>
              <Text style={styles.cardLabel}>CELL AFFINITY (WHAT IT REMEMBERS)</Text>
              {snap.affinity.length === 0 ? (
                <Text style={styles.empty}>no memory yet — play some games</Text>
              ) : (
                snap.affinity.slice(0, 8).map((a) => {
                  const max = snap.affinity[0]?.weight ?? 1
                  return (
                    <View key={a.index} style={styles.affRow}>
                      <Text style={styles.affIndex}>CELL {String(a.index).padStart(2, '0')}</Text>
                      <View style={styles.affTrack}>
                        <View
                          style={[
                            styles.affFill,
                            { width: `${Math.max(4, (a.weight / max) * 100)}%`, backgroundColor: Theme.purple },
                          ]}
                        />
                      </View>
                      <Text style={styles.affWeight}>{fmtW(a.weight)}</Text>
                    </View>
                  )
                })
              )}
            </View>

            <Text style={styles.footer}>INTERNAL TOOLING ONLY — NOT SHIPPED</Text>
          </ScrollView>
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
    maxHeight: '88%',
    backgroundColor: Theme.panel,
    borderTopLeftRadius: radius(6),
    borderTopRightRadius: radius(6),
    borderWidth: 1,
    borderColor: Theme.border,
  },
  handle: {
    alignSelf: 'center',
    width: 44,
    height: 4,
    borderRadius: 2,
    backgroundColor: Theme.border,
    marginTop: spacing(2),
    marginBottom: spacing(2),
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing(5),
    paddingBottom: spacing(3),
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.06)',
  },
  title: {
    color: Theme.cyan,
    fontSize: fontSize(15),
    fontWeight: '800',
    letterSpacing: 2,
  },
  subtitle: {
    color: Theme.muted,
    fontSize: fontSize(12),
    marginTop: 2,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing(2),
  },
  badge: {
    paddingVertical: spacing(1),
    paddingHorizontal: spacing(2),
    borderRadius: radius(2),
    borderWidth: 1,
    borderColor: 'rgba(254, 240, 138, 0.5)',
    backgroundColor: 'rgba(254, 240, 138, 0.12)',
  },
  badgeText: {
    color: Theme.yellow,
    fontSize: fontSize(9),
    fontWeight: '800',
    letterSpacing: 1,
  },
  closeBtn: {
    width: 26,
    height: 26,
    borderRadius: 13,
    borderWidth: 1,
    borderColor: Theme.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  closeText: {
    color: Theme.muted,
    fontSize: fontSize(12),
  },
  body: {
    paddingHorizontal: spacing(4),
    paddingTop: spacing(3),
    paddingBottom: spacing(8),
    gap: spacing(3),
  },
  card: {
    backgroundColor: 'rgba(15, 23, 42, 0.85)',
    borderRadius: radius(3),
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    borderTopColor: 'rgba(255, 255, 255, 0.14)',
    padding: spacing(3),
    gap: spacing(2.5),
  },
  cardLabel: {
    color: Theme.muted,
    fontSize: fontSize(10),
    fontWeight: '800',
    letterSpacing: 2,
  },
  row: {
    flexDirection: 'row',
    gap: spacing(3),
  },
  metric: {
    flex: 1,
    gap: 2,
  },
  metricValue: {
    color: Theme.text,
    fontSize: fontSize(18),
    fontWeight: '800',
  },
  metricValueCyan: {
    color: Theme.cyan,
    fontSize: fontSize(22),
    fontWeight: '800',
  },
  metricLabel: {
    color: Theme.muted,
    fontSize: fontSize(9),
    fontWeight: '700',
    letterSpacing: 1,
  },
  barTrack: {
    height: 6,
    borderRadius: 3,
    backgroundColor: Theme.bg,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    backgroundColor: Theme.cyan,
  },
  debugLine: {
    color: Theme.subtle,
    fontSize: fontSize(10),
  },
  empty: {
    color: Theme.muted,
    fontSize: fontSize(12),
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 3,
  },
  tile: {
    borderRadius: radius(1),
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tileText: {
    fontWeight: '800',
  },
  rankRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing(2),
  },
  rankIndex: {
    width: 74,
    color: Theme.text,
    fontSize: fontSize(11),
    fontWeight: '700',
  },
  rankTrack: {
    flex: 1,
    height: 6,
    borderRadius: 3,
    backgroundColor: Theme.bg,
    overflow: 'hidden',
  },
  rankFill: {
    height: '100%',
  },
  rankPct: {
    width: 40,
    textAlign: 'right',
    color: Theme.subtle,
    fontSize: fontSize(11),
  },
  meterWrap: {
    gap: spacing(1.5),
  },
  meterTrack: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: spacing(5),
    borderRadius: radius(2),
    backgroundColor: Theme.bg,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
    position: 'relative',
    overflow: 'hidden',
  },
  meterEnd: {
    color: Theme.muted,
    fontSize: fontSize(9),
    fontWeight: '700',
    letterSpacing: 1,
    paddingHorizontal: spacing(1.5),
    zIndex: 1,
  },
  meterMarker: {
    position: 'absolute',
    top: (spacing(5) - 10) / 2,
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: Theme.cyan,
    shadowColor: Theme.cyan,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 4,
  },
  meterValue: {
    color: Theme.cyan,
    fontSize: fontSize(12),
    fontWeight: '800',
    textAlign: 'center',
  },
  percRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: spacing(2),
  },
  percCol: {
    flex: 1,
    gap: 2,
  },
  percTrack: {
    height: 6,
    borderRadius: 3,
    backgroundColor: Theme.bg,
    overflow: 'hidden',
  },
  percFill: {
    height: '100%',
  },
  percLabel: {
    color: Theme.muted,
    fontSize: fontSize(8),
    fontWeight: '700',
    letterSpacing: 1,
    textAlign: 'center',
  },
  percScore: {
    color: Theme.subtle,
    fontSize: fontSize(10),
  },
  formRow: {
    flexDirection: 'row',
    gap: spacing(3),
  },
  chipRow: {
    flexDirection: 'row',
    gap: spacing(2),
  },
  chip: {
    paddingVertical: spacing(1),
    paddingHorizontal: spacing(2),
    borderRadius: radius(2),
    borderWidth: 1,
  },
  chipText: {
    fontSize: fontSize(11),
    fontWeight: '800',
  },
  affRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing(2),
  },
  affIndex: {
    width: 64,
    color: Theme.text,
    fontSize: fontSize(11),
    fontWeight: '700',
  },
  affTrack: {
    flex: 1,
    height: 6,
    borderRadius: 3,
    backgroundColor: Theme.bg,
    overflow: 'hidden',
  },
  affFill: {
    height: '100%',
  },
  affWeight: {
    width: 40,
    textAlign: 'right',
    color: Theme.purple,
    fontSize: fontSize(11),
  },
  footer: {
    color: Theme.muted,
    fontSize: fontSize(9),
    fontWeight: '700',
    letterSpacing: 2,
    textAlign: 'center',
    marginTop: spacing(1),
  },
})