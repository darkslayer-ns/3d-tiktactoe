/**
 * Builds the "what the model knows about you" snapshot for the current board.
 *
 * Pure (no RN imports) so jest can test it. Reads the live AI seams — engine,
 * predictor, mover, profiles, stats — and flattens them into a displayable
 * snapshot. Nothing here mutates game state.
 */

import type { Board } from '../game/board'
import { P1, type Cell } from '../game/types'
import type { AiDecision, Difficulty, EvalEngine } from '../ai/types'
import type { OpponentPredictor } from '../ai/predictor'
import type { LookaheadMover } from '../ai/mover'
import type { PerceptionProfile, PlayerProfile } from '../ai/profile'
import type { GameStats } from '../ai/stats'
import { argmax, sigmoid } from '../ai/math'
import type { NativeAIState } from '../native/TfmEngine'

export interface PredictionRow {
  index: number
  prob: number
}

export interface AffinityRow {
  index: number
  weight: number
}

export interface LineBar {
  label: string
  value: number
  weight: number
}

export interface ModelKnowledgeSnapshot {
  winProbHuman: number | null
  winProbAi: number | null
  bestMoveIndex: number | null
  /** Every empty cell, ranked by the model's predicted probability. */
  predictions: PredictionRow[]
  /** The human's remembered cells, ranked by affinity weight. */
  affinity: AffinityRow[]
  aggression: number
  perception: number
  perceptionBars: LineBar[]
  stats: GameStats
  winRate: number
  adaptive: number
  lastDecision: AiDecision | null
  difficulty: Difficulty
}

export interface KnowledgeInput {
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
  nativeState?: NativeAIState | null
}

const PERCEPTION_BARS: Array<[string, keyof Pick<PerceptionProfile, 'axis' | 'face' | 'space'>]> = [
  ['AXIS', 'axis'],
  ['FACE', 'face'],
  ['SPACE', 'space'],
]

function nativePerceptionValue(values: number[], key: 'axis' | 'face' | 'space'): number {
  return values[key === 'axis' ? 0 : key === 'face' ? 1 : 2] ?? 0
}

export function buildKnowledgeSnapshot(input: KnowledgeInput): ModelKnowledgeSnapshot {
  const { engine, board, humanSide, difficulty, predictor, mover, profile, perception, stats, adaptive, nativeState } = input

  let winProbHuman: number | null = null
  let winProbAi: number | null = null
  let bestMoveIndex: number | null = null

  if (engine && board) {
    try {
      const humanEval = engine.evalPosition(board.cells, humanSide)
      winProbHuman = sigmoid(humanEval.value)
      if (board.emptyCount() > 0) {
        const aiSide: Cell = humanSide === P1 ? 2 : 1
        const aiEval = engine.evalPosition(board.cells, aiSide)
        winProbAi = sigmoid(aiEval.value)
        const policy = aiEval.policy
        let best = -1
        let bestV = -Infinity
        for (let i = 0; i < policy.length; i++) {
          if (board.cells[i] === 0 && policy[i] > bestV) {
            bestV = policy[i]
            best = i
          }
        }
        bestMoveIndex = best >= 0 ? best : argmax(humanEval.policy)
      }
    } catch {
      // engine hiccup — leave eval fields null
    }
  }

  let predictions: PredictionRow[] = []
  if (predictor && predictor.model && board) {
    try {
      const topK = Math.min(64, board.emptyCount())
      const dist = predictor.predictDistribution(humanSide, 1.0, topK, board)
      predictions = dist.map(([index, prob]) => ({ index, prob }))
    } catch {
      // predictor requires the engine — fall back to empty
    }
  }

  let affinity: AffinityRow[] = []
  if (nativeState) {
    for (let i = 0; i + 2 < nativeState.affinity.length; i += 3) {
      if (nativeState.affinity[i] === humanSide && nativeState.affinity[i + 2] > 0) {
        affinity.push({ index: nativeState.affinity[i + 1], weight: nativeState.affinity[i + 2] })
      }
    }
    affinity.sort((a, b) => b.weight - a.weight)
  } else if (predictor) {
    const row = predictor.affinity.get(humanSide)
    if (row) {
      affinity = Array.from(row.entries())
        .map(([index, weight]) => ({ index, weight }))
        .filter((r) => r.weight > 0)
        .sort((a, b) => b.weight - a.weight)
    }
  }

  const nativePerception = nativeState?.perception ?? []
  const nativeStats = nativeState?.stats ?? []
  const nativeAggression = nativeState?.aggression ?? 0
  const nativePerceptionScore = (() => {
    const axis = nativePerception[0] ?? 0
    const face = nativePerception[1] ?? 0
    const space = nativePerception[2] ?? 0
    const total = axis + face + space
    return total < 1 ? 0.5 : (space + 0.5 * face) / total
  })()
  const nativeStatsValue: GameStats = nativeState
    ? { wins: nativeStats[0] ?? 0, losses: nativeStats[1] ?? 0, draws: nativeStats[2] ?? 0 }
    : stats
  const aggression = nativeState ? nativeAggression : profile ? profile.aggression() : 0
  const perceptionBars: LineBar[] = PERCEPTION_BARS.map(([label, key]) => ({
    label,
    value: nativeState ? nativePerceptionValue(nativePerception, key) : perception ? perception[key] : 0,
    weight: nativeState ? nativePerceptionScore : perception ? perception.score() : 0,
  }))

  return {
    winProbHuman,
    winProbAi,
    bestMoveIndex,
    predictions,
    affinity,
    aggression,
    perception: nativeState ? nativePerceptionScore : perception ? perception.score() : 0.5,
    perceptionBars,
    stats: nativeStatsValue,
    winRate: nativeStatsValue.wins + nativeStatsValue.losses >= 2 ? nativeStatsValue.wins / (nativeStatsValue.wins + nativeStatsValue.losses) : 0.5,
    adaptive: nativeState ? nativeState.adaptive : adaptive,
    lastDecision: mover ? mover.lastDecision : null,
    difficulty,
  }
}
