export const API = ''

export type Cell = 0 | 1 | 2

export interface GameSnapshot {
  id: string
  size: number
  mode: 'pve' | 'ave'
  difficulty: string
  x_agent: string
  o_agent: string
  cells: Cell[]
  current_player: number
  winner: number
  winning_line: [number, number, number][] | null
  over: boolean
  thinking: boolean
}

export interface GameEvent {
  type: 'started' | 'move' | 'thinking' | 'gameover' | 'stopped'
  player?: number
  index?: number
  coord?: [number, number, number]
  game?: GameSnapshot
}

export async function createGame(cfg: {
  size: number
  mode: 'pve' | 'ave'
  difficulty: string
  x_agent?: string
  o_agent?: string
  ai_delay?: number
  human_side?: number
}): Promise<string> {
  const res = await fetch(`${API}/games`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail ?? 'could not create game')
  }
  const data = await res.json()
  return data.game_id
}

export async function getGame(id: string): Promise<GameSnapshot> {
  const res = await fetch(`${API}/games/${id}`)
  return res.json()
}

export async function makeMove(id: string, index: number): Promise<GameSnapshot> {
  const res = await fetch(`${API}/games/${id}/move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ index }),
  })
  const data = await res.json()
  return data.game
}

export interface AiDecision {
  player: number
  chosen: number
  coord: [number, number, number]
  kind: string
  value: number | null
  depth: number
  scored: [number, number][]
  line: { player: number; index: number; coord: [number, number, number] }[]
}

export interface AdminGame extends GameSnapshot {
  move_count: number
  moves: { player: number; index: number; coord: [number, number, number] }[]
  x_decision: AiDecision | null
  o_decision: AiDecision | null
}

export async function listGames(): Promise<AdminGame[]> {
  const res = await fetch(`${API}/games`)
  const data = await res.json()
  return data.games ?? []
}

export function gameSocket(id: string, onEvent: (e: GameEvent) => void): WebSocket {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${window.location.host}/games/${id}/stream`)
  ws.onmessage = (msg) => onEvent(JSON.parse(msg.data))
  ws.onopen = () => ws.send(JSON.stringify({ type: 'start' }))
  return ws
}