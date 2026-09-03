/**
 * Native 3D board for Neon Cube — a port of the web Board3D
 * (frontend/src/components/Board3D.tsx) to @react-three/fiber/native
 * (expo-gl provides the GL context; the native <Canvas> creates it for us).
 *
 * Visuals match the web build:
 *  - X marks = two crossed cyan emissive bars
 *  - O marks = pink emissive torus ring
 *  - empty cells = dark translucent slots that pulse cyan on the selected
 *    axis and dim off-axis
 *  - a glowing purple cylinder draws the winning line
 *  - a yellow translucent box marks the pending (selected) cell
 *  - wireframe shell + two colored point lights + ambient light
 *
 * Touch controls (no drei / no DOM OrbitControls):
 *  - ONE-FINGER DRAG: orbit. Implemented with RN's responder *capture* phase on
 *    the wrapper View (onMoveShouldSetResponderCapture). RNGH gestures are NOT
 *    used: on Android the expo-gl GLSurfaceView consumes touches before RNGH's
 *    native handlers ever see them. The responder capture runs in the same JS
 *    responder system R3F uses (its PanResponder on the GLView) and can steal
 *    drags/pinches while letting taps fall through to the canvas.
 *  - TWO-FINGER PINCH: zoom, tracked manually from nativeEvent.touches.
 *  - CELL TAPS: R3F-native mesh events (onPointerDown + onClick). The native
 *    Canvas only fires `onClick` when the whole gesture moved < 20px; we add
 *    a stricter 10px threshold on top for web parity by comparing the pointer
 *    down position recorded in onPointerDown.
 *
 * R3F-native gotchas coded around:
 *  - `onClick` needs the PanResponder that the native Canvas installs. Because
 *    we only capture the responder once the gesture moves > 10px (or a second
 *    finger lands), a tap never triggers a capture and R3F sees the full tap; a
 *    drag cancels R3F's pointer (no click is fired), so drags never place a mark.
 *  - Non-interactive meshes get `raycast={() => null}` so they stay visible
 *    but are never hit by the raycaster (same trick as the web build).
 *  - The native Canvas manages dpr/antialias itself; we pass `gl.antialias`
 *    which it maps to GLView `msaaSamples`.
 */

import { memo, useMemo, useRef, useCallback, useEffect, type RefObject } from 'react'
import { View, StyleSheet, type GestureResponderEvent } from 'react-native'
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber/native'
import * as THREE from 'three'
import { Theme } from '../theme'
import { cellCoord, type Cell, type Coord } from '../../game/types'

/** Spread cells outward so inner cells of 4×4×4 / 5×5×5 are reachable. */
const EXPLODE = 0.4

/** Side length of each cell's box (same for every board size). */
const SLOT_SIZE = 0.82 * (1 + EXPLODE) * 0.8
/** Thickness of each slot's glowing edge bar (GL linewidth is ignored on Metal,
 * so the "cube lines" are thin boxes instead of line segments). */
const EDGE_THICKNESS = 0.038

/** The 12 edges of a slot box, as (position, box size) bar transforms. */
const slotEdges: Array<{ pos: [number, number, number]; size: [number, number, number] }> = (() => {
  const h = SLOT_SIZE / 2
  const t = EDGE_THICKNESS
  const bars: Array<{ pos: [number, number, number]; size: [number, number, number] }> = []
  for (const sy of [-1, 1]) {
    for (const sz of [-1, 1]) bars.push({ pos: [0, sy * h, sz * h], size: [SLOT_SIZE, t, t] })
  }
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) bars.push({ pos: [sx * h, 0, sz * h], size: [t, SLOT_SIZE, t] })
  }
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) bars.push({ pos: [sx * h, sy * h, 0], size: [t, t, SLOT_SIZE] })
  }
  return bars
})()

/** Camera orbit limits (web OrbitControls used 3..14). */
const MIN_DISTANCE = 4
const MAX_DISTANCE = 40
const MIN_PHI = 0.15
const MAX_PHI = Math.PI - 0.15

const INITIAL_CAMERA_POSITION: [number, number, number] = [5, 4.5, 5.5]

/** Camera distance that fits the whole n×n×n cube (with the explode spread).
 * Uses the REAL rendered half-extent (outermost cell center + half a cell box)
 * so every board size fills the screen consistently. */
function fitDistance(size: number, aspect = 0.5, vFovDeg = 45): number {
  const halfExtent = ((size - 1) / 2) * (1 + EXPLODE) + SLOT_SIZE / 2
  const radius = halfExtent * Math.sqrt(3)
  const vHalf = (vFovDeg * Math.PI) / 360
  const hHalf = Math.atan(Math.tan(vHalf) * Math.max(0.1, aspect))
  const limitingHalf = Math.min(hHalf, vHalf)
  const d = radius / Math.tan(limitingHalf)
  return clamp(d * 1.1, MIN_DISTANCE, MAX_DISTANCE)
}

/** Spherical target the camera eases toward (theta/phi around cube center). */
export interface OrbitTarget {
  theta: number
  phi: number
  distance: number
}

export interface Board3DProps {
  size: number
  cells: Cell[]
  onCellClick: (index: number) => void
  pendingIndex: number | null
  winningLine: Coord[] | null
  lastAiMove: number | null
  /** Cell recommended by the Hint button — pulses cyan. */
  hintIndex: number | null
  /** true while the opponent is computing — drives the cube's pulse animation */
  thinking: boolean
  /** increments on every game start — triggers the cube "teleports in" pop */
  startKey: number
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

/** The X-row and Y-column passing through `center` (a "+" cross in its layer). */
export function axisCross(center: number | null, size: number): Set<number> {
  const set = new Set<number>()
  if (center == null || center < 0) return set
  const [x, y, z] = cellCoord(center, size)
  for (let i = 0; i < size; i++) {
    set.add(i + size * (y + size * z)) // row along X
    set.add(x + size * (i + size * z)) // column along Y
  }
  return set
}

/** Live highlight state for a cell, computed in the frame loop (no re-render). */
function slotFocusState(index: number, size: number, pending: number) {
  if (pending < 0 || pending >= size ** 3) return { onAxis: false, focusing: false }
  const [px, py, pz] = cellCoord(pending, size)
  const [x, y, z] = cellCoord(index, size)
  const onAxis = (x === px && z === pz) || (y === py && z === pz)
  return { onAxis, focusing: true }
}

/** cell index -> world position (same mapping as the web build). */
function cellPosition(index: number, size: number, expl: number): [number, number, number] {
  const [x, y, z] = cellCoord(index, size)
  const off = (size - 1) / 2
  const k = 1 + expl
  return [(x - off) * k, (off - y) * k, (z - off) * k]
}

/** Current spherical of the initial camera angle, fit to `size`. */
function defaultOrbitTarget(size: number): OrbitTarget {
  const s = new THREE.Spherical().setFromVector3(
    new THREE.Vector3(...INITIAL_CAMERA_POSITION),
  )
  return { theta: s.theta, phi: s.phi, distance: fitDistance(size) }
}

interface MarkProps {
  value: Cell
  index: number
  position: [number, number, number]
  size: number
  focusRef: RefObject<{ pending: number; hint: number; thinking: boolean }>
  onPointerDown: (e: ThreeEvent<PointerEvent>) => void
  onClick: (e: ThreeEvent<MouseEvent>, index: number) => void
}

const Mark = memo(function Mark({ value, index, position, size, focusRef, onPointerDown, onClick }: MarkProps) {
  const scale = 0.55
  // Non-interactive marks stay visible but are never hit by the raycaster —
  // legality is enforced in clickCell, so marks are always hittable here.
  const meshProps = {}

  // "Teleport" pop-in on mount: scale 0 → 1 with a springy overshoot plus an
  // emissive flash that settles to `intensity`.
  const g = useRef<THREE.Group>(null)
  const start = useRef(Date.now())
  const mats = useRef<THREE.MeshStandardMaterial[]>([])

  useFrame(() => {
    const grp = g.current
    if (!grp) return
    // Live dim from the frame loop (no React re-render needed on pending change).
    const { focusing, onAxis } = slotFocusState(index, size, focusRef.current.pending)
    const intensity = focusing && !onAxis ? 0.2 : 0.9
    const st = start.current
    const t = st === 0 ? 1 : Math.min(1, (Date.now() - st) / 420)
    if (st !== 0 && t >= 1) start.current = 0
    if (t < 1) {
      const c1 = 1.70158
      const c3 = c1 + 1
      const s = 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2)
      grp.scale.setScalar(Math.max(0.001, s))
    } else {
      grp.scale.setScalar(1)
    }
    const flash = st !== 0 ? 1 + 2.5 * (1 - t) : 1
    for (const m of mats.current) m.emissiveIntensity = intensity * flash
  })

  const refMat = (m: THREE.MeshStandardMaterial | null) => {
    if (m && !mats.current.includes(m)) mats.current.push(m)
  }

  if (value === 1) {
    // an X lying in the face plane: two thin bars crossing at 45°/135°
    const xLen = 1.15 * scale
    return (
      <group
        ref={g}
        position={position}
        onPointerDown={onPointerDown}
        onClick={(e: ThreeEvent<MouseEvent>) => onClick(e, index)}
      >
        <mesh {...meshProps} rotation={[0, 0, Math.PI / 4]}>
          <boxGeometry args={[xLen, 0.22, 0.22]} />
          <meshStandardMaterial
            ref={refMat}
            color="#22d3ee"
            emissive="#22d3ee"
            roughness={0.25}
          />
        </mesh>
        <mesh {...meshProps} rotation={[0, 0, -Math.PI / 4]}>
          <boxGeometry args={[xLen, 0.22, 0.22]} />
          <meshStandardMaterial
            ref={refMat}
            color="#22d3ee"
            emissive="#22d3ee"
            roughness={0.25}
          />
        </mesh>
      </group>
    )
  }
  return (
    <group
      ref={g}
      position={position}
      onPointerDown={onPointerDown}
      onClick={(e: ThreeEvent<MouseEvent>) => onClick(e, index)}
    >
      <mesh {...meshProps}>
        <torusGeometry args={[0.42 * scale, 0.12, 16, 48]} />
        <meshStandardMaterial
          ref={refMat}
          color="#f472b6"
          emissive="#f472b6"
          roughness={0.25}
        />
      </mesh>
    </group>
  )
})

interface SlotProps {
  index: number
  position: [number, number, number]
  size: number
  focusRef: RefObject<{ pending: number; hint: number; thinking: boolean }>
  onPointerDown: (e: ThreeEvent<PointerEvent>) => void
  onClick: (e: ThreeEvent<MouseEvent>, index: number) => void
}

const AnimatedSlot = memo(function AnimatedSlot({ index, position, size, focusRef, onPointerDown, onClick }: SlotProps) {
  const edgeMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: '#0284c7', transparent: true, opacity: 0.3 }),
    [],
  )
  const posV = useMemo(() => new THREE.Vector3(position[0], position[1], position[2]), [position])

  useFrame((state) => {
    const e = edgeMat
    const t = state.clock.elapsedTime
    const { pending, hint, thinking } = focusRef.current
    // Live highlight from the frame loop (no React re-render on pending change).
    const { focusing, onAxis } = slotFocusState(index, size, pending)
    const hintPulse = index === hint
    const dim = focusing && !onAxis
    // Depth fade: cells farther from the camera go fainter (atmospheric).
    const camDist = state.camera.position.length() || 1
    const depth = state.camera.position.distanceTo(posV) / camDist
    const fade = clamp(1.9 - depth, 0.15, 1)
    if (onAxis || hintPulse) {
      e.color.set('#0284c7')
      e.opacity = (0.4 + 0.15 * Math.sin(t * 6)) * fade
    } else if (thinking) {
      // Opponent computing: the borders breathe (no text UI).
      e.color.set('#0284c7')
      e.opacity = (0.2 + 0.15 * (0.5 + 0.5 * Math.sin(t * 4))) * fade
    } else {
      e.color.set(dim ? '#0c4a6e' : '#0284c7')
      e.opacity = (dim ? 0.18 : 0.3) * fade
    }
  })

  return (
    <group position={position}>
      {/* invisible hit target (transparent faces would show triangle seams).
          Always hittable — legality is enforced in clickCell. */}
      <mesh onPointerDown={onPointerDown} onClick={(e) => onClick(e, index)}>
        <boxGeometry args={[SLOT_SIZE, SLOT_SIZE, SLOT_SIZE]} />
        <meshStandardMaterial transparent opacity={0} depthWrite={false} />
      </mesh>
      {/* glowing cyan border: one thin box per edge, sharing one animated material */}
      {slotEdges.map((b, i) => (
        <mesh key={i} position={b.pos} raycast={() => null} material={edgeMat}>
          <boxGeometry args={b.size} />
        </mesh>
      ))}
    </group>
  )
})

function WinBeam({ line, size }: { line: Coord[]; size: number }) {
  const off = (size - 1) / 2
  const k = 1 + EXPLODE
  const { mid, len, quat } = useMemo(() => {
    const a = line[0]
    const b = line[line.length - 1]
    const p1 = new THREE.Vector3((a[0] - off) * k, (off - a[1]) * k, (a[2] - off) * k)
    const p2 = new THREE.Vector3((b[0] - off) * k, (off - b[1]) * k, (b[2] - off) * k)
    const mid = p1.clone().add(p2).multiplyScalar(0.5)
    const len = p1.distanceTo(p2)
    const dir = p2.clone().sub(p1).normalize()
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir)
    // Pass quaternion as a plain [x, y, z, w] tuple: R3F mutates the object's
    // quaternion via fromArray(), whereas passing a THREE.Quaternion instance
    // makes R3F try to assign the read-only property and crash.
    return { mid: mid.toArray() as [number, number, number], len, quat: [q.x, q.y, q.z, q.w] as [number, number, number, number] }
  }, [line, off, k])

  // "Flash" the winning line for the first ~1.6s (until the result overlay
  // appears) — a quick pulsing bloom.
  const mat = useRef<THREE.MeshStandardMaterial>(null)
  const start = useRef(Date.now())
  useFrame(() => {
    const m = mat.current
    if (!m) return
    const t = (Date.now() - start.current) / 1000
    const pulse = 0.5 + 0.5 * Math.abs(Math.sin(t * 7))
    m.emissiveIntensity = 1.5 + 2.5 * pulse
    m.opacity = 0.6 + 0.4 * pulse
  })

  return (
    <group position={mid}>
      <mesh raycast={() => null} quaternion={quat}>
        <cylinderGeometry args={[0.07, 0.07, len + 0.4, 8]} />
        <meshStandardMaterial
          ref={mat}
          color="#ffffff"
          emissive="#e879f9"
          emissiveIntensity={2.5}
          transparent
          opacity={0.95}
        />
      </mesh>
    </group>
  )
}

function PendingHighlight({ position }: { position: [number, number, number] }) {
  const s = 0.9 * (1 + EXPLODE)
  return (
    <mesh raycast={() => null} position={position}>
      <boxGeometry args={[s, s, s]} />
      <meshBasicMaterial color="#fef08a" transparent opacity={0.35} depthWrite={false} />
    </mesh>
  )
}

function LastAiMoveHighlight({ position }: { position: [number, number, number] }) {
  const mat = useRef<THREE.MeshBasicMaterial>(null)
  const s = 0.62 * (1 + EXPLODE)
  useFrame((state) => {
    const m = mat.current
    if (!m) return
    m.opacity = 0.2 + 0.15 * Math.sin(state.clock.elapsedTime * 6)
  })
  return (
    <mesh raycast={() => null} position={position}>
      <boxGeometry args={[s, s, s]} />
      <meshBasicMaterial
        ref={mat}
        color="#ffffff"
        transparent
        opacity={0.4}
        depthWrite={false}
      />
    </mesh>
  )
}

/**
 * "Teleport" pop-in: scales the cube from 0 up with a springy overshoot on
 * every game start. Rendered INSIDE the Canvas so useFrame is valid here.
 */
function CubePop({ startKey, children }: { startKey: number; children: React.ReactNode }) {
  const g = useRef<THREE.Group>(null)
  const startRef = useRef(0)

  useEffect(() => {
    startRef.current = Date.now()
  }, [startKey])

  useFrame(() => {
    const grp = g.current
    if (!grp) return
    const t = Math.min(1, (Date.now() - startRef.current) / 750)
    if (t >= 1) {
      grp.scale.setScalar(1)
      return
    }
    const c1 = 1.70158
    const c3 = c1 + 1
    const s = 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2)
    grp.scale.setScalar(Math.max(0.001, s))
  })

  return <group ref={g}>{children}</group>
}

/**
 * Keeps the cube fitted to the ACTUAL canvas viewport (real width/height, so
 * the correct limiting FOV) until the user manually zooms.
 */
function FitRig({
  size,
  userZoomed,
  target,
}: {
  size: number
  userZoomed: RefObject<boolean>
  target: RefObject<OrbitTarget>
}) {
  const { width, height } = useThree((s) => s.size)
  useEffect(() => {
    if (userZoomed.current) return
    target.current.distance = fitDistance(size, width / Math.max(1, height))
  }, [size, width, height, userZoomed, target])
  return null
}

/**
 * Eases the camera along a spherical orbit toward `target` (fed by the
 * pan/pinch gestures). Runs inside the R3F frame loop.
 */
function OrbitRig({ target }: { target: RefObject<OrbitTarget> }) {
  const spherical = useRef<THREE.Spherical | null>(null)

  useFrame((state, dt) => {
    const cam = state.camera
    const t = target.current
    if (!spherical.current) {
      // Start at the fitted orbit so there's no initial zoom/settle motion.
      spherical.current = new THREE.Spherical(t.distance, t.phi, t.theta)
      cam.position.setFromSphericalCoords(t.distance, t.phi, t.theta)
      cam.lookAt(0, 0, 0)
      return
    }
    const cur = spherical.current
    const damp = 1 - Math.exp(-dt * 8)
    cur.theta += (t.theta - cur.theta) * damp
    cur.phi += (t.phi - cur.phi) * damp
    cur.radius += (t.distance - cur.radius) * damp
    cam.position.setFromSphericalCoords(cur.radius, cur.phi, cur.theta)
    cam.lookAt(0, 0, 0)
  })

  return null
}

interface BoardMeshProps {
  size: number
  cells: Cell[]
  pendingIndex: number | null
  winningLine: Coord[] | null
  lastAiMove: number | null
  focusRef: RefObject<{ pending: number; hint: number; thinking: boolean }>
  onPointerDown: (e: ThreeEvent<PointerEvent>) => void
  handleClick: (e: ThreeEvent<MouseEvent>, index: number) => void
}

function BoardMesh({
  size,
  cells,
  pendingIndex,
  winningLine,
  lastAiMove,
  focusRef,
  onPointerDown,
  handleClick,
}: BoardMeshProps) {
  const slots = useMemo(() => {
    const arr: { index: number; pos: [number, number, number] }[] = []
    for (let i = 0; i < size ** 3; i++) arr.push({ index: i, pos: cellPosition(i, size, EXPLODE) })
    return arr
  }, [size])

  return (
    <group>
      {slots.map(({ index, pos }) => {
        const value = cells[index]
        if (value !== 0) {
          return (
            <Mark
              key={index}
              value={value}
              index={index}
              position={pos}
              size={size}
              focusRef={focusRef}
              onPointerDown={onPointerDown}
              onClick={handleClick}
            />
          )
        }
        return (
          <AnimatedSlot
            key={index}
            index={index}
            position={pos}
            size={size}
            focusRef={focusRef}
            onPointerDown={onPointerDown}
            onClick={handleClick}
          />
        )
      })}

      {pendingIndex != null &&
        pendingIndex >= 0 &&
        pendingIndex < size ** 3 &&
        cells[pendingIndex] === 0 && (
          <PendingHighlight position={cellPosition(pendingIndex, size, EXPLODE)} />
        )}

      {lastAiMove != null &&
        lastAiMove >= 0 &&
        lastAiMove < size ** 3 && (
          <LastAiMoveHighlight position={cellPosition(lastAiMove, size, EXPLODE)} />
        )}

      {winningLine && winningLine.length >= 2 && <WinBeam line={winningLine} size={size} />}
    </group>
  )
}

export function Board3D({
  size,
  cells,
  onCellClick,
  pendingIndex,
  winningLine,
  lastAiMove,
  hintIndex,
  thinking,
  startKey,
}: Board3DProps) {
  // Spherical orbit target, written by the gestures, read by OrbitRig.
  const target = useRef<OrbitTarget>(defaultOrbitTarget(size))
  const userZoomed = useRef(false)
  const pinchStartDist = useRef(0)
  // Live highlight state read by the slots' frame loops. Updated here in an RN
  // effect (fires instantly) so clicks don't re-render the R3F scene.
  const focusRef = useRef<{ pending: number; hint: number; thinking: boolean }>({
    pending: -1,
    hint: -1,
    thinking: false,
  })
  useEffect(() => {
    focusRef.current = { pending: pendingIndex ?? -1, hint: hintIndex ?? -1, thinking }
  }, [pendingIndex, hintIndex, thinking])
  // Pointer-down position for the manual 10px click threshold (web parity).
  const downRef = useRef<{ x: number; y: number } | null>(null)
  // Keep the latest onCellClick without changing handleClick's identity (so the
  // memoized slots don't re-render on every GameScreen state change).
  const onCellClickRef = useRef(onCellClick)
  onCellClickRef.current = onCellClick

  const handlePointerDown = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    downRef.current = { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY }
  }, [])

  const handleClick = useCallback(
    (e: ThreeEvent<MouseEvent>, index: number) => {
      e.stopPropagation()
      const d = downRef.current
      const x = e.nativeEvent.offsetX
      const y = e.nativeEvent.offsetY
      if (d && Math.hypot(x - d.x, y - d.y) > 10) return
      onCellClickRef.current(index)
    },
    [],
  )

const pan = useRef({ theta: 0, phi: 0 })
  const gestureStart = useRef<{ x: number; y: number } | null>(null)
  const pinchRefDist = useRef(0)
  const pinchActive = useRef(false)

  // Android: react-native-gesture-handler never sees touches over the expo-gl
  // GLSurfaceView (it consumes them at the native layer before RNGH's handlers).
  // Instead we use RN's *responder capture* phase on the wrapper View, which runs
  // before the GLView's own PanResponder (that R3F installs for cell taps):
  //   - start phase: never claim -> taps fall through to the GLView -> cell taps.
  //   - move phase: claim when the gesture becomes a drag or a second finger
  //     lands -> we own pan/pinch and R3F cancels its pointer (no stray mark).
  // This uses the same JS responder system R3F relies on, so it works over GL.
  const handleResponderGrant = useCallback((e: GestureResponderEvent) => {
    const touches = e.nativeEvent.touches
    gestureStart.current = gestureStart.current ?? {
      x: e.nativeEvent.pageX,
      y: e.nativeEvent.pageY,
    }
    pan.current = { theta: target.current.theta, phi: target.current.phi }
    pinchStartDist.current = target.current.distance
    pinchActive.current = touches && touches.length >= 2
    if (pinchActive.current) {
      pinchRefDist.current = Math.hypot(
        touches[1].pageX - touches[0].pageX,
        touches[1].pageY - touches[0].pageY,
      )
    }
  }, [])

  const handleResponderMove = useCallback((e: GestureResponderEvent) => {
    const touches = e.nativeEvent.touches
    if (touches && touches.length >= 2) {
      const dist = Math.hypot(
        touches[1].pageX - touches[0].pageX,
        touches[1].pageY - touches[0].pageY,
      )
      if (dist > 1) {
        if (!pinchActive.current || pinchRefDist.current <= 0) {
          pinchActive.current = true
          pinchRefDist.current = dist
          pinchStartDist.current = target.current.distance
        }
        userZoomed.current = true
        target.current.distance = clamp(
          pinchStartDist.current * (pinchRefDist.current / dist),
          MIN_DISTANCE,
          MAX_DISTANCE,
        )
      }
      return
    }
    pinchActive.current = false
    if (gestureStart.current) {
      const t = target.current
      t.theta = pan.current.theta - (e.nativeEvent.pageX - gestureStart.current.x) * 0.008
      t.phi = clamp(
        pan.current.phi - (e.nativeEvent.pageY - gestureStart.current.y) * 0.008,
        MIN_PHI,
        MAX_PHI,
      )
    }
  }, [])

  const handleResponderEnd = useCallback(() => {
    gestureStart.current = null
    pinchActive.current = false
    pinchRefDist.current = 0
  }, [])

  return (
    <View
      style={styles.canvasHost}
      collapsable={false}
      onStartShouldSetResponderCapture={(e) => {
        gestureStart.current = { x: e.nativeEvent.pageX, y: e.nativeEvent.pageY }
        return false
      }}
      onMoveShouldSetResponderCapture={(e) => {
        const touches = e.nativeEvent.touches
        if (touches && touches.length >= 2) return true
        const s = gestureStart.current
        if (!s) return false
        return (
          Math.abs(e.nativeEvent.pageX - s.x) > 10 ||
          Math.abs(e.nativeEvent.pageY - s.y) > 10
        )
      }}
      onResponderGrant={handleResponderGrant}
      onResponderMove={handleResponderMove}
      onResponderRelease={handleResponderEnd}
      onResponderTerminate={handleResponderEnd}
    >
        <Canvas
          style={styles.canvas}
          camera={{ position: INITIAL_CAMERA_POSITION, fov: 45 }}
          gl={{ antialias: true }}
        >
          <ambientLight intensity={0.5} />
          <pointLight position={[6, 6, 6]} intensity={1.2} color="#22d3ee" />
          <pointLight position={[-6, -4, 4]} intensity={0.8} color="#f472b6" />
          <CubePop startKey={startKey}>
            <BoardMesh
              size={size}
              cells={cells}
              pendingIndex={pendingIndex}
              winningLine={winningLine}
              lastAiMove={lastAiMove}
              focusRef={focusRef}
              onPointerDown={handlePointerDown}
              handleClick={handleClick}
            />
          </CubePop>
          <FitRig size={size} userZoomed={userZoomed} target={target} />
          <OrbitRig target={target} />
        </Canvas>
</View>
  )
}

const styles = StyleSheet.create({
  canvasHost: {
    flex: 1,
    overflow: 'hidden',
    backgroundColor: Theme.bg,
  },
  canvas: {
    flex: 1,
  },
})