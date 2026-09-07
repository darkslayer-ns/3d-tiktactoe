/**
 * Native 3D board for Neon Cube — a port of the web Board3D to
 * @react-three/fiber/native, reworked for low-end Android perf.
 *
 * Perf model:
 *  - Every visual is one of a handful of InstancedMesh draw calls. A 5×5×5
 *    board used to be ~125 slots × ~16 meshes (~2,000 draw calls); it's now
 *    5 instanced draw calls (slots / X / O / X-halo / O-halo) + 1 hit mesh.
 *  - Geometry is low-poly and baked from Blender-authored GLBs into
 *    `src/three/models.ts` (see scripts/gen_models.mjs), so there is no
 *    runtime GLB parsing and no `file://` fetch on Android.
 *  - The frame loop writes instance matrices/colors directly — no React
 *    reconciler commits on moves/thinking, same as the previous version.
 *
 * Visuals match the previous build:
 *  - X marks = two crossed cyan emissive bars (beveled Blender mesh)
 *  - O marks = pink emissive torus ring
 *  - empty cells = dark translucent slot frames that pulse cyan on the
 *    selected axis / hint, dim off-axis, and "breathe" while thinking
 *  - a glowing purple cylinder draws the winning line
 *  - a yellow box marks the pending (selected) cell; white pulses last AI move
 *
 * Subtle polish (opt-in via props, always on here):
 *  - marks billboard to face the camera, so the flat X/O cards read face-on
 *    from any orbit angle
 *  - an additive "halo" copy gives marks a soft rim/glow; idle color pulse
 *
 * Win/lose is a Blender-rendered trophy image shown by the GameOverOverlay.
 */

import { memo, useCallback, useEffect, useMemo, useRef, type ReactNode, type RefObject } from 'react'
import { View, AppState, StyleSheet, type GestureResponderEvent } from 'react-native'
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber/native'
import * as THREE from 'three'
import { Theme } from '../theme'
import { cellCoord, type Cell, type Coord } from '../../game/types'
import { modelGeometry } from '../../three/geometry'

/** Spread cells outward so inner cells of 4×4×4 / 5×5×5 are reachable. */
const EXPLODE = 0.4

/** Side length of each cell's invisible hit box. */
const HIT_SIZE = 0.918

/** Camera orbit limits (web OrbitControls used 3..14). */
const MIN_DISTANCE = 4
const MAX_DISTANCE = 40
const MIN_PHI = 0.15
const MAX_PHI = Math.PI - 0.15

const INITIAL_CAMERA_POSITION: [number, number, number] = [5, 4.5, 5.5]

/** Camera distance that fits the whole n×n×n cube (with the explode spread). */
function fitDistance(size: number, aspect = 0.5, vFovDeg = 45): number {
  const halfExtent = ((size - 1) / 2) * (1 + EXPLODE) + 0.478
  const radius = halfExtent * Math.sqrt(3)
  const vHalf = (vFovDeg * Math.PI) / 360
  const hHalf = Math.atan(Math.tan(vHalf) * Math.max(0.1, aspect))
  const limitingHalf = Math.min(hHalf, vHalf)
  const d = radius / Math.tan(limitingHalf)
  return clamp(d * 1.1, MIN_DISTANCE, MAX_DISTANCE)
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

/** Springy pop-in easing (easeOutBack). */
function easeOutBack(t: number): number {
  const c1 = 1.70158
  const c3 = c1 + 1
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2)
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
  /** game result, for the win/lose celebration (0 = none/draw) */
  winner?: number
  over?: boolean
  /** which side the human plays (1 = X, 2 = O) — decides win vs lose */
  humanSide?: number
  /** optional gate: cells where this returns false are not tappable */
  interactive?: (index: number) => boolean
}

/**
 * Mutable board state read every frame by the frame loops. Board3D mutates it
 * on every render (RN side, instant) WITHOUT re-rendering the R3F Canvas
 * (memoized), so all visuals update on the next GL frame — no React
 * reconciler commits, which are what lagged big boards.
 */
interface GameStateRef {
  cells: Cell[]
  size: number
  pending: number
  hint: number
  thinking: boolean
  lastAiMove: number
  winningLine: Coord[] | null
  winner: number
  over: boolean
  humanSide: number
  interactive: ((index: number) => boolean) | null
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
  const s = new THREE.Spherical().setFromVector3(new THREE.Vector3(...INITIAL_CAMERA_POSITION))
  return { theta: s.theta, phi: s.phi, distance: fitDistance(size) }
}

interface InstancesProps {
  size: number
  gameRef: RefObject<GameStateRef>
  onPointerDown: (e: ThreeEvent<PointerEvent>) => void
  handleClick: (e: ThreeEvent<MouseEvent>, index: number) => void
}

/** Half extent of the outer cube (used to place celebration tokens above it). */
function boardTop(size: number): number {
  return ((size - 1) / 2) * (1 + EXPLODE) + 0.478
}

/**
 * All board visuals as instanced meshes, updated entirely from the frame loop.
 */
function Instances({ size, gameRef, onPointerDown, handleClick }: InstancesProps) {
  const count = size ** 3

  const slotGeo = useMemo(() => modelGeometry('slot'), [])
  const xGeo = useMemo(() => modelGeometry('mark_x'), [])
  const oGeo = useMemo(() => modelGeometry('mark_o'), [])
  const hitGeo = useMemo(() => new THREE.BoxGeometry(1, 1, 1), [])

  const slotMat = useMemo(() => {
    const mat = new THREE.MeshBasicMaterial({
      color: '#ffffff',
      transparent: true,
      opacity: 0.35,
      depthWrite: false,
    })
    // The per-instance color (cyan × per-cell brightness) drives BOTH the tint
    // and the alpha, restoring the translucent "glass" slot edges of the
    // original build (which faded opacity per cell) instead of an opaque box.
    mat.onBeforeCompile = (shader) => {
      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <opacity_fragment>',
        `#include <opacity_fragment>
      diffuseColor.a *= max(vColor.r, max(vColor.g, vColor.b));`,
      )
    }
    return mat
  }, [])
  const xMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#22d3ee',
        emissive: new THREE.Color('#22d3ee'),
        emissiveIntensity: 0.5,
        roughness: 0.3,
        metalness: 0.05,
      }),
    [],
  )
  const oMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#f472b6',
        emissive: new THREE.Color('#f472b6'),
        emissiveIntensity: 0.5,
        roughness: 0.3,
        metalness: 0.05,
      }),
    [],
  )
  const haloXMat = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: '#ffffff',
        transparent: true,
        opacity: 0.14,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  )
  const haloOMat = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: '#ffffff',
        transparent: true,
        opacity: 0.14,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  )
  const hitMat = useMemo(
    () => new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }),
    [],
  )

  const slotsRef = useRef<THREE.InstancedMesh>(null)
  const xRef = useRef<THREE.InstancedMesh>(null)
  const oRef = useRef<THREE.InstancedMesh>(null)
  const haloXRef = useRef<THREE.InstancedMesh>(null)
  const haloORef = useRef<THREE.InstancedMesh>(null)
  const hitsRef = useRef<THREE.InstancedMesh>(null)

  // Precomputed cell world positions (constant per size).
  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      const [x, y, z] = cellPosition(i, size, EXPLODE)
      arr[i * 3] = x
      arr[i * 3 + 1] = y
      arr[i * 3 + 2] = z
    }
    return arr
  }, [count, size])

  const born = useMemo(() => new Float32Array(count).fill(-1), [count])
  // Sync prevVal with the CURRENT board on mount (or on a scene remount after
  // the GL context is recreated on foreground). Without this, an existing X/O
  // would look like a brand-new placement and replay its pop-in animation.
  const prevVal = useMemo(() => {
    const arr = new Int8Array(count)
    const cells = gameRef.current?.cells
    if (cells) for (let i = 0; i < count; i++) arr[i] = cells[i] ?? 0
    return arr
  }, [count])
  // Detects a render-clock reset (e.g. right after resume) so marks settle
  // immediately instead of animating from a stale "born" timestamp.
  const lastNow = useRef<number | null>(null)

  // Reusable temporaries (no per-frame allocations).
  const m = useMemo(() => new THREE.Matrix4(), [])
  const q = useMemo(() => new THREE.Quaternion(), [])
  const v = useMemo(() => new THREE.Vector3(), [])
  const dir = useMemo(() => new THREE.Vector3(), [])
  const lightPos = useMemo(() => new THREE.Vector3(6, 10, 4), [])
  const s = useMemo(() => new THREE.Vector3(), [])
  const c = useMemo(() => new THREE.Color(), [])
  const zAxis = useMemo(() => new THREE.Vector3(0, 0, 1), [])

  const cSlot = useMemo(() => new THREE.Color('#0284c7'), [])
  const cSlotDim = useMemo(() => new THREE.Color('#0c4a6e'), [])
  const cSlotBright = useMemo(() => new THREE.Color('#22d3ee'), [])
  const cCyan = useMemo(() => new THREE.Color('#22d3ee'), [])
  const cPink = useMemo(() => new THREE.Color('#f472b6'), [])
  const cGold = useMemo(() => new THREE.Color('#fbbf24'), [])

  useFrame((state) => {
    const slots = slotsRef.current
    const xm = xRef.current
    const om = oRef.current
    const hx = haloXRef.current
    const ho = haloORef.current
    const hits = hitsRef.current
    if (!slots || !xm || !om || !hx || !ho || !hits) return

    const g = gameRef.current
    const cells = g.cells
    const pending = g.pending
    const now = state.clock.elapsedTime
    const camPos = state.camera.position
    const camDist = camPos.length() || 1
    const half = boardTop(size)

    // A recreated GL context can restart the render clock. Treat the current
    // board as settled instead of replaying every existing mark's pop-in.
    if (lastNow.current != null && now < lastNow.current - 0.5) {
      for (let i = 0; i < count; i++) {
        born[i] = -1
        prevVal[i] = cells[i] ?? 0
      }
    }
    lastNow.current = now

    // Marks are authored at a fixed size; shrink them slightly on bigger cubes
    // so the denser lattice doesn't collapse into a wall of overlapping cards
    // when they billboard toward the camera.
    const markScale = size >= 6 ? 0.8 : size >= 5 ? 0.88 : size >= 4 ? 0.94 : 1

    const focusing = pending >= 0 && pending < count && cells[pending] === 0
    let ppx = -1
    let ppy = -1
    let ppz = -1
    if (focusing) {
      const pc = cellCoord(pending, size)
      ppx = pc[0]
      ppy = pc[1]
      ppz = pc[2]
    }

    // Winning-line cells flash gold.
    let winSet: Set<number> | null = null
    if (g.winningLine && g.winningLine.length >= 2) {
      winSet = new Set(g.winningLine.map((c) => c[0] + size * (c[1] + size * c[2])))
    }
    const loseDim = g.over && g.winner !== 0 && g.winner !== g.humanSide ? 0.35 : 1

    for (let i = 0; i < count; i++) {
      const px = positions[i * 3]
      const py = positions[i * 3 + 1]
      const pz = positions[i * 3 + 2]
      const val = cells[i]
      const filled = val !== 0
      const cc = cellCoord(i, size)
      const onAxis = focusing && ((cc[0] === ppx && cc[2] === ppz) || (cc[1] === ppy && cc[2] === ppz))
      const hintPulse = i === g.hint
      const dim = focusing && !onAxis

      // Depth fade: map each cell's camera distance onto 0 (nearest) .. 1
      // (farthest) across the cube's extent, then darken far cells so the
      // lattice clearly reads as 3D.
      const depth = camPos.distanceTo(v.set(px, py, pz))
      const depthT = clamp((depth - (camDist - half)) / Math.max(0.01, 2 * half), 0, 1)
      const fade = 1 - 0.88 * depthT

      // Light falloff: marks subtly darken the farther they sit from the key
      // light (same idea as the slots' depth fade, but light-based).
      const lightDist = lightPos.distanceTo(v.set(px, py, pz))
      const lightLen = lightPos.length() || 1
      const lightT = clamp((lightDist - (lightLen - half)) / Math.max(0.01, 2 * half), 0, 1)
      const lightFalloff = 1 - 0.35 * lightT

      // ---- slot frame (empty cells only) ----
      if (!filled) {
        let bright
        if (onAxis || hintPulse) {
          c.copy(cSlotBright)
          bright = (0.55 + 0.25 * Math.sin(now * 6)) * fade
        } else if (g.thinking) {
          c.copy(cSlot)
          bright = (0.45 + 0.2 * (0.5 + 0.5 * Math.sin(now * 4))) * fade
        } else {
          c.copy(dim ? cSlotDim : cSlot)
          bright = (dim ? 0.42 : 0.72) * fade
        }
        c.multiplyScalar(bright * loseDim)
        slots.setColorAt(i, c)
        m.makeScale(1, 1, 1)
        m.setPosition(px, py, pz)
        slots.setMatrixAt(i, m)
      } else {
        m.makeScale(0, 0, 0)
        slots.setMatrixAt(i, m)
      }

      // ---- marks + halo ----
      const mark = val === 1 ? xm : om
      const halo = val === 1 ? hx : ho
      const haloBrand = val === 1 ? cCyan : cPink
      const winner = !!(winSet && winSet.has(i) && g.over)
      if (filled) {
        if (prevVal[i] !== val) born[i] = now
        const bt = born[i]
        const t = bt >= 0 ? Math.min(1, (now - bt) / 0.42) : 1
        const sc = t >= 1 ? 1 : Math.max(0.001, easeOutBack(t))

        // full billboard: point the mark's flat front (+Z) at the camera so the
        // X/O card reads face-on from any orbit angle
        dir.set(camPos.x - px, camPos.y - py, camPos.z - pz).normalize()
        q.setFromUnitVectors(zAxis, dir)

        const scM = sc * markScale
        m.compose(v.set(px, py, pz), q, s.set(scM, scM, scM))
        mark.setMatrixAt(i, m)
        m.compose(v.set(px, py, pz), q, s.set(scM * 1.07, scM * 1.07, scM * 1.07))
        halo.setMatrixAt(i, m)

        // Body brightness (material color is saturated brand; instanceColor
        // modulates it): dim off-axis, brief flash on pop-in, gold-bright win.
        if (winner) {
          c.setScalar(1.35 * fade * lightFalloff)
        } else {
          let mul = 1
          if (dim) mul = 0.4
          if (t < 1) mul *= 1 + 0.7 * (1 - t)
          c.setScalar(mul * fade * loseDim * lightFalloff)
        }
        mark.setColorAt(i, c)

        // Halo tint: brand color normally, pulsing gold on the winning line.
        if (winner) {
          c.copy(cGold).multiplyScalar((0.7 + 0.5 * Math.sin(now * 7)) * lightFalloff)
        } else {
          c.copy(haloBrand).multiplyScalar(lightFalloff)
        }
        halo.setColorAt(i, c)

        // hide the other mark
        const other = val === 1 ? om : xm
        const otherHalo = val === 1 ? ho : hx
        m.makeScale(0, 0, 0)
        other.setMatrixAt(i, m)
        otherHalo.setMatrixAt(i, m)
      } else {
        m.makeScale(0, 0, 0)
        xm.setMatrixAt(i, m)
        om.setMatrixAt(i, m)
        hx.setMatrixAt(i, m)
        ho.setMatrixAt(i, m)
      }
      prevVal[i] = val

      // ---- hit target ----
      const hittable =
        !filled &&
        (!focusing || onAxis) &&
        (g.interactive ? g.interactive(i) : true)
      if (hittable) {
        m.makeScale(HIT_SIZE, HIT_SIZE, HIT_SIZE)
        m.setPosition(px, py, pz)
      } else {
        m.makeScale(0, 0, 0)
      }
      hits.setMatrixAt(i, m)
    }

    slots.instanceMatrix.needsUpdate = true
    xm.instanceMatrix.needsUpdate = true
    om.instanceMatrix.needsUpdate = true
    hx.instanceMatrix.needsUpdate = true
    ho.instanceMatrix.needsUpdate = true
    hits.instanceMatrix.needsUpdate = true
    if (slots.instanceColor) slots.instanceColor.needsUpdate = true
    if (xm.instanceColor) xm.instanceColor.needsUpdate = true
    if (om.instanceColor) om.instanceColor.needsUpdate = true
    if (hx.instanceColor) hx.instanceColor.needsUpdate = true
    if (ho.instanceColor) ho.instanceColor.needsUpdate = true
  })

  return (
    <>
      <instancedMesh ref={slotsRef} args={[slotGeo, slotMat, count]} frustumCulled={false} raycast={() => null} />
      <instancedMesh ref={xRef} args={[xGeo, xMat, count]} frustumCulled={false} raycast={() => null} />
      <instancedMesh ref={oRef} args={[oGeo, oMat, count]} frustumCulled={false} raycast={() => null} />
      <instancedMesh ref={haloXRef} args={[xGeo, haloXMat, count]} frustumCulled={false} raycast={() => null} />
      <instancedMesh ref={haloORef} args={[oGeo, haloOMat, count]} frustumCulled={false} raycast={() => null} />
      <instancedMesh
        ref={hitsRef}
        args={[hitGeo, hitMat, count]}
        frustumCulled={false}
        onPointerDown={onPointerDown}
        onClick={(e) => {
          if (e.instanceId != null) handleClick(e, e.instanceId)
        }}
      />
    </>
  )
}

function WinBeam({ gameRef }: { gameRef: RefObject<GameStateRef> }) {
  const g = useRef<THREE.Group>(null)
  const mat = useRef<THREE.MeshStandardMaterial>(null)
  const lastLine = useRef('')
  const start = useRef(Date.now())

  useFrame(() => {
    const grp = g.current
    const m = mat.current
    const line = gameRef.current.winningLine
    if (grp) {
      if (line && line.length >= 2) {
        const sig = line.join('|')
        if (sig !== lastLine.current) {
          lastLine.current = sig
          const off = (gameRef.current.size - 1) / 2
          const k = 1 + EXPLODE
          const a = line[0]
          const b = line[line.length - 1]
          const p1 = new THREE.Vector3((a[0] - off) * k, (off - a[1]) * k, (a[2] - off) * k)
          const p2 = new THREE.Vector3((b[0] - off) * k, (off - b[1]) * k, (b[2] - off) * k)
          const mid = p1.clone().add(p2).multiplyScalar(0.5)
          const len = p1.distanceTo(p2)
          const dir = p2.clone().sub(p1).normalize()
          const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir)
          grp.position.copy(mid)
          grp.quaternion.copy(q)
          grp.scale.set(1, len + 0.4, 1)
          start.current = Date.now()
        }
        grp.visible = true
      } else {
        grp.visible = false
      }
    }
    if (m && grp && grp.visible) {
      const t = (Date.now() - start.current) / 1000
      const pulse = 0.5 + 0.5 * Math.abs(Math.sin(t * 7))
      m.emissiveIntensity = 1.5 + 2.5 * pulse
      m.opacity = 0.6 + 0.4 * pulse
    }
  })

  return (
    <group ref={g} visible={false}>
      <mesh raycast={() => null}>
        <cylinderGeometry args={[0.07, 0.07, 1, 8]} />
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

function PendingHighlight({ gameRef }: { gameRef: RefObject<GameStateRef> }) {
  const m = useRef<THREE.Mesh>(null)
  const s = 0.9 * (1 + EXPLODE)
  useFrame(() => {
    const mesh = m.current
    if (!mesh) return
    const g = gameRef.current
    const p = g.pending
    const show = p >= 0 && p < g.cells.length && g.cells[p] === 0
    mesh.visible = show
    if (show) {
      const [x, y, z] = cellPosition(p, g.size, EXPLODE)
      mesh.position.set(x, y, z)
    }
  })
  return (
    <mesh ref={m} raycast={() => null} visible={false}>
      <boxGeometry args={[s, s, s]} />
      <meshBasicMaterial color="#fef08a" transparent opacity={0.35} depthWrite={false} />
    </mesh>
  )
}

function LastAiMoveHighlight({ gameRef }: { gameRef: RefObject<GameStateRef> }) {
  const m = useRef<THREE.Mesh>(null)
  const mat = useRef<THREE.MeshBasicMaterial>(null)
  const s = 0.62 * (1 + EXPLODE)
  useFrame((state) => {
    const mesh = m.current
    const matl = mat.current
    if (!mesh || !matl) return
    const g = gameRef.current
    const idx = g.lastAiMove
    const show = idx >= 0 && idx < g.cells.length && g.cells[idx] !== 0
    mesh.visible = show
    if (show) {
      const [x, y, z] = cellPosition(idx, g.size, EXPLODE)
      mesh.position.set(x, y, z)
      matl.opacity = 0.2 + 0.15 * Math.sin(state.clock.elapsedTime * 6)
    }
  })
  return (
    <mesh ref={m} raycast={() => null} visible={false}>
      <boxGeometry args={[s, s, s]} />
      <meshBasicMaterial ref={mat} color="#ffffff" transparent opacity={0.4} depthWrite={false} />
    </mesh>
  )
}

/** "Teleport" pop-in: scales the cube from 0 up on every game start. */
function CubePop({ startKey, children }: { startKey: number; children: ReactNode }) {
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
    const s = Math.max(0.001, easeOutBack(t))
    grp.scale.setScalar(s)
  })

  return <group ref={g}>{children}</group>
}

/**
 * Pauses the R3F render loop while the app is in the background and resumes +
 * forces a redraw when it comes back. expo-gl invalidates the GL drawable when
 * an app is backgrounded (and R3F native does not handle this itself), so a
 * still-running frame loop against a dead surface is what freezes the board on
 * foreground — stopping the loop avoids that and lets it re-render cleanly.
 */
function AppLifecycle() {
  const setFrameloop = useThree((s) => s.setFrameloop)
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    const sub = AppState.addEventListener('change', (next) => {
      if (next === 'active') {
        setFrameloop('always')
        invalidate()
      } else {
        setFrameloop('never')
      }
    })
    return () => sub.remove()
  }, [setFrameloop, invalidate])
  return null
}

/** Keeps the cube fitted to the actual canvas viewport until the user zooms. */
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

/** Eases the camera along a spherical orbit toward `target`. */
function OrbitRig({ target }: { target: RefObject<OrbitTarget> }) {
  const spherical = useRef<THREE.Spherical | null>(null)

  useFrame((state, dt) => {
    const cam = state.camera
    const t = target.current
    if (!spherical.current) {
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
  winner = 0,
  over = false,
  humanSide = 1,
  interactive,
}: Board3DProps) {
  const target = useRef<OrbitTarget>(defaultOrbitTarget(size))
  const userZoomed = useRef(false)
  const pinchStartDist = useRef(0)

  const gameRef = useRef<GameStateRef>({
    cells,
    size,
    pending: pendingIndex ?? -1,
    hint: hintIndex ?? -1,
    thinking,
    lastAiMove: lastAiMove ?? -1,
    winningLine,
    winner,
    over,
    humanSide,
    interactive: interactive ?? null,
  })
  gameRef.current.cells = cells
  gameRef.current.size = size
  gameRef.current.pending = pendingIndex ?? -1
  gameRef.current.hint = hintIndex ?? -1
  gameRef.current.thinking = thinking
  gameRef.current.lastAiMove = lastAiMove ?? -1
  gameRef.current.winningLine = winningLine
  gameRef.current.winner = winner
  gameRef.current.over = over
  gameRef.current.humanSide = humanSide
  gameRef.current.interactive = interactive ?? null

  const downRef = useRef<{ x: number; y: number } | null>(null)
  const onCellClickRef = useRef(onCellClick)
  onCellClickRef.current = onCellClick

  const handlePointerDown = useCallback((e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    downRef.current = { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY }
  }, [])

  const handleClick = useCallback((e: ThreeEvent<MouseEvent>, index: number) => {
    e.stopPropagation()
    const d = downRef.current
    const x = e.nativeEvent.offsetX
    const y = e.nativeEvent.offsetY
    if (d && Math.hypot(x - d.x, y - d.y) > 10) return
    onCellClickRef.current(index)
  }, [])

  const pan = useRef({ theta: 0, phi: 0 })
  const gestureStart = useRef<{ x: number; y: number } | null>(null)
  const pinchRefDist = useRef(0)
  const pinchActive = useRef(false)

  const handleResponderGrant = useCallback((e: GestureResponderEvent) => {
    const touches = e.nativeEvent.touches
    gestureStart.current = gestureStart.current ?? {
      x: e.nativeEvent.pageX,
      y: e.nativeEvent.pageY,
    }
    pan.current = { theta: target.current.theta, phi: target.current.phi }
    pinchStartDist.current = target.current.distance
    pinchActive.current = !!touches && touches.length >= 2
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
      <BoardScene
        gameRef={gameRef}
        size={size}
        startKey={startKey}
        onPointerDown={handlePointerDown}
        handleClick={handleClick}
        target={target}
        userZoomed={userZoomed}
      />
    </View>
  )
}

// Memoized: re-renders ONLY when size/startKey change, so game-state updates
// (cells/pending/thinking) never re-run the R3F Canvas — the frame loop reads
// `gameRef` instead.
const BoardScene = memo(function BoardScene({
  gameRef,
  size,
  startKey,
  onPointerDown,
  handleClick,
  target,
  userZoomed,
}: {
  gameRef: RefObject<GameStateRef>
  size: number
  startKey: number
  onPointerDown: (e: ThreeEvent<PointerEvent>) => void
  handleClick: (e: ThreeEvent<MouseEvent>, index: number) => void
  target: RefObject<OrbitTarget>
  userZoomed: RefObject<boolean>
}) {
  return (
    <Canvas
      style={styles.canvas}
      camera={{ position: INITIAL_CAMERA_POSITION, fov: 45 }}
      gl={{ antialias: true, powerPreference: 'high-performance' }}
    >
      <ambientLight intensity={0.55} />
      <directionalLight position={[6, 10, 4]} intensity={1.0} color="#ffffff" />
      <pointLight position={[6, 6, 6]} intensity={1.2} color="#22d3ee" />
      <pointLight position={[-6, -4, 4]} intensity={0.8} color="#f472b6" />
      <AppLifecycle />
      <CubePop startKey={startKey}>
        <Instances size={size} gameRef={gameRef} onPointerDown={onPointerDown} handleClick={handleClick} />
        <WinBeam gameRef={gameRef} />
        <PendingHighlight gameRef={gameRef} />
        <LastAiMoveHighlight gameRef={gameRef} />
      </CubePop>
      <FitRig size={size} userZoomed={userZoomed} target={target} />
      <OrbitRig target={target} />
    </Canvas>
  )
})

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
