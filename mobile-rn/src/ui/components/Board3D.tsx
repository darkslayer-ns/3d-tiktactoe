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
 *  - ONE-FINGER DRAG: orbit. Implemented with react-native-gesture-handler
 *    `Gesture.Pan` wrapped around the <Canvas>. RNGH runs at the native
 *    gesture layer, so it never steals the JS responder R3F needs for taps.
 *  - TWO-FINGER PINCH: zoom. `Gesture.Pinch`, composed simultaneously.
 *  - CELL TAPS: R3F-native mesh events (onPointerDown + onClick). The native
 *    Canvas only fires `onClick` when the whole gesture moved < 20px; we add
 *    a stricter 10px threshold on top for web parity by comparing the pointer
 *    down position recorded in onPointerDown.
 *
 * R3F-native gotchas coded around:
 *  - `onClick` needs the PanResponder that the native Canvas installs. RNGH
 *    pan only activates after a movement threshold, so a tap never activates
 *    it and R3F sees the full tap; a drag cancels R3F's pointer (no click is
 *    fired), which is exactly what we want (drag never places a mark).
 *  - Non-interactive meshes get `raycast={() => null}` so they stay visible
 *    but are never hit by the raycaster (same trick as the web build).
 *  - The native Canvas manages dpr/antialias itself; we pass `gl.antialias`
 *    which it maps to GLView `msaaSamples`.
 */

import { useMemo, useRef, useCallback, type RefObject } from 'react'
import { View, StyleSheet } from 'react-native'
import { Canvas, useFrame, type ThreeEvent } from '@react-three/fiber/native'
import { Gesture, GestureDetector } from 'react-native-gesture-handler'
import * as THREE from 'three'
import { Theme } from '../theme'
import { cellCoord, type Cell, type Coord } from '../../game/types'

/** Spread cells outward so inner cells of 4×4×4 / 5×5×5 are reachable. */
const EXPLODE = 0.4

/** Camera orbit limits (web OrbitControls used 3..14). */
const MIN_DISTANCE = 4
const MAX_DISTANCE = 14
const MIN_PHI = 0.15
const MAX_PHI = Math.PI - 0.15

const INITIAL_CAMERA_POSITION: [number, number, number] = [5, 4.5, 5.5]

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
  /** true when the cell may be selected by the player right now */
  interactive: (index: number) => boolean
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

/** cell index -> world position (same mapping as the web build). */
function cellPosition(index: number, size: number, expl: number): [number, number, number] {
  const [x, y, z] = cellCoord(index, size)
  const off = (size - 1) / 2
  const k = 1 + expl
  return [(x - off) * k, (off - y) * k, (z - off) * k]
}

/** Current spherical of the initial camera position — the default orbit. */
function defaultOrbitTarget(): OrbitTarget {
  const s = new THREE.Spherical().setFromVector3(
    new THREE.Vector3(...INITIAL_CAMERA_POSITION),
  )
  return { theta: s.theta, phi: s.phi, distance: s.radius }
}

interface MarkProps {
  value: Cell
  index: number
  position: [number, number, number]
  highlighted?: boolean
  dim?: boolean
  interactive: boolean
  onPointerDown: (e: ThreeEvent<PointerEvent>) => void
  onClick: (e: ThreeEvent<MouseEvent>, index: number) => void
}

function Mark({ value, index, position, highlighted, dim, interactive, onPointerDown, onClick }: MarkProps) {
  const scale = 0.55
  const intensity = dim ? 0.12 : highlighted ? 2.2 : 0.9
  // Non-interactive marks stay visible but are never hit by the raycaster.
  const meshProps = interactive ? {} : { raycast: () => null }

  if (value === 1) {
    // an X lying in the face plane: two thin bars crossing at 45°/135°
    const xLen = 1.15 * scale
    const mat = (
      <meshStandardMaterial
        color="#22d3ee"
        emissive="#22d3ee"
        emissiveIntensity={intensity}
        roughness={0.25}
      />
    )
    return (
      <group
        position={position}
        onPointerDown={onPointerDown}
        onClick={(e: ThreeEvent<MouseEvent>) => onClick(e, index)}
      >
        <mesh {...meshProps} rotation={[0, 0, Math.PI / 4]}>
          <boxGeometry args={[xLen, 0.22, 0.22]} />
          {mat}
        </mesh>
        <mesh {...meshProps} rotation={[0, 0, -Math.PI / 4]}>
          <boxGeometry args={[xLen, 0.22, 0.22]} />
          {mat}
        </mesh>
      </group>
    )
  }
  return (
    <group
      position={position}
      onPointerDown={onPointerDown}
      onClick={(e: ThreeEvent<MouseEvent>) => onClick(e, index)}
    >
      <mesh {...meshProps}>
        <torusGeometry args={[0.42 * scale, 0.12, 16, 48]} />
        <meshStandardMaterial
          color="#f472b6"
          emissive="#f472b6"
          emissiveIntensity={intensity}
          roughness={0.25}
        />
      </mesh>
    </group>
  )
}

interface SlotProps {
  index: number
  position: [number, number, number]
  interactive: boolean
  dim: boolean
  pulse: boolean
  onPointerDown: (e: ThreeEvent<PointerEvent>) => void
  onClick: (e: ThreeEvent<MouseEvent>, index: number) => void
}

function AnimatedSlot({ index, position, interactive, dim, pulse, onPointerDown, onClick }: SlotProps) {
  const spacing = 0.82 * (1 + EXPLODE)
  const slot = spacing * 0.94
  const matRef = useRef<THREE.MeshStandardMaterial>(null)

  useFrame((state) => {
    const mat = matRef.current
    if (!mat) return
    const target = dim ? 0.05 : 0.28
    mat.opacity += (target - mat.opacity) * 0.12
    if (pulse) {
      mat.emissive.set('#22d3ee')
      mat.emissiveIntensity = 0.7 + 0.6 * Math.sin(state.clock.elapsedTime * 6)
      mat.color.set('#164e63')
    } else {
      mat.emissiveIntensity = 0
      mat.color.set('#0f172a')
    }
  })

  return (
    <mesh
      position={position}
      {...(interactive ? {} : { raycast: () => null })}
      onPointerDown={onPointerDown}
      onClick={(e) => onClick(e, index)}
    >
      <boxGeometry args={[slot, slot, slot]} />
      <meshStandardMaterial
        ref={matRef}
        color="#0f172a"
        emissive="#22d3ee"
        emissiveIntensity={0}
        transparent
        opacity={0.28}
        roughness={0.6}
        metalness={0.3}
      />
    </mesh>
  )
}

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
    const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir)
    return { mid: mid.toArray() as [number, number, number], len, quat }
  }, [line, off, k])
  return (
    <group position={mid}>
      <mesh raycast={() => null} quaternion={quat}>
        <cylinderGeometry args={[0.05, 0.05, len + 0.4, 8]} />
        <meshStandardMaterial
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

/**
 * Eases the camera along a spherical orbit toward `target` (fed by the
 * pan/pinch gestures). Runs inside the R3F frame loop.
 */
function OrbitRig({ target }: { target: RefObject<OrbitTarget> }) {
  const spherical = useRef<THREE.Spherical | null>(null)

  useFrame((state, dt) => {
    const cam = state.camera
    if (!spherical.current) {
      spherical.current = new THREE.Spherical().setFromVector3(cam.position)
    }
    const cur = spherical.current
    const t = target.current
    const damp = 1 - Math.exp(-dt * 6)
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
  interactive: (index: number) => boolean
  onPointerDown: (e: ThreeEvent<PointerEvent>) => void
  handleClick: (e: ThreeEvent<MouseEvent>, index: number) => void
}

function BoardMesh({
  size,
  cells,
  pendingIndex,
  winningLine,
  interactive,
  onPointerDown,
  handleClick,
}: BoardMeshProps) {
  const slots = useMemo(() => {
    const arr: { index: number; pos: [number, number, number] }[] = []
    for (let i = 0; i < size ** 3; i++) arr.push({ index: i, pos: cellPosition(i, size, EXPLODE) })
    return arr
  }, [size])

  const focusing = pendingIndex != null && pendingIndex >= 0 && pendingIndex < size ** 3
  const axisSet = useMemo(() => axisCross(focusing ? pendingIndex : null, size), [focusing, pendingIndex, size])

  return (
    <group>
      {/* outer wireframe shell — never blocks clicks */}
      <mesh raycast={() => null}>
        <boxGeometry args={[size * (1 + EXPLODE), size * (1 + EXPLODE), size * (1 + EXPLODE)]} />
        <meshBasicMaterial color="#1e293b" wireframe transparent opacity={0.35} />
      </mesh>

      {slots.map(({ index, pos }) => {
        const value = cells[index]
        const onAxis = axisSet.has(index)
        const dim = focusing && !onAxis
        if (value !== 0) {
          return (
            <Mark
              key={index}
              value={value}
              index={index}
              position={pos}
              dim={dim}
              interactive={interactive(index)}
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
            interactive={interactive(index)}
            dim={dim}
            pulse={onAxis}
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
  interactive,
}: Board3DProps) {
  // Spherical orbit target, written by the gestures, read by OrbitRig.
  const target = useRef<OrbitTarget>(defaultOrbitTarget())
  const panStart = useRef<{ theta: number; phi: number }>({ theta: 0, phi: 0 })
  const pinchStartDist = useRef(0)
  // Pointer-down position for the manual 10px click threshold (web parity).
  const downRef = useRef<{ x: number; y: number } | null>(null)

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
      onCellClick(index)
    },
    [onCellClick],
  )

  const pan = useMemo(
    () =>
      Gesture.Pan()
        .maxPointers(1)
        .onStart(() => {
          panStart.current = { theta: target.current.theta, phi: target.current.phi }
        })
        .onUpdate((e) => {
          const t = target.current
          t.theta = panStart.current.theta - e.translationX * 0.008
          t.phi = clamp(panStart.current.phi + e.translationY * 0.008, MIN_PHI, MAX_PHI)
        }),
    [],
  )

  const pinch = useMemo(
    () =>
      Gesture.Pinch()
        .onStart(() => {
          pinchStartDist.current = target.current.distance
        })
        .onUpdate((e) => {
          if (e.scale <= 0.01) return
          target.current.distance = clamp(
            pinchStartDist.current / e.scale,
            MIN_DISTANCE,
            MAX_DISTANCE,
          )
        }),
    [],
  )

  const composed = useMemo(() => Gesture.Simultaneous(pan, pinch), [pan, pinch])

  return (
    <GestureDetector gesture={composed}>
      <View style={styles.canvasHost}>
        <Canvas
          style={styles.canvas}
          camera={{ position: INITIAL_CAMERA_POSITION, fov: 45 }}
          gl={{ antialias: true }}
        >
          <ambientLight intensity={0.5} />
          <pointLight position={[6, 6, 6]} intensity={1.2} color="#22d3ee" />
          <pointLight position={[-6, -4, 4]} intensity={0.8} color="#f472b6" />
          <BoardMesh
            size={size}
            cells={cells}
            pendingIndex={pendingIndex}
            winningLine={winningLine}
            interactive={interactive}
            onPointerDown={handlePointerDown}
            handleClick={handleClick}
          />
          <OrbitRig target={target} />
        </Canvas>
      </View>
    </GestureDetector>
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