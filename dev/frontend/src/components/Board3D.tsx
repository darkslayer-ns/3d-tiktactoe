import { useMemo, useRef } from 'react'
import { Canvas, useFrame, ThreeEvent } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'

export type CellValue = 0 | 1 | 2
export type Cells = CellValue[]

export interface Board3DProps {
  size: number
  cells: Cells
  onCellClick?: (index: number) => void
  /** fired when the user taps the board but misses every cell */
  onEmptyClick?: () => void
  hoverable?: (index: number) => boolean
  winningLine?: [number, number, number][] | null
  autoRotate?: boolean
  /** spread cells outward from the center so inner cells are reachable */
  explode?: number
  /** highlight cells (e.g. predicted move) */
  highlight?: number[]
  /** an empty cell selected but not yet confirmed (double-click to lock) */
  pendingIndex?: number | null
}

function cellPosition(index: number, size: number, expl = 0): [number, number, number] {
  const z = Math.floor(index / (size * size))
  const r = index % (size * size)
  const y = Math.floor(r / size)
  const x = r % size
  const off = (size - 1) / 2
  const k = 1 + expl
  return [(x - off) * k, (off - y) * k, (z - off) * k]
}

/** The X-row and Y-column passing through `center` (a "+" cross in its layer). */
export function axisCross(center: number | null, size: number): Set<number> {
  const set = new Set<number>()
  if (center == null || center < 0) return set
  const z = Math.floor(center / (size * size))
  const r = center % (size * size)
  const y = Math.floor(r / size)
  const x = r % size
  for (let i = 0; i < size; i++) {
    set.add(i + size * (y + size * z)) // row along X
    set.add(x + size * (i + size * z)) // column along Y
  }
  return set
}

interface MarkProps {
  value: CellValue
  index: number
  position: [number, number, number]
  highlighted?: boolean
  dim?: boolean
  onClick?: (i: number) => void
}

function Mark({ value, index, position, highlighted, dim, onClick }: MarkProps) {
  const scale = 0.55
  const intensity = dim ? 0.12 : highlighted ? 2.2 : 0.9
  const start = useRef<{ x: number; y: number } | null>(null)
  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation()
    const s = start.current
    const dx = s ? e.nativeEvent.clientX - s.x : 0
    const dy = s ? e.nativeEvent.clientY - s.y : 0
    if (Math.hypot(dx, dy) > 10) return
    onClick?.(index)
  }
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
        onPointerDown={(e) => {
          e.stopPropagation()
          start.current = { x: e.nativeEvent.clientX, y: e.nativeEvent.clientY }
        }}
        onClick={handleClick}
      >
        <mesh rotation={[0, 0, Math.PI / 4]}>
          <boxGeometry args={[xLen, 0.22, 0.22]} />
          {mat}
        </mesh>
        <mesh rotation={[0, 0, -Math.PI / 4]}>
          <boxGeometry args={[xLen, 0.22, 0.22]} />
          {mat}
        </mesh>
      </group>
    )
  }
  return (
    <group
      position={position}
      onPointerDown={(e) => {
        e.stopPropagation()
        start.current = { x: e.nativeEvent.clientX, y: e.nativeEvent.clientY }
      }}
      onClick={handleClick}
    >
      <mesh>
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

function AnimatedSlot({
  index,
  position,
  interactive,
  onClick,
  explode,
  dim,
  pulse,
}: {
  index: number
  position: [number, number, number]
  interactive: boolean
  onClick?: (i: number) => void
  explode: number
  /** fade this slot out (off-axis) */
  dim?: boolean
  /** pulse this slot (a possible box on the selected axis) */
  pulse?: boolean
}) {
  const spacing = 0.82 * (1 + explode)
  const slot = spacing * 0.94
  const matRef = useRef<THREE.MeshStandardMaterial>(null)
  const start = useRef<{ x: number; y: number } | null>(null)

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
      onPointerDown={(e) => {
        e.stopPropagation()
        start.current = { x: e.nativeEvent.clientX, y: e.nativeEvent.clientY }
      }}
      onClick={(e) => {
        e.stopPropagation()
        if (!interactive || !onClick) return
        const s = start.current
        const dx = s ? e.nativeEvent.clientX - s.x : 0
        const dy = s ? e.nativeEvent.clientY - s.y : 0
        if (Math.hypot(dx, dy) > 10) return
        onClick(index)
      }}
      onPointerOver={(e) => {
        if (!interactive) return
        e.stopPropagation()
        document.body.style.cursor = 'pointer'
      }}
      onPointerOut={() => {
        document.body.style.cursor = 'auto'
      }}
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

function WinBeam({ line, size, explode }: { line: [number, number, number][]; size: number; explode: number }) {
  const [a, b] = [line[0], line[line.length - 1]]
  const off = (size - 1) / 2
  const k = 1 + explode
  const p1 = new THREE.Vector3((a[0] - off) * k, (off - a[1]) * k, (a[2] - off) * k)
  const p2 = new THREE.Vector3((b[0] - off) * k, (off - b[1]) * k, (b[2] - off) * k)
  const mid = p1.clone().add(p2).multiplyScalar(0.5)
  const len = p1.distanceTo(p2)
  const dir = p2.clone().sub(p1).normalize()
  const quat = useMemo(() => {
    const q = new THREE.Quaternion()
    q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir)
    return q
  }, [dir.x, dir.y, dir.z])
  return (
    <group position={mid.toArray()}>
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

function PendingHighlight({ position, explode }: { position: [number, number, number]; explode: number }) {
  const s = 0.9 * (1 + explode)
  return (
    <mesh raycast={() => null} position={position}>
      <boxGeometry args={[s, s, s]} />
      <meshBasicMaterial
        color="#fef08a"
        transparent
        opacity={0.35}
        depthWrite={false}
      />
    </mesh>
  )
}

function PredictedGlow({ position }: { position: [number, number, number] }) {
  return (
    <mesh raycast={() => null} position={position}>
      <boxGeometry args={[0.95, 0.95, 0.95]} />
      <meshBasicMaterial color="#a3e635" transparent opacity={0.18} depthWrite={false} />
    </mesh>
  )
}

function BoardMesh({
  size,
  cells,
  onCellClick,
  onEmptyClick,
  hoverable,
  winningLine,
  explode,
  highlight,
  pendingIndex,
}: Board3DProps) {
  const expl = explode ?? 0
  const slots = useMemo(() => {
    const arr: { index: number; pos: [number, number, number] }[] = []
    for (let i = 0; i < size ** 3; i++) arr.push({ index: i, pos: cellPosition(i, size, expl) })
    return arr
  }, [size, expl])

  const highlightSet = useMemo(() => new Set(highlight ?? []), [highlight])

  const focusing = pendingIndex != null && pendingIndex >= 0 && pendingIndex < size ** 3
  const axisSet = useMemo(() => axisCross(focusing ? pendingIndex : null, size), [focusing, pendingIndex, size])

  return (
    <group
      onClick={(e) => {
        // a click that missed every cell (wireframe/gap) toggles the board open
        e.stopPropagation()
        onEmptyClick?.()
      }}
    >
      {/* outer wireframe shell, scaled with the explode — never blocks clicks */}
      <mesh raycast={() => null}>
        <boxGeometry args={[size * (1 + expl), size * (1 + expl), size * (1 + expl)]} />
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
              highlighted={highlightSet.has(index)}
              dim={dim}
              onClick={onCellClick}
            />
          )
        }
        return (
          <group key={index}>
            <AnimatedSlot
              index={index}
              position={pos}
              interactive={!!(onCellClick && hoverable && hoverable(index))}
              onClick={onCellClick}
              explode={expl}
              dim={dim}
              pulse={onAxis}
            />
            {highlightSet.has(index) && <PredictedGlow position={pos} />}
          </group>
        )
      })}
      {pendingIndex != null &&
        pendingIndex >= 0 &&
        pendingIndex < size ** 3 &&
        cells[pendingIndex] === 0 && (
          <PendingHighlight position={cellPosition(pendingIndex, size, expl)} explode={expl} />
        )}
      {winningLine && winningLine.length >= 2 && (
        <WinBeam line={winningLine} size={size} explode={expl} />
      )}
    </group>
  )
}

export function Board3D(props: Board3DProps) {
  return (
    <Canvas
      camera={{ position: [5, 4.5, 5.5], fov: 45 }}
      gl={{ antialias: true, alpha: true }}
      style={{ background: 'transparent' }}
    >
      <ambientLight intensity={0.5} />
      <pointLight position={[6, 6, 6]} intensity={1.2} color="#22d3ee" />
      <pointLight position={[-6, -4, 4]} intensity={0.8} color="#f472b6" />
      <BoardMesh {...props} />
      <OrbitControls
        enablePan={false}
        autoRotate={props.autoRotate}
        autoRotateSpeed={1.4}
        minDistance={3}
        maxDistance={14}
      />
    </Canvas>
  )
}