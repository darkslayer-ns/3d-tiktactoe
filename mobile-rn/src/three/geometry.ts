import * as THREE from 'three'
import { MODELS, type ModelData, type ModelName } from './models'

const cache = new Map<ModelName, THREE.BufferGeometry>()

/** Build (and cache) a BufferGeometry from the Blender-baked model data. */
export function modelGeometry(name: ModelName): THREE.BufferGeometry {
  const cached = cache.get(name)
  if (cached) return cached
  const data = MODELS[name]
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(data.positions, 3))
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(data.normals, 3))
  geo.setIndex(data.indices)
  geo.computeBoundingSphere()
  cache.set(name, geo)
  return geo
}

/** Full extent (max - min) along each axis of a model, as [x, y, z]. */
export function modelSize(name: ModelName): [number, number, number] {
  return MODELS[name].size
}

export type { ModelData, ModelName }
