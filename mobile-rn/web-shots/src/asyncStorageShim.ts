/**
 * localStorage-backed AsyncStorage for the web harness. The real app's
 * persistence (opponent memory / welcome flag) works identically here.
 */
const store = new Map<string, string>()

export default {
  async getItem(k: string): Promise<string | null> {
    return store.get(k) ?? null
  },
  async setItem(k: string, v: string): Promise<void> {
    store.set(k, v)
  },
  async removeItem(k: string): Promise<void> {
    store.delete(k)
  },
  async multiGet(keys: string[]): Promise<Array<[string, string | null]>> {
    return keys.map((k) => [k, store.get(k) ?? null])
  },
  async multiSet(kv: Array<[string, string]>): Promise<void> {
    kv.forEach(([k, v]) => store.set(k, v))
  },
  async multiRemove(keys: string[]): Promise<void> {
    keys.forEach((k) => store.delete(k))
  },
}