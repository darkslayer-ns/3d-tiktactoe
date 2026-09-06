import { createSfxController, type SfxName, type SfxTrack } from '../audio/sfx'

class FakeTrack implements SfxTrack {
  seeks = 0
  played = 0
  private resolveSeek: (() => void) | null = null
  /** When set, seekTo waits for the test to release it (manual seek). */
  manual = false

  seekTo(_pos: number): Promise<unknown> {
    this.seeks += 1
    if (!this.manual) return Promise.resolve()
    return new Promise((res) => {
      this.resolveSeek = () => res(undefined)
    })
  }

  play(): void {
    this.played += 1
  }

  releaseSeek(): void {
    this.resolveSeek?.()
    this.resolveSeek = null
  }
}

function makeController() {
  const tracks = new Map<SfxName, FakeTrack>()
  const haptics: SfxName[] = []
  const track = (name: SfxName): FakeTrack => {
    let t = tracks.get(name)
    if (!t) {
      t = new FakeTrack()
      tracks.set(name, t)
    }
    return t
  }
  const controller = createSfxController({
    getTrack: track,
    haptic: (n) => haptics.push(n),
  })
  return { controller, track, haptics }
}

const flush = async () => {
  await Promise.resolve()
  await Promise.resolve()
}

describe('createSfxController (overlap policy)', () => {
  it('seeks then plays its track, and fires the haptic', async () => {
    const { controller, track, haptics } = makeController()
    void controller.play('ai')
    await flush()
    expect(track('ai').seeks).toBe(1)
    expect(track('ai').played).toBe(1)
    expect(haptics).toEqual(['ai'])
  })

  it('waits for the rewind to land before playing', async () => {
    const { controller, track } = makeController()
    track('ai').manual = true
    void controller.play('ai')
    await flush()
    expect(track('ai').played).toBe(0) // play deferred until seek resolves
    track('ai').releaseSeek()
    await flush()
    expect(track('ai').played).toBe(1)
  })

  it('overlaps: a new sound does not stop the one already playing', async () => {
    const { controller, track } = makeController()
    void controller.play('ai')
    await flush()
    void controller.play('lose')
    await flush()
    // both tracks were started; the ai track is never paused/stopped (there is
    // no pause in the overlap API — the two just mix)
    expect(track('ai').played).toBe(1)
    expect(track('lose').seeks).toBe(1)
    expect(track('lose').played).toBe(1)
  })

  it('restarts the same sound on rapid retrigger', async () => {
    const { controller, track } = makeController()
    void controller.play('ai')
    await flush()
    void controller.play('ai')
    await flush()
    expect(track('ai').seeks).toBe(2)
    expect(track('ai').played).toBe(2)
  })

  it('no-ops when the track is missing', async () => {
    const { controller } = makeController()
    const missing = createSfxController({
      getTrack: () => null,
      haptic: () => {},
    })
    await expect(missing.play('win')).resolves.toBeUndefined()
    expect(controller).toBeDefined()
  })
})