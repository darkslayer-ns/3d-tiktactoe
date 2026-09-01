import type { ReactNode } from 'react'

// No-op gesture shim — screenshots are static, so pan/pinch are disabled.
// GestureDetector renders its children directly; Gesture.* are chainable no-ops.

function chainable(): any {
  const self: any = {}
  const noop = () => self
  for (const m of ['maxPointers', 'onStart', 'onUpdate', 'onEnd', 'onFinalize', 'enabled', 'runOnJS'])
    self[m] = noop
  return self
}

export const GestureDetector = ({ children }: { children: ReactNode }) => <>{children}</>

export const GestureHandlerRootView = ({
  children,
  style,
}: {
  children: ReactNode
  style?: any
}) => <div style={style}>{children}</div>

export const Gesture = {
  Pan: chainable,
  Pinch: chainable,
  Tap: chainable,
  LongPress: chainable,
  Simultaneous: chainable,
  Exclusive: chainable,
  Race: chainable,
}