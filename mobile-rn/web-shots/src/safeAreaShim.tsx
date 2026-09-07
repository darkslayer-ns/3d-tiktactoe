import React from 'react'
import { View } from 'react-native'

// Web stand-in for react-native-safe-area-context: no notches on the web
// harness, so insets are always zero.

export function SafeAreaProvider({ children }: any) {
  return <>{children}</>
}

export function SafeAreaView({ children, ...rest }: any) {
  return <View {...rest}>{children}</View>
}

export function useSafeAreaInsets() {
  return { top: 0, right: 0, bottom: 0, left: 0 }
}

export function SafeAreaConsumer({ children }: any) {
  return children({ top: 0, right: 0, bottom: 0, left: 0 })
}

export const SafeAreaInsetsContext = React.createContext({
  top: 0,
  right: 0,
  bottom: 0,
  left: 0,
})

export const initialWindowMetrics = {
  insets: { top: 0, right: 0, bottom: 0, left: 0 },
  frame: { x: 0, y: 0, width: 0, height: 0 },
}
