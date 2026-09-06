/**
 * No-op web shim for `expo-haptics` — the screenshot harness runs in a
 * browser (react-native-web) where the native haptics module doesn't exist.
 */

export enum ImpactFeedbackStyle {
  Light = 'light',
  Medium = 'medium',
  Heavy = 'heavy',
}

export enum NotificationFeedbackType {
  Success = 'success',
  Warning = 'warning',
  Error = 'error',
}

export async function selectionAsync(): Promise<void> {}
export async function impactAsync(): Promise<void> {}
export async function notificationAsync(): Promise<void> {}