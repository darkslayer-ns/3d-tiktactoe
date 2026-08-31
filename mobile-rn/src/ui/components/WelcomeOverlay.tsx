/**
 * First-launch welcome: brand + a "Start playing" button in a card anchored to
 * the bottom of the screen, so the AI-vs-AI demo stays fully visible above it.
 */

import { Image, Pressable, StyleSheet, Text, View } from 'react-native'
import { Theme, fontSize, radius, spacing } from '../theme'

interface WelcomeOverlayProps {
  onStart: () => void
  buttonLabel?: string
}

export function WelcomeOverlay({ onStart, buttonLabel = 'Start playing' }: WelcomeOverlayProps) {
  return (
    <View style={styles.overlay}>
      <View style={styles.backdrop} pointerEvents="none" />
      <View style={styles.card}>
        <Image
          source={require('../../../assets/isocube_logo.png')}
          style={styles.logo}
          resizeMode="contain"
        />

        <View style={styles.guide}>
          <View style={styles.guideRow}>
            <Text style={styles.guideBullet}>1</Text>
            <Text style={styles.guideText}>Tap a cell to select it, then Place to drop your mark.</Text>
          </View>
          <View style={styles.guideRow}>
            <Text style={styles.guideBullet}>2</Text>
            <Text style={styles.guideText}>
              Fill a straight line — any direction, including depth — to win.
            </Text>
          </View>
          <View style={styles.guideRow}>
            <Text style={styles.guideBullet}>3</Text>
            <Text style={styles.guideText}>
              Choose your side, difficulty and board size in the menu.
            </Text>
          </View>
        </View>

        <Pressable
          onPress={onStart}
          style={({ pressed }) => [styles.startBtn, pressed && styles.startBtnPressed]}
        >
          {({ pressed }) => (
            <Text style={[styles.startBtnText, pressed && styles.startBtnTextPressed]}>
              {buttonLabel}
            </Text>
          )}
        </Pressable>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'flex-end',
  },
  backdrop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(2, 6, 23, 0.25)',
  },
  card: {
    alignItems: 'center',
    paddingHorizontal: spacing(6),
    paddingTop: spacing(5),
    paddingBottom: spacing(8),
    borderTopLeftRadius: radius(6),
    borderTopRightRadius: radius(6),
    borderWidth: 1,
    borderColor: 'rgba(34, 211, 238, 0.3)',
    backgroundColor: 'rgba(2, 6, 23, 0.88)',
  },
  logo: {
    width: 260,
    height: 100,
  },
  tagline: {
    color: Theme.cyan,
    fontSize: fontSize(13),
    fontWeight: '700',
    letterSpacing: 1,
    marginTop: spacing(2),
    marginBottom: spacing(4),
  },
  guide: {
    alignSelf: 'stretch',
    marginBottom: spacing(5),
  },
  guideRow: {
    flexDirection: 'row',
    gap: spacing(2),
    marginBottom: spacing(2),
  },
  guideBullet: {
    color: Theme.cyan,
    fontSize: fontSize(10),
    lineHeight: fontSize(15),
  },
  guideText: {
    flex: 1,
    color: Theme.text,
    fontSize: fontSize(13),
    lineHeight: fontSize(18),
    opacity: 0.9,
  },
  startBtn: {
    paddingVertical: spacing(3.5),
    paddingHorizontal: spacing(10),
    borderRadius: radius(4),
    borderWidth: 2,
    borderColor: Theme.cyan,
    alignItems: 'center',
    shadowColor: Theme.cyan,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
  },
  startBtnPressed: {
    backgroundColor: Theme.cyan,
  },
  startBtnText: {
    color: Theme.cyan,
    fontSize: fontSize(15),
    fontWeight: '800',
    letterSpacing: 2,
    textTransform: 'uppercase',
  },
  startBtnTextPressed: {
    color: Theme.bg,
  },
})