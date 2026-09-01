import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// Web harness that renders the REAL mobile-rn UI components (react-native-web)
// for Playwright App-Store screenshot capture. Aliases swap RN-native modules
// for web equivalents; gestures/storage are no-op/local shims.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      'react-native': 'react-native-web',
      '@react-three/fiber/native': '@react-three/fiber',
      'react-native-gesture-handler': path.resolve(__dirname, 'src/gestureShim.tsx'),
      '@react-native-async-storage/async-storage': path.resolve(
        __dirname,
        'src/asyncStorageShim.ts',
      ),
    },
  },
  server: { port: 5199, host: '127.0.0.1' },
  optimizeDeps: { include: ['three', 'react-native-web', 'react-dom'] },
})