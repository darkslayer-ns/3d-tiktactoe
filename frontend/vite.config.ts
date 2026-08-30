import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    allowedHosts: ['.ngrok-free.dev'],
    proxy: {
      '/games': { target: 'http://localhost:8100', ws: true, changeOrigin: true },
      '/admin': { target: 'http://localhost:8100', changeOrigin: true },
    },
  },
})