import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        // 后端默认在本机 8000；需要连别的实例时用 VITE_API_TARGET 覆盖，
        // 例如同时开一个演示实例做对照验证：
        //   VITE_API_TARGET=http://127.0.0.1:8130 npm run dev -- --port 5199
        // 注意用 127.0.0.1 而不是 localhost —— Node 会把 localhost 解析成 IPv6 ::1，
        // 而 uvicorn 只监听 IPv4，会造成代理 502。
        target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
