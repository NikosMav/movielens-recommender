import react from '@vitejs/plugin-react';
import {defineConfig} from 'vite';

const proxy = {
  '/v1': {
    target: process.env.MOVIELENS_API_URL ?? 'http://127.0.0.1:8000',
    changeOrigin: true,
  },
};

export default defineConfig({
  plugins: [react()],
  server: {proxy},
  preview: {proxy},
});
