import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'fs'

// Serve data/latest/ directly from the repo root during dev,
// so the dashboard always reads the latest files without a manual copy step.
const repoDataDir = path.resolve(__dirname, '../../data/latest')

function serveRepoData() {
  return {
    name: 'serve-repo-data',
    configureServer(server: any) {
      server.middlewares.use((req: any, res: any, next: any) => {
        const prefix = '/data/latest/'
        if (!req.url?.startsWith(prefix)) return next()
        const file = path.join(repoDataDir, req.url.slice(prefix.length).split('?')[0])
        if (fs.existsSync(file) && fs.statSync(file).isFile()) {
          res.setHeader('Content-Type',
            file.endsWith('.json') ? 'application/json' :
            file.endsWith('.csv')  ? 'text/csv' : 'application/octet-stream'
          )
          fs.createReadStream(file).pipe(res)
        } else {
          next()
        }
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), serveRepoData()],
  base: './',
  server: {
    proxy: {
      '/api': 'http://localhost:5050',
    },
  },
  test: {
    environment: 'node',
  },
})
