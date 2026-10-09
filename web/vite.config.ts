import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';
import { VitePWA } from 'vite-plugin-pwa';

// GitHub Pages serves the site from a subpath (/OpenPronounce/); the CI
// workflow sets VITE_BASE accordingly, local dev stays at /
export default defineConfig(({ mode }) => ({
  base: process.env.VITE_BASE ?? '/',
  plugins: [
    tailwindcss(),
    VitePWA({
      // the PWA service worker would take over coi-serviceworker's scope and
      // strip the COOP/COEP headers GitHub Pages cannot set natively
      injectRegister: false,
      includeAssets: ['manifest.webmanifest', 'assets/logo.svg', 'assets/favicon.svg'],
      manifest: {
        name: 'OpenPronounce',
        short_name: 'OpenPronounce',
        description: 'Say a sentence, see which sounds went wrong — fully in your browser.',
        start_url: '/',
        display: 'standalone',
        background_color: '#fcfcfb',
        theme_color: '#ea580c',
        icons: [
          { src: '/assets/logo.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any' },
        ],
      },
      workbox: {
        // app shell only; the ~830 MB of models are cached by the libraries
        // themselves (transformers.js Cache API, vits-web OPFS)
        globPatterns: ['**/*.{js,css,html,svg,wasm}'],
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
      },
    }),
  ],
  build: {
    rollupOptions: {
      input: {
        main: new URL('./index.html', import.meta.url).pathname,
        spike: new URL('./spike.html', import.meta.url).pathname,
      },
    },
  },
  server: {
    // COOP/COEP: required for WASM threads (SharedArrayBuffer) in dev;
    // in production these go in public/_headers (Netlify/Cloudflare).
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  worker: {
    format: 'es',
  },
}));
