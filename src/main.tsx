import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

const LEGACY_CACHE_NAMES = ['paddle-models-v1']
const LEGACY_CACHE_PREFIXES = ['workbox-']

async function cleanupLegacyRuntime() {
  if ('serviceWorker' in navigator) {
    try {
      const registrations = await navigator.serviceWorker.getRegistrations()
      await Promise.all(registrations.map((registration) => registration.unregister()))
    } catch {
      // Ignore cleanup failures; this is a best-effort migration path.
    }
  }

  if ('caches' in window) {
    try {
      const cacheNames = await caches.keys()
      await Promise.all(
        cacheNames
          .filter(
            (cacheName) =>
              LEGACY_CACHE_NAMES.includes(cacheName) ||
              LEGACY_CACHE_PREFIXES.some((prefix) => cacheName.startsWith(prefix)),
          )
          .map((cacheName) => caches.delete(cacheName)),
      )
    } catch {
      // Ignore cleanup failures; this is a best-effort migration path.
    }
  }
}

function scheduleLegacyRuntimeCleanup() {
  void cleanupLegacyRuntime()

  window.addEventListener(
    'load',
    () => {
      void cleanupLegacyRuntime()
    },
    { once: true },
  )
}

scheduleLegacyRuntimeCleanup()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <App />,
)
