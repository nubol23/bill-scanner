import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

const LEGACY_CACHE_NAMES = ['paddle-models-v1']
const LEGACY_CACHE_PREFIXES = ['workbox-']

function cleanupLegacyRuntime() {
  const unregisterServiceWorkers =
    'serviceWorker' in navigator
      ? navigator.serviceWorker
          .getRegistrations()
          .then((registrations) =>
            Promise.all(registrations.map((registration) => registration.unregister())),
          )
      : Promise.resolve([])

  const clearLegacyCaches =
    'caches' in window
      ? caches
          .keys()
          .then((cacheNames) =>
            Promise.all(
              cacheNames
                .filter(
                  (cacheName) =>
                    LEGACY_CACHE_NAMES.includes(cacheName) ||
                    LEGACY_CACHE_PREFIXES.some((prefix) => cacheName.startsWith(prefix)),
                )
                .map((cacheName) => caches.delete(cacheName)),
            ),
          )
      : Promise.resolve([])

  void Promise.allSettled([unregisterServiceWorkers, clearLegacyCaches])
}

cleanupLegacyRuntime()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <App />,
)
