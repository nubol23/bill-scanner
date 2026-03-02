import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      injectRegister: "auto",
      registerType: "autoUpdate",
      devOptions: {
        enabled: false,
      },
      manifest: {
        name: "Billete Scanner",
        short_name: "Billetes",
        theme_color: "#f9fafb",
        background_color: "#f9fafb",
        display: "standalone",
      },
      workbox: {
        runtimeCaching: [
          {
            urlPattern: ({ request, url }) =>
              request.method === "GET" &&
              url.origin === self.location.origin &&
              url.pathname.startsWith("/models/paddle-ocr/"),
            handler: "CacheFirst",
            options: {
              cacheName: "paddle-models-v1",
              cacheableResponse: {
                statuses: [200],
              },
              expiration: {
                maxEntries: 5,
                maxAgeSeconds: 60 * 60 * 24 * 365,
                purgeOnQuotaError: true,
              },
            },
          },
        ],
      },
    }),
  ],
  base: "/",
  define: {
    Module: "window.Module",
  },
});
