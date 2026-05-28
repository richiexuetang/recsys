import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy API calls to the Go server in dev so fetches are same-origin.
// The Go server also sets permissive CORS, so direct calls work too.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/catalog": "http://localhost:8080",
      "/score": "http://localhost:8080",
      "/rank": "http://localhost:8080",
      "/health": "http://localhost:8080",
    },
  },
});
