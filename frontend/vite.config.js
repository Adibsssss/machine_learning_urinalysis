import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/predict": "http://localhost:5000",
      "/metadata": "http://localhost:5000",
      "/eda":      "http://localhost:5000",
      "/eda_list": "http://localhost:5000",
    },
  },
});
