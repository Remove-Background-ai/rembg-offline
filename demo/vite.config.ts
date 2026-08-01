import { defineConfig } from "vite";

export default defineConfig({
  // onnxruntime-web's wasm/worker assets break under esbuild pre-bundling
  optimizeDeps: { exclude: ["@huggingface/transformers"] },
  server: { host: true },
});
