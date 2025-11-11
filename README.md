# rembg-webgpu

Offline background removal for the browser using the `rembg.com`'s Distilled model via `@huggingface/transformers`, with a simple API and downloadable ONNX progress you can hook into for your own loader UI.
- Advanced monkey patching is used internally to reliably capture and display download progress across different browser environments and fetch implementations
- No UI included. You build your own file-picker/design.
- Call one async function; get back a blob URL of the transparent image and a small preview URL.
- Exported progress API lets you render your own installer-style loader (download/build/ready).

## Why rembbg-webgpu is different?

Most browser-based background removers are just thin wrappers around demo code.
We went deeper.

rembg-webgpu started from the foundations of @huggingface/transformers, then got rebuilt and optimized into something actually usable for production-like environments.

## Install

This package expects `@huggingface/transformers` to be available (peer dependency).

```bash
npm i rembg-webgpu @huggingface/transformers
```

Your bundler must support web workers via `new URL('./worker.ts', import.meta.url)` (Vite, Webpack 5, etc.).  
Modern browsers only.

## API

```ts
import { removeBackground, subscribeToProgress, forceWASMMode } from 'rembg-webgpu';

// Optional: subscribe to ONNX download/build progress to show a loader
const unsubscribe = subscribeToProgress(({ phase, progress }) => {
  // phase: 'idle' | 'downloading' | 'building' | 'ready' | 'error'
  // progress: 0..100 (ready sets to 100)
  // Render your loader based on this state
  console.log(phase, progress);
});

// Example: using an <input type="file">
async function handleFile(file: File) {
  // Create an object URL (or provide any web-accessible URL)
  const url = URL.createObjectURL(file);
  try {
    const result = await removeBackground(url);
    // result.blobUrl (full res, png), result.previewUrl (small preview, png), result.width/height
    const img = document.querySelector('#result') as HTMLImageElement;
    img.src = result.blobUrl;
  } finally {
    // Revoke your input object URL when safe
    // URL.revokeObjectURL(url);
  }
}

// Optional: if you detect suspicious outputs on this device, call this
// before the next operation to force WASM (skip WebGPU)
forceWASMMode();
```

### Types

```ts
type ProgressPhase = 'idle' | 'downloading' | 'building' | 'ready' | 'error';

type ProgressState = {
  phase: ProgressPhase;
  progress: number;      // 0..100
  errorMsg?: string;
  sessionId: number;
};

type RemoveBackgroundResult = {
  blobUrl: string;       // full-resolution, transparent
  previewUrl: string;    // <=450px preview
  width: number;
  height: number;
  processingTimeSeconds: number;
};
```

## Notes

- The first call downloads and initializes the model; subsequent calls are instant from cache.
- The library attempts to use WebGPU when available, otherwise falls back to WASM.
- You control input resizing before calling `removeBackground(url)` if you need to cap large dimensions for stability/performance.
- Blobs are not automatically revoked. Revoke when you no longer need the URLs to avoid memory leaks.


