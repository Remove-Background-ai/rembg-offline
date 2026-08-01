# rembg-webgpu
Blazing fast and Robust Background removal for the Web.

> This is [popjam-io's fork](https://github.com/popjam-io/rembg-webgpu) of
> [Remove-Background-ai/rembg-webgpu](https://github.com/Remove-Background-ai/rembg-webgpu)
> with Safari (WebKit) and mobile iOS support. See
> [Safari / iOS support](#safari--ios-support) for details.

**[🚀 Try Live Demo](https://www.rembg.com/en/free-background-remover)** – See it in action with your own images

## Benchmark

Performance benchmarks on M1 MacBook Pro (WebGPU enabled):

| Resolution | Total Time |
|------------|------------|
| 1000x1000  | **0.73s**  |
| 1024×1536  | **0.95s**  |
| 3000×3000  | **1.40s**  |
| 5203×7800  | **3.05s**  |

*Note: First-time initialization adds delay for model download and compilation (cached thereafter). WASM fallback is approximately 3-5× slower than WebGPU.*

**Have different hardware?** We'd love to see benchmarks from your device! Submit a PR with your results (include device specs, browser, and whether WebGPU or WASM was used).

# What is it?

**rembg-webgpu** is a production-ready, client-side background removal library that runs entirely in the browser. Built on rembg.com's distilled AI model and powered by `@huggingface/transformers`, it delivers state-of-the-art segmentation without server dependencies or privacy compromises.

**Core Features:**
- **Intelligent Backend Selection** – Automatically detects and uses the best available backend:
  - WebGPU with FP16 (shader-f16) for maximum performance
  - WebGPU with FP32 fallback if FP16 unavailable
  - WASM with the quantized model (q8, ~44MB) as universal fallback
  - WASM with FP32 as last resort
- **Runtime Fallback** – A backend that fails to initialize, throws during
  inference, or silently produces an empty mask (the classic broken-fp16
  signature) is automatically demoted to the next tier — no hard failures
- **Runtime Capability Detection** – Query device capabilities before initialization via `getCapabilities()`
- **Zero Server Dependency** – Complete offline processing; your users' images never leave their device
- **Granular Progress Tracking** – Advanced hooks for download/building/ready phases with percentage progress
- **Advanced Optimization** – OffscreenCanvas worker-based compositing with automatic main-thread fallback
- **Smart Caching** – Memory + browser cache for instant subsequent loads
- **Automatic Preview Generation** – Returns both full-resolution and optimized preview URLs
- **Headless by Design** – No opinionated UI; bring your own interface and workflows  
- **TypeScript Native** – Full type safety with exported types for all APIs

## Why rembg-webgpu is Different

Unlike most browser-based background removal solutions that are merely thin wrappers around demo code, **rembg-webgpu** was engineered from the ground up for production environments.

We started with `@huggingface/transformers` as a foundation, then extensively rebuilt and optimized the entire pipeline with:
- Custom fetch interception for granular download progress tracking
- Intelligent device capability detection and automatic backend selection
- Worker-based compositing architecture to keep the main thread responsive
- Memory-efficient chunked processing for large images
- Sophisticated caching strategies across memory and browser storage

The result is a library that doesn't just work in demos—it scales to real-world applications with thousands of users.

## Install


```bash
npm i rembg-webgpu
```

Your bundler must support web workers via `new URL('./worker.ts', import.meta.url)` (Vite, Webpack 5, etc.).  
Modern browsers only.

## Sample code

```ts
import { removeBackground, subscribeToProgress, getCapabilities } from 'rembg-webgpu';

// Optional: Check device capabilities before initialization
const capability = await getCapabilities();
console.log(`Backend: ${capability.device}, Precision: ${capability.dtype}`);
// Possible results:
// - { device: 'webgpu', dtype: 'fp16' } - Best performance
// - { device: 'webgpu', dtype: 'fp32' } - Good performance
// - { device: 'wasm', dtype: 'q8' }     - Universal fallback (quantized, ~44MB)
// - { device: 'wasm', dtype: 'fp32' }   - Last resort

// Optional: Subscribe to ONNX download/build progress to show a loader
const unsubscribe = subscribeToProgress(({ phase, progress }) => {
  // phase: 'idle' | 'downloading' | 'building' | 'ready' | 'error'
  // progress: 0..100 (ready sets to 100)
  console.log(`${phase}: ${progress}%`);
});

// Remove background from an image
const result = await removeBackground(imageUrl);

// Clean up when done
unsubscribe();
```
## Full Documentation & walkthrough guide

[rembg.com's blog](https://www.rembg.com/en/blog/remove-backgrounds-browser-rembg-webgpu)

## Safari / iOS support

Safari 26+ (macOS, iOS, iPadOS) ships WebGPU, and this fork runs the WebGPU
FP16 path on it. Older Safari (≤ 18 / iOS ≤ 18) has no WebGPU and uses the
WASM fallback. What makes this work:

- **`@huggingface/transformers` v3.8.x is the recommended peer** for Safari.
  v3 bundles ONNX Runtime's JSEP WebGPU backend, which works on WebKit.
  v4 bundles ORT's newer WebGPU EP, which currently fails on WebKit
  (`webgpuInit is not a function`) — and that failure corrupts ORT's WASM
  backend in the same page. When v4 + WebKit is detected, this library
  automatically skips WebGPU and goes straight to WASM (override with
  `init({ allowWebGPU: true })` once ORT fixes WebKit support).
- **WASM tier uses the quantized model** (q8, ~44MB) instead of fp32
  (~176MB). On iOS the fp32 download + inference memory exceeded Safari's
  per-tab budget and got the page killed and reloaded by the OS watchdog.
- **iOS WASM runs at 512×512** (instead of 1024×1024) to stay inside the
  memory budget on older iPhones. Override with `init({ processSize: 1024 })`.
- **Runtime demotion** – any tier that fails to initialize, throws during
  inference, or returns an all-zero mask before ever producing a valid one is
  demoted down the ladder automatically.
- Safari private browsing (no usable Cache API) no longer breaks model
  loading; the model is simply re-downloaded per session.

Verified via Playwright WebKit 26.5 (desktop + iPhone emulation): WebGPU FP16
inference passes on transformers.js v3.8.1, and WASM q8 passes on both v3 and
v4. Run the matrix yourself with `npm run test:browsers`, or try it
interactively with `npm run demo` (use `?device=wasm&dtype=q8` to force a tier).

## Technical Details

**Backend Selection**
- Automatically detects WebGPU support and FP16 (shader-f16) capability
- Falls back gracefully: WebGPU FP16 → WebGPU FP32 → WASM q8 → WASM FP32
- `getCapabilityLadder()` returns the full ladder; `getActiveCapability()`
  reports the tier actually in use after init

**Performance Optimizations**
- First call downloads and initializes the model on initial run-up; subsequent calls use memory + browser cache
- Worker-based OffscreenCanvas compositing offloads processing from main thread
- Chunked image processing (512px strips) prevents memory spikes on large images
- Automatic preview generation (≤450px) for instant UI feedback


**Resource Management**
- You control input image sizing before calling `removeBackground(url)` for optimal performance
- Blob URLs are not automatically revoked—call `URL.revokeObjectURL()` when done to prevent memory leaks
- Model weights (~40-50MB) cached in browser after first download

## Roadmap

- [x] WebGPU acceleration with FP16/FP32 precision detection
- [x] Automatic WASM fallback
- [x] Runtime device capability detection API
- [x] Granular progress tracking for model downloads
- [x] OffscreenCanvas worker-based compositing
- [x] Memory + browser caching
- [x] Offline-first architecture
- [x] Full TypeScript support
- [x] Safari (WebKit) support incl. WebGPU on Safari 26+
- [x] Mobile-optimized version (quantized WASM tier, iOS memory guards)
- [ ] Native batch processing API
- [ ] Custom model support with zero-config

## Attribution

Background Removal Library provided by [www.rembg.com](https://www.rembg.com)

## License

This project is licensed under the RemBG Attribution License (MIT-Compatible). See the [LICENSE](LICENSE) file for details.


