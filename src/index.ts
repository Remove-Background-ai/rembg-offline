import { RawImage } from "@huggingface/transformers";
import { init, demoteBackend, markBackendVerified, getActiveCapability, resetBackend, type InitOptions, type LoadedModel } from "./init.js";
import { onnxProgress, type ProgressState, type ProgressPhase } from "./progress.js";
import { getCapabilities, getCapabilityLadder, isWebGPUSupported, isWebKit, isIOS, type CapabilityOptions, type DeviceCapability } from "./capabilities.js";

// Public types
export type { ProgressState, ProgressPhase, DeviceCapability, CapabilityOptions, InitOptions, LoadedModel };

export type RemoveBackgroundResult = {
  blobUrl: string;        // full-resolution image (transparent background)
  previewUrl: string;     // small preview (<= 450px)
  width: number;
  height: number;
  processingTimeSeconds: number;
};

// An entirely empty mask from an unverified backend is treated as backend
// breakage (broken fp16/WebGPU paths yield NaN -> all-zero masks) rather than
// a legitimately subject-free image, and triggers a fallback.
function isAllZero(a: Uint8Array): boolean {
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== 0) return false;
  }
  return true;
}

// Helper to run compositing in a worker (with OffscreenCanvas) when supported
async function composeOffMainThread(
  bitmap: ImageBitmap,
  alpha: Uint8Array,
  width: number,
  height: number
): Promise<{ mainBlob: Blob; previewBlob: Blob }> {
  return new Promise((resolve, reject) => {
    try {
      // Helper to create a compositor worker with robust fallback
      const createCompositorWorker = (): { worker: Worker; revokeUrl?: string } => {
        // 1) Try bundler-friendly URL (ESM worker). Works in Vite/Webpack when node_modules is processed.
        try {
          const w = new Worker(new URL("./rembg-compositor.worker.js", import.meta.url), { type: "module" });
          return { worker: w };
        } catch (_err) {
          // 2) Fallback to inline classic worker (no imports; pure JS)
          const inline = `
self.onmessage = async (evt) => {
  const data = evt.data;
  if (!data || data.type !== 'compose') return;
  try {
    const { width, height, alphaBuffer, alphaByteOffset, alphaLength, bitmap, previewMax } = data;
    if (!width || !height || !bitmap) throw new Error('Invalid compose arguments');
    const alpha = new Uint8Array(alphaBuffer, alphaByteOffset, alphaLength);
    if (alpha.length !== alphaLength) throw new Error('Alpha length mismatch');
    if (typeof OffscreenCanvas === 'undefined') throw new Error('OffscreenCanvas not supported');

    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('2D context unavailable');
    ctx.drawImage(bitmap, 0, 0);

    const H_CHUNK = 512;
    for (let y0 = 0; y0 < height; y0 += H_CHUNK) {
      const h = Math.min(H_CHUNK, height - y0);
      const strip = ctx.getImageData(0, y0, width, h);
      const dataRGBA = strip.data;
      const start = y0 * width;
      for (let row = 0; row < h; row++) {
        const rowStartAlpha = start + row * width;
        const rowStartRGBA = row * width * 4;
        for (let x = 0; x < width; x++) {
          dataRGBA[rowStartRGBA + x * 4 + 3] = alpha[rowStartAlpha + x];
        }
      }
      ctx.putImageData(strip, 0, y0);
    }

    // Prefer PNG for reliable alpha; validate blob size
    let mainBlob = await canvas.convertToBlob({ type: 'image/png', quality: 0.95 });
    if (!mainBlob || mainBlob.size < 500) {
      // Attempt a secondary encode path if suspiciously small
      mainBlob = await canvas.convertToBlob({ type: 'image/png' });
    }

    // Build preview (PNG)
    const maxSide = Math.max(width, height);
    const scale = Math.min(1, previewMax / maxSide);
    const pW = Math.max(1, Math.round(width * scale));
    const pH = Math.max(1, Math.round(height * scale));
    const pCanvas = new OffscreenCanvas(pW, pH);
    const pCtx = pCanvas.getContext('2d', { willReadFrequently: true });
    if (!pCtx) throw new Error('2D context unavailable (preview)');
    pCtx.drawImage(canvas, 0, 0, pW, pH);
    let previewBlob = await pCanvas.convertToBlob({ type: 'image/png', quality: 0.6 });
    if (!previewBlob || previewBlob.size < 100) {
      previewBlob = await pCanvas.convertToBlob({ type: 'image/png' });
    }

    try { bitmap?.close?.(); } catch {}
    self.postMessage({ type: 'result', mainBlob, previewBlob });
  } catch (err) {
    self.postMessage({ type: 'error', message: err?.message || String(err) });
  }
};
`;
          const blob = new Blob([inline], { type: "text/javascript" });
          const url = URL.createObjectURL(blob);
          const w = new Worker(url); // classic worker
          return { worker: w, revokeUrl: url };
        }
      };

      const { worker, revokeUrl } = createCompositorWorker();
      const cleanup = () => { try { worker.terminate(); } catch {} };
      worker.onmessage = (evt: MessageEvent<any>) => {
        const data = evt.data;
        if (!data) return;
        if (data.type === 'result') {
          cleanup();
          if (revokeUrl) { try { URL.revokeObjectURL(revokeUrl); } catch {} }
          resolve({ mainBlob: data.mainBlob as Blob, previewBlob: data.previewBlob as Blob });
        } else if (data.type === 'error') {
          cleanup();
          if (revokeUrl) { try { URL.revokeObjectURL(revokeUrl); } catch {} }
          reject(new Error(String(data.message)));
        }
      };
      worker.onerror = (err) => {
        cleanup();
        if (revokeUrl) { try { URL.revokeObjectURL(revokeUrl); } catch {} }
        reject(err as any);
      };
      worker.postMessage(
        {
          type: 'compose',
          width,
          height,
          alphaBuffer: alpha.buffer,
          alphaByteOffset: alpha.byteOffset,
          alphaLength: alpha.length,
          bitmap,
          previewMax: 450,
        },
        [alpha.buffer as ArrayBuffer, bitmap as unknown as Transferable]
      );
    } catch (e) {
      reject(e);
    }
  });
}

/**
 * Subscribe to ONNX/model loading progress.
 * Returns an unsubscribe function.
 */
export function subscribeToProgress(listener: (state: ProgressState) => void): () => void {
  return onnxProgress.subscribe(listener);
}

/**
 * Get available device and precision capabilities.
 * Call this to check what backend will be used before initialization.
 * 
 * @returns Promise resolving to device capability (webgpu-fp16, webgpu-fp32, or wasm-fp32)
 * 
 * @example
 * ```typescript
 * const capability = await getCapabilities();
 * console.log(`Using ${capability.device} with ${capability.dtype}`);
 * ```
 */
export { getCapabilities, getCapabilityLadder, isWebGPUSupported, isWebKit, isIOS };

/**
 * Initialize the model (loads it into memory).
 * Can be called explicitly for eager loading, or will be called automatically on first removeBackground().
 *
 * The model will automatically use the best available backend
 * (WebGPU FP16 > WebGPU FP32 > WASM q8 > WASM FP32), falling through the
 * ladder when a tier fails to initialize — which happens on Safari/WebKit
 * versions whose WebGPU implementation rejects ONNX Runtime's shaders.
 * Use getCapabilities() to check what will be tried first.
 */
export { init, getActiveCapability, resetBackend };

/**
 * Remove background from an image URL.
 * - You provide your own file/upload UI.
 * - Call this function with the selected file URL (e.g., an object URL or a web-accessible URL).
 * - Returns a blob URL of the composited transparent image and a small preview URL.
 */
export async function removeBackground(url: string): Promise<RemoveBackgroundResult> {
  if (!url) throw new Error("URL is empty");

  // Load the image (already resized as needed by the app using this library)
  const image = await RawImage.fromURL(url);
  const originalWidth = image.width;
  const originalHeight = image.height;

  const start = performance.now();

  // Inference, with automatic backend demotion: a tier that throws — or
  // returns an all-zero mask before it has ever produced a valid one (the
  // classic silent fp16/WebGPU failure on WebKit) — is swapped for the next
  // tier down the ladder and the inference is retried. Once a tier produces a
  // valid mask it is considered verified and never demoted again.
  let mask: any;
  for (;;) {
    const { model, processor } = await init();
    try {
      const { pixel_values } = await (processor as any)(image);
      const { output } = await (model as any)({ input: pixel_values });

      const maskRaw = await RawImage.fromTensor(output[0].mul(255).to("uint8"));
      if (isAllZero(maskRaw.data as Uint8Array) && await demoteBackend()) {
        continue;
      }
      markBackendVerified();

      // Prepare alpha mask in original size
      mask = await maskRaw.resize(originalWidth, originalHeight);
      if (maskRaw !== mask) {
        // GC hint for the intermediate full-size mask
        (maskRaw as any).data = null;
      }
      break;
    } catch (e) {
      if (await demoteBackend()) continue;
      throw e;
    }
  }

  const alpha = mask.data as Uint8Array;
  if (alpha.length !== originalWidth * originalHeight) {
    throw new Error("Mask size mismatch");
  }

  // Try worker-based compositing; fallback to main thread if needed
  let mainBlob: Blob;
  let previewBlob: Blob;
  try {
    const srcCanvas = image.toCanvas();
    const bitmap = await createImageBitmap(srcCanvas);
    // Help GC
    // @ts-ignore
    srcCanvas.width = 0;
    const res = await composeOffMainThread(bitmap, alpha, originalWidth, originalHeight);
    mainBlob = res.mainBlob;
    previewBlob = res.previewBlob;
  } catch (_e) {
    // Fallback main thread compositing
    const canvas = document.createElement("canvas");
    canvas.width = originalWidth;
    canvas.height = originalHeight;
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    const fallbackSrc = image.toCanvas();
    ctx.drawImage(fallbackSrc, 0, 0);
    // @ts-ignore
    fallbackSrc.width = 0;

    const H_CHUNK = 512;
    for (let y0 = 0; y0 < originalHeight; y0 += H_CHUNK) {
      const h = Math.min(H_CHUNK, originalHeight - y0);
      const strip = ctx.getImageData(0, y0, originalWidth, h);
      const data = strip.data;
      const startAlpha = y0 * originalWidth;
      for (let row = 0; row < h; row++) {
        const rowStartAlpha = startAlpha + row * originalWidth;
        const rowStartRGBA = row * originalWidth * 4;
        for (let x = 0; x < originalWidth; x++) {
          data[rowStartRGBA + x * 4 + 3] = alpha[rowStartAlpha + x];
        }
      }
      ctx.putImageData(strip, 0, y0);
    }

    // Use PNG instead of WebP - better alpha channel support across browsers
    mainBlob = await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))), "image/png", 0.95)
    );

    const previewCanvas = document.createElement("canvas");
    const maxPreview = 450;
    const previewScale = Math.min(1, maxPreview / Math.max(originalWidth, originalHeight));
    previewCanvas.width = Math.max(1, Math.round(originalWidth * previewScale));
    previewCanvas.height = Math.max(1, Math.round(originalHeight * previewScale));
    const pctx = previewCanvas.getContext("2d", { willReadFrequently: true })!;
    pctx.drawImage(canvas, 0, 0, previewCanvas.width, previewCanvas.height);
    previewBlob = await new Promise<Blob>((resolve, reject) =>
      previewCanvas.toBlob((b) => (b ? resolve(b) : reject(new Error("preview toBlob failed"))), "image/png", 0.6)
    );
  }

  // GC hint
  // @ts-ignore
  (mask as any).data = null;

  const blobUrl = URL.createObjectURL(mainBlob);
  const previewUrl = URL.createObjectURL(previewBlob);

  const end = performance.now();
  const processingTimeSeconds = (Math.max(0, end - start) / 1000);

  return {
    blobUrl,
    previewUrl,
    width: originalWidth,
    height: originalHeight,
    processingTimeSeconds
  };
}


