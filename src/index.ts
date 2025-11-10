import { RawImage } from "@huggingface/transformers";
import { init, forceWASMMode } from "./init";
import { onnxProgress, type ProgressState, type ProgressPhase } from "./progress";

// Public types
export type { ProgressState, ProgressPhase };

export type RemoveBackgroundResult = {
  blobUrl: string;        // full-resolution image (transparent background)
  previewUrl: string;     // small preview (<= 450px)
  width: number;
  height: number;
  processingTimeSeconds: number;
};

// Helper to run compositing in a worker (with OffscreenCanvas) when supported
async function composeOffMainThread(
  bitmap: ImageBitmap,
  alpha: Uint8Array,
  width: number,
  height: number
): Promise<{ mainBlob: Blob; previewBlob: Blob }> {
  return new Promise((resolve, reject) => {
    try {
      // Vite/Webpack-friendly worker import via URL
      // Note: Use .js extension because tsc compiles .ts -> .js but doesn't transform URL strings
      const worker = new Worker(new URL("./rembg-compositor.worker.js", import.meta.url), { type: "module" });
      const cleanup = () => { try { worker.terminate(); } catch {} };
      worker.onmessage = (evt: MessageEvent<any>) => {
        const data = evt.data;
        if (!data) return;
        if (data.type === 'result') {
          cleanup();
          resolve({ mainBlob: data.mainBlob as Blob, previewBlob: data.previewBlob as Blob });
        } else if (data.type === 'error') {
          cleanup();
          reject(new Error(String(data.message)));
        }
      };
      worker.onerror = (err) => { cleanup(); reject(err as any); };
      worker.postMessage(
        {
          type: 'compose',
          width,
          height,
          alphaBuffer: alpha.buffer,
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
 * Force the next initialization to use WASM (disables WebGPU attempt on next call).
 * Useful if the device produces a faulty mask with WebGPU.
 */
export { forceWASMMode };

/**
 * Remove background from an image URL.
 * - You provide your own file/upload UI.
 * - Call this function with the selected file URL (e.g., an object URL or a web-accessible URL).
 * - Returns a blob URL of the composited transparent image and a small preview URL.
 */
export async function removeBackground(url: string): Promise<RemoveBackgroundResult> {
  if (!url) throw new Error("URL is empty");

  const { model, processor } = await init();

  // Load the image (already resized as needed by the app using this library)
  const image = await RawImage.fromURL(url);
  const originalWidth = image.width;
  const originalHeight = image.height;

  const start = performance.now();

  // Inference
  const { pixel_values } = await (processor as any)(image);
  const { output } = await (model as any)({ input: pixel_values });

  // Prepare alpha mask in original size
  const maskRaw = await RawImage.fromTensor(output[0].mul(255).to("uint8"));
  const mask = await maskRaw.resize(originalWidth, originalHeight);
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


