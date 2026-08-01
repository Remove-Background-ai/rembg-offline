import { AutoModel, AutoProcessor, env } from "@huggingface/transformers";
import { onnxProgress } from "./progress.js";
import { getCapabilityLadder, isIOS, type CapabilityOptions, type DeviceCapability } from "./capabilities.js";

// HMR-safe guard for browser environments (library consumers may HMR)
declare global {
  interface Window { __rembg_offline_fetch_patched__?: boolean; }
}

const MODEL_ID = "briaai/RMBG-1.4";
const DEFAULT_PROCESS_SIZE = 1024;
// WASM inference at 1024x1024 exceeds iOS Safari's per-tab memory budget and
// gets the page reloaded by the OS watchdog; halve the resolution there.
const IOS_WASM_PROCESS_SIZE = 512;

// Track the current active init session to correctly attribute progress
let activeSessionId = 0;
let cachedLoad: Promise<LoadedModel> | null = null;
let originalFetch: typeof window.fetch | null = null;

// single-flight + memory cache for ONNX responses (avoid redownloading)
const inflight = new Map<string, Promise<ArrayBuffer>>();
const memCache = new Map<string, ArrayBuffer>();

// Match ONNX files coming from the HF hosting used by transformers.js
const ONNX_PATH_HINT = "/onnx/";

function patchFetchOnce() {
  if (typeof window === "undefined") return; // SSR guard
  if ((window as any).__rembg_offline_fetch_patched__) return;
  (window as any).__rembg_offline_fetch_patched__ = true;

  originalFetch = window.fetch.bind(window);
  window.fetch = async (resource: RequestInfo | URL, init?: RequestInit) => {
    const url = String(resource);
    // Only intercept ONNX model files to track download progress
    if (!url.includes(ONNX_PATH_HINT)) {
      return (originalFetch as any)(resource, init);
    }

    // Serve from memory cache instantly (progress -> near complete)
    if (memCache.has(url)) {
      const buf = memCache.get(url)!;
      onnxProgress.setNetworkProgress(99, activeSessionId);
      return new Response(buf, {
        headers: { "content-type": "application/octet-stream", "content-length": String(buf.byteLength) },
        status: 200
      });
    }

    // If another init is already fetching, stream from its result
    if (inflight.has(url)) {
      const bufPromise = inflight.get(url)!;
      const stream = new ReadableStream<Uint8Array>({
        async start(controller) {
          const buf = await bufPromise;
          controller.enqueue(new Uint8Array(buf));
          controller.close();
        }
      });
      return new Response(stream, { headers: { "content-type": "application/octet-stream" }, status: 200 });
    }

    // First real download for this URL
    onnxProgress.setNetworkProgress(0, activeSessionId);

    const res = await (originalFetch as any)(resource, init);
    if (!res.body) return res;

    const total = Number(res.headers.get("content-length") || 0);
    let loaded = 0;
    const chunks: Uint8Array[] = [];

    let resolveBuf!: (v: ArrayBuffer) => void;
    let rejectBuf!: (e: any) => void;
    const bufPromise = new Promise<ArrayBuffer>((resolve, reject) => {
      resolveBuf = resolve; rejectBuf = reject;
    });
    inflight.set(url, bufPromise);

    const reader = res.body.getReader();
    const tracked = new ReadableStream<Uint8Array>({
      async pull(controller) {
        try {
          const { done, value } = await reader.read();
          if (done) {
            onnxProgress.setNetworkProgress(99, activeSessionId);
            const totalLen = chunks.reduce((s, c) => s + c.byteLength, 0);
            const merged = new Uint8Array(totalLen);
            let off = 0; for (const c of chunks) { merged.set(c, off); off += c.byteLength; }
            const buf = merged.buffer;
            memCache.set(url, buf);
            inflight.delete(url);
            resolveBuf(buf);
            controller.close();
            return;
          }
          if (value) {
            controller.enqueue(value);
            chunks.push(value);
            if (total > 0) {
              loaded += value.byteLength;
              const pct = Math.min(99, Math.floor((loaded / total) * 100));
              onnxProgress.setNetworkProgress(pct, activeSessionId);
            }
          }
        } catch (e) {
          inflight.delete(url);
          onnxProgress.setError(activeSessionId, (e as any)?.message || String(e));
          rejectBuf(e);
          throw e;
        }
      },
      cancel(reason) { try { reader.cancel(reason); } catch {} }
    });

    return new Response(tracked, { status: res.status, statusText: res.statusText, headers: res.headers });
  };
}

// Detect a usable Cache API. Safari private browsing exposes `caches` but
// rejects on open, which would otherwise abort model loading entirely.
async function browserCacheAvailable(): Promise<boolean> {
  try {
    if (typeof caches === "undefined") return false;
    await caches.open("transformers-cache");
    return true;
  } catch {
    return false;
  }
}

export type LoadedModel = {
  model: any;
  processor: any;
  capability: DeviceCapability;
  processSize: number;
};

export type InitOptions = {
  setModelLoaded?: (b: boolean) => void;
  /** Force a specific backend tier instead of auto-detection (disables fallback). */
  capability?: DeviceCapability;
  /** Override the processing resolution (default 1024; 512 on the iOS WASM tier). */
  processSize?: number;
  /** Override the automatic WebGPU support check (see isWebGPUSupported). */
  allowWebGPU?: boolean;
};

// Backend fallback state. `ladderIndex` only ever moves down (toward WASM);
// once a tier has produced a valid mask it is marked verified and kept.
let ladder: DeviceCapability[] = [];
let ladderIndex = 0;
let forcedCapability: DeviceCapability | null = null;
let backendVerified = false;
let processSizeOverride: number | undefined;

function processSizeFor(cap: DeviceCapability): number {
  if (processSizeOverride) return processSizeOverride;
  if (cap.device === 'wasm' && isIOS()) return IOS_WASM_PROCESS_SIZE;
  return DEFAULT_PROCESS_SIZE;
}

async function loadTier(cap: DeviceCapability, sessionId: number): Promise<LoadedModel> {
  const processSize = processSizeFor(cap);
  console.log(`[rembg] Loading model: device=${cap.device} dtype=${cap.dtype} size=${processSize}`);

  const model = await AutoModel.from_pretrained(MODEL_ID, {
    config: { model_type: "custom" },
    device: cap.device,
    dtype: cap.dtype,
  } as any);
  onnxProgress.setBuilding(sessionId);
  const processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    config: {
      do_normalize: true,
      do_pad: false,
      do_rescale: true,
      do_resize: true,
      image_mean: [0.5, 0.5, 0.5],
      image_std: [1, 1, 1],
      resample: 2,
      rescale_factor: 0.00392156862745098,
      size: { width: processSize, height: processSize }
    }
  });

  return { model, processor, capability: cap, processSize };
}

async function disposeLoad(load: Promise<LoadedModel> | null) {
  if (!load) return;
  try {
    const { model } = await load;
    await model?.dispose?.();
  } catch {}
}

/** The tier currently in use (null before the first successful init). */
export function getActiveCapability(): DeviceCapability | null {
  if (forcedCapability) return forcedCapability;
  return ladder[ladderIndex] ?? null;
}

/** Whether the active backend has produced at least one valid mask. */
export function isBackendVerified(): boolean {
  return backendVerified;
}

export function markBackendVerified(): void {
  backendVerified = true;
}

function canDemote(): boolean {
  return !forcedCapability && !backendVerified && ladderIndex + 1 < ladder.length;
}

/**
 * Drop to the next tier in the capability ladder (e.g. webgpu/fp16 ->
 * webgpu/fp32 -> wasm/q8). Used when a backend initializes but then fails or
 * produces garbage at inference time — which is exactly how broken
 * WebGPU-on-WebKit combinations manifest. Returns false when there is nothing
 * left to fall back to, the backend was forced, or it already proved itself.
 */
export async function demoteBackend(): Promise<boolean> {
  if (!canDemote()) return false;
  const previous = cachedLoad;
  cachedLoad = null;
  ladderIndex++;
  console.warn(`[rembg] Falling back to ${ladder[ladderIndex].device}/${ladder[ladderIndex].dtype}`);
  await disposeLoad(previous);
  return true;
}

/** Reset all backend state (mainly for tests). The next init() re-detects. */
export async function resetBackend(): Promise<void> {
  const previous = cachedLoad;
  cachedLoad = null;
  ladder = [];
  ladderIndex = 0;
  forcedCapability = null;
  backendVerified = false;
  await disposeLoad(previous);
}

export async function init(options?: InitOptions | ((b: boolean) => void)): Promise<LoadedModel> {
  const opts: InitOptions = typeof options === "function" ? { setModelLoaded: options } : (options ?? {});
  patchFetchOnce();

  if (opts.capability &&
      (opts.capability.device !== forcedCapability?.device || opts.capability.dtype !== forcedCapability?.dtype)) {
    await resetBackend();
    forcedCapability = opts.capability;
  }
  if (opts.processSize && opts.processSize !== processSizeOverride) {
    const forced = forcedCapability;
    await resetBackend();
    forcedCapability = forced;
    processSizeOverride = opts.processSize;
  }

  if (cachedLoad) return cachedLoad;

  // transformers.js env – avoid local models, allow browser caches
  env.allowLocalModels = false;

  cachedLoad = (async () => {
    const sessionId = onnxProgress.beginNewSession();
    activeSessionId = sessionId;
    try {
      if (opts.setModelLoaded) opts.setModelLoaded(false);

      env.useBrowserCache = await browserCacheAvailable();

      if (forcedCapability) {
        const loaded = await loadTier(forcedCapability, sessionId);
        onnxProgress.setReady(sessionId);
        if (opts.setModelLoaded) opts.setModelLoaded(true);
        return loaded;
      }

      if (ladder.length === 0) {
        const capabilityOptions: CapabilityOptions = { allowWebGPU: opts.allowWebGPU };
        ladder = await getCapabilityLadder(capabilityOptions);
        ladderIndex = 0;
      }

      // Walk the ladder: a tier that fails to even initialize (session
      // creation / shader compilation errors) drops to the next one.
      let lastError: unknown = null;
      while (ladderIndex < ladder.length) {
        const cap = ladder[ladderIndex];
        try {
          const loaded = await loadTier(cap, sessionId);
          onnxProgress.setReady(sessionId);
          if (opts.setModelLoaded) opts.setModelLoaded(true);
          return loaded;
        } catch (e) {
          lastError = e;
          if (ladderIndex + 1 >= ladder.length) break;
          console.warn(`[rembg] ${cap.device}/${cap.dtype} failed to initialize, trying next backend`, e);
          ladderIndex++;
        }
      }
      throw lastError ?? new Error("No usable inference backend");
    } catch (e: any) {
      cachedLoad = null;
      onnxProgress.setError(activeSessionId, e?.message || String(e));
      if (opts.setModelLoaded) opts.setModelLoaded(false);
      throw e;
    }
  })();

  return cachedLoad;
}
