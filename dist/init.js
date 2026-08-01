import { AutoModel, AutoProcessor, env } from "@huggingface/transformers";
import { onnxProgress } from "./progress.js";
import { getCapabilityLadder, isIOS } from "./capabilities.js";
const MODEL_ID = "briaai/RMBG-1.4";
const DEFAULT_PROCESS_SIZE = 1024;
// WASM inference at 1024x1024 exceeds iOS Safari's per-tab memory budget and
// gets the page reloaded by the OS watchdog; halve the resolution there.
const IOS_WASM_PROCESS_SIZE = 512;
// Track the current active init session to correctly attribute progress
let activeSessionId = 0;
let cachedLoad = null;
let originalFetch = null;
// single-flight + memory cache for ONNX responses (avoid redownloading)
const inflight = new Map();
const memCache = new Map();
// Match ONNX files coming from the HF hosting used by transformers.js
const ONNX_PATH_HINT = "/onnx/";
function patchFetchOnce() {
    if (typeof window === "undefined")
        return; // SSR guard
    if (window.__rembg_offline_fetch_patched__)
        return;
    window.__rembg_offline_fetch_patched__ = true;
    originalFetch = window.fetch.bind(window);
    window.fetch = async (resource, init) => {
        const url = String(resource);
        // Only intercept ONNX model files to track download progress
        if (!url.includes(ONNX_PATH_HINT)) {
            return originalFetch(resource, init);
        }
        // Serve from memory cache instantly (progress -> near complete)
        if (memCache.has(url)) {
            const buf = memCache.get(url);
            onnxProgress.setNetworkProgress(99, activeSessionId);
            return new Response(buf, {
                headers: { "content-type": "application/octet-stream", "content-length": String(buf.byteLength) },
                status: 200
            });
        }
        // If another init is already fetching, stream from its result
        if (inflight.has(url)) {
            const bufPromise = inflight.get(url);
            const stream = new ReadableStream({
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
        const res = await originalFetch(resource, init);
        if (!res.body)
            return res;
        const total = Number(res.headers.get("content-length") || 0);
        let loaded = 0;
        const chunks = [];
        let resolveBuf;
        let rejectBuf;
        const bufPromise = new Promise((resolve, reject) => {
            resolveBuf = resolve;
            rejectBuf = reject;
        });
        inflight.set(url, bufPromise);
        const reader = res.body.getReader();
        const tracked = new ReadableStream({
            async pull(controller) {
                try {
                    const { done, value } = await reader.read();
                    if (done) {
                        onnxProgress.setNetworkProgress(99, activeSessionId);
                        const totalLen = chunks.reduce((s, c) => s + c.byteLength, 0);
                        const merged = new Uint8Array(totalLen);
                        let off = 0;
                        for (const c of chunks) {
                            merged.set(c, off);
                            off += c.byteLength;
                        }
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
                }
                catch (e) {
                    inflight.delete(url);
                    onnxProgress.setError(activeSessionId, e?.message || String(e));
                    rejectBuf(e);
                    throw e;
                }
            },
            cancel(reason) { try {
                reader.cancel(reason);
            }
            catch { } }
        });
        return new Response(tracked, { status: res.status, statusText: res.statusText, headers: res.headers });
    };
}
// Detect a usable Cache API. Safari private browsing exposes `caches` but
// rejects on open, which would otherwise abort model loading entirely.
async function browserCacheAvailable() {
    try {
        if (typeof caches === "undefined")
            return false;
        await caches.open("transformers-cache");
        return true;
    }
    catch {
        return false;
    }
}
// Backend fallback state. `ladderIndex` only ever moves down (toward WASM);
// once a tier has produced a valid mask it is marked verified and kept.
let ladder = [];
let ladderIndex = 0;
let forcedCapability = null;
let backendVerified = false;
let processSizeOverride;
function processSizeFor(cap) {
    if (processSizeOverride)
        return processSizeOverride;
    if (cap.device === 'wasm' && isIOS())
        return IOS_WASM_PROCESS_SIZE;
    return DEFAULT_PROCESS_SIZE;
}
async function loadTier(cap, sessionId) {
    const processSize = processSizeFor(cap);
    console.log(`[rembg] Loading model: device=${cap.device} dtype=${cap.dtype} size=${processSize}`);
    const model = await AutoModel.from_pretrained(MODEL_ID, {
        config: { model_type: "custom" },
        device: cap.device,
        dtype: cap.dtype,
    });
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
async function disposeLoad(load) {
    if (!load)
        return;
    try {
        const { model } = await load;
        await model?.dispose?.();
    }
    catch { }
}
/** The tier currently in use (null before the first successful init). */
export function getActiveCapability() {
    if (forcedCapability)
        return forcedCapability;
    return ladder[ladderIndex] ?? null;
}
/** Whether the active backend has produced at least one valid mask. */
export function isBackendVerified() {
    return backendVerified;
}
export function markBackendVerified() {
    backendVerified = true;
}
function canDemote() {
    return !forcedCapability && !backendVerified && ladderIndex + 1 < ladder.length;
}
/**
 * Drop to the next tier in the capability ladder (e.g. webgpu/fp16 ->
 * webgpu/fp32 -> wasm/q8). Used when a backend initializes but then fails or
 * produces garbage at inference time — which is exactly how broken
 * WebGPU-on-WebKit combinations manifest. Returns false when there is nothing
 * left to fall back to, the backend was forced, or it already proved itself.
 */
export async function demoteBackend() {
    if (!canDemote())
        return false;
    const previous = cachedLoad;
    cachedLoad = null;
    ladderIndex++;
    console.warn(`[rembg] Falling back to ${ladder[ladderIndex].device}/${ladder[ladderIndex].dtype}`);
    await disposeLoad(previous);
    return true;
}
/** Reset all backend state (mainly for tests). The next init() re-detects. */
export async function resetBackend() {
    const previous = cachedLoad;
    cachedLoad = null;
    ladder = [];
    ladderIndex = 0;
    forcedCapability = null;
    backendVerified = false;
    await disposeLoad(previous);
}
export async function init(options) {
    const opts = typeof options === "function" ? { setModelLoaded: options } : (options ?? {});
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
    if (cachedLoad)
        return cachedLoad;
    // transformers.js env – avoid local models, allow browser caches
    env.allowLocalModels = false;
    cachedLoad = (async () => {
        const sessionId = onnxProgress.beginNewSession();
        activeSessionId = sessionId;
        try {
            if (opts.setModelLoaded)
                opts.setModelLoaded(false);
            env.useBrowserCache = await browserCacheAvailable();
            if (forcedCapability) {
                const loaded = await loadTier(forcedCapability, sessionId);
                onnxProgress.setReady(sessionId);
                if (opts.setModelLoaded)
                    opts.setModelLoaded(true);
                return loaded;
            }
            if (ladder.length === 0) {
                const capabilityOptions = { allowWebGPU: opts.allowWebGPU };
                ladder = await getCapabilityLadder(capabilityOptions);
                ladderIndex = 0;
            }
            // Walk the ladder: a tier that fails to even initialize (session
            // creation / shader compilation errors) drops to the next one.
            let lastError = null;
            while (ladderIndex < ladder.length) {
                const cap = ladder[ladderIndex];
                try {
                    const loaded = await loadTier(cap, sessionId);
                    onnxProgress.setReady(sessionId);
                    if (opts.setModelLoaded)
                        opts.setModelLoaded(true);
                    return loaded;
                }
                catch (e) {
                    lastError = e;
                    if (ladderIndex + 1 >= ladder.length)
                        break;
                    console.warn(`[rembg] ${cap.device}/${cap.dtype} failed to initialize, trying next backend`, e);
                    ladderIndex++;
                }
            }
            throw lastError ?? new Error("No usable inference backend");
        }
        catch (e) {
            cachedLoad = null;
            onnxProgress.setError(activeSessionId, e?.message || String(e));
            if (opts.setModelLoaded)
                opts.setModelLoaded(false);
            throw e;
        }
    })();
    return cachedLoad;
}
