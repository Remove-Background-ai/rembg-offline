import { AutoModel, AutoProcessor, env } from "@huggingface/transformers";
import { onnxProgress } from "./progress.js";
import { getCapabilities } from "./capabilities.js";
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
export async function init(setModelLoaded) {
    patchFetchOnce();
    if (cachedLoad)
        return cachedLoad;
    // transformers.js env – avoid local models, allow browser caches
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    cachedLoad = (async () => {
        const sessionId = onnxProgress.beginNewSession();
        activeSessionId = sessionId;
        try {
            if (setModelLoaded)
                setModelLoaded(false);
            // Get the best available device and precision
            const capability = await getCapabilities();
            const { device, dtype } = capability;
            // Log what we're using
            if (device === 'webgpu' && dtype === 'fp16') {
                console.log("[rembg] ✅ Using WebGPU with FP16 precision (shader-f16 supported)");
            }
            else if (device === 'webgpu' && dtype === 'fp32') {
                console.log("[rembg] ⚠️ Using WebGPU with FP32 precision (shader-f16 not available)");
            }
            else {
                console.log("[rembg] Using WASM backend with FP32");
            }
            const modelOptions = {
                config: { model_type: "custom" },
                device,
                dtype,
            };
            if (device === 'wasm') {
                modelOptions.executionProviders = ['wasm'];
            }
            console.log("[rembg] 🚀 Model initialization:", capability);
            // Load model → after bytes fetched, we transition to "building"
            const model = await AutoModel.from_pretrained("briaai/RMBG-1.4", modelOptions);
            onnxProgress.setBuilding(sessionId);
            const processor = await AutoProcessor.from_pretrained("briaai/RMBG-1.4", {
                config: {
                    do_normalize: true,
                    do_pad: false,
                    do_rescale: true,
                    do_resize: true,
                    image_mean: [0.5, 0.5, 0.5],
                    image_std: [1, 1, 1],
                    resample: 2,
                    rescale_factor: 0.00392156862745098,
                    size: { width: 1024, height: 1024 }
                }
            });
            onnxProgress.setReady(sessionId);
            if (setModelLoaded)
                setModelLoaded(true);
            return { model, processor };
        }
        catch (e) {
            cachedLoad = null;
            onnxProgress.setError(activeSessionId, e?.message || String(e));
            if (setModelLoaded)
                setModelLoaded(false);
            throw e;
        }
    })();
    return cachedLoad;
}
