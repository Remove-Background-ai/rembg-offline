/**
 * Device and precision capabilities for background removal.
 */
import { env } from "@huggingface/transformers";
/** True on WebKit-based browsers (Safari on macOS, and every browser on iOS). */
export function isWebKit() {
    if (typeof navigator === 'undefined')
        return false;
    const ua = navigator.userAgent || '';
    return /AppleWebKit/.test(ua) && !/Chrome|Chromium|Edg\//.test(ua);
}
/** True on iOS/iPadOS devices, including iPads that report a macOS user agent. */
export function isIOS() {
    if (typeof navigator === 'undefined')
        return false;
    const ua = navigator.userAgent || '';
    if (/iPhone|iPad|iPod/.test(ua))
        return true;
    // iPadOS 13+ masquerades as macOS but exposes multi-touch
    return /Macintosh/.test(ua) && (navigator.maxTouchPoints || 0) > 1;
}
function transformersMajorVersion() {
    const v = env?.version;
    const major = typeof v === "string" ? parseInt(v.split(".")[0], 10) : NaN;
    return Number.isFinite(major) ? major : 0;
}
/**
 * Whether ONNX Runtime's WebGPU backend is expected to work in this browser.
 *
 * transformers.js v3 bundles ORT's JSEP WebGPU backend, which works on
 * Safari 26+ / iOS 26+. transformers.js v4 bundles ORT's newer WebGPU EP,
 * which currently fails on WebKit ("webgpuInit is not a function") — and a
 * failed WebGPU attempt corrupts ORT's WASM backend state in the same page,
 * so on that combination WebGPU must not even be attempted.
 */
export function isWebGPUSupported() {
    return !(isWebKit() && transformersMajorVersion() >= 4);
}
/**
 * Full fallback ladder, best tier first. The adapter check only proves WebGPU
 * exists — not that ONNX Runtime's shaders compile on it, so init() walks
 * this ladder and inference demotes through it on failure.
 *
 * The WASM tier prefers the quantized model (~44MB): forcing fp32 there means a
 * ~176MB download and a peak memory footprint that gets the tab killed on iOS.
 */
export async function getCapabilityLadder(options) {
    const wasmTiers = [
        { device: 'wasm', dtype: 'q8' },
        { device: 'wasm', dtype: 'fp32' },
    ];
    try {
        const allowWebGPU = options?.allowWebGPU ?? isWebGPUSupported();
        if (!allowWebGPU)
            return wasmTiers;
        const gpu = globalThis.navigator?.gpu;
        if (!gpu)
            return wasmTiers;
        const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter)
            return wasmTiers;
        const hasFP16 = adapter.features.has("shader-f16");
        const gpuTiers = hasFP16
            ? [{ device: 'webgpu', dtype: 'fp16' }, { device: 'webgpu', dtype: 'fp32' }]
            : [{ device: 'webgpu', dtype: 'fp32' }];
        return [...gpuTiers, ...wasmTiers];
    }
    catch {
        return wasmTiers;
    }
}
/**
 * Check WebGPU availability and FP16 support.
 * Returns the best available device and precision for background removal.
 * Note: this is the tier init() tries first; if it fails at runtime the
 * library automatically falls back to the next tier in getCapabilityLadder().
 *
 * @returns Promise resolving to the best available device capability
 *
 * @example
 * ```typescript
 * const capability = await getCapabilities();
 *
 * if (capability.device === 'webgpu' && capability.dtype === 'fp16') {
 *   console.log('Best performance available: WebGPU with FP16');
 * } else if (capability.device === 'webgpu') {
 *   console.log('Good performance: WebGPU with FP32');
 * } else {
 *   console.log('Fallback: WASM backend');
 * }
 * ```
 */
export async function getCapabilities(options) {
    const ladder = await getCapabilityLadder(options);
    return ladder[0];
}
