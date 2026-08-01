/**
 * Device and precision capabilities for background removal.
 */
export type DeviceCapability = {
    device: 'webgpu';
    dtype: 'fp16';
} | {
    device: 'webgpu';
    dtype: 'fp32';
} | {
    device: 'wasm';
    dtype: 'q8';
} | {
    device: 'wasm';
    dtype: 'fp32';
};
/** True on WebKit-based browsers (Safari on macOS, and every browser on iOS). */
export declare function isWebKit(): boolean;
/** True on iOS/iPadOS devices, including iPads that report a macOS user agent. */
export declare function isIOS(): boolean;
/**
 * Whether ONNX Runtime's WebGPU backend is expected to work in this browser.
 *
 * transformers.js v3 bundles ORT's JSEP WebGPU backend, which works on
 * Safari 26+ / iOS 26+. transformers.js v4 bundles ORT's newer WebGPU EP,
 * which currently fails on WebKit ("webgpuInit is not a function") — and a
 * failed WebGPU attempt corrupts ORT's WASM backend state in the same page,
 * so on that combination WebGPU must not even be attempted.
 */
export declare function isWebGPUSupported(): boolean;
export type CapabilityOptions = {
    /**
     * Include WebGPU tiers when an adapter is available. Defaults to an
     * automatic check (see isWebGPUSupported) that skips WebGPU on
     * WebKit + transformers.js v4, where ORT's WebGPU EP is known-broken.
     */
    allowWebGPU?: boolean;
};
/**
 * Full fallback ladder, best tier first. The adapter check only proves WebGPU
 * exists — not that ONNX Runtime's shaders compile on it, so init() walks
 * this ladder and inference demotes through it on failure.
 *
 * The WASM tier prefers the quantized model (~44MB): forcing fp32 there means a
 * ~176MB download and a peak memory footprint that gets the tab killed on iOS.
 */
export declare function getCapabilityLadder(options?: CapabilityOptions): Promise<DeviceCapability[]>;
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
export declare function getCapabilities(options?: CapabilityOptions): Promise<DeviceCapability>;
