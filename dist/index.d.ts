import { init, getActiveCapability, resetBackend, type InitOptions, type LoadedModel } from "./init.js";
import { type ProgressState, type ProgressPhase } from "./progress.js";
import { getCapabilities, getCapabilityLadder, isWebGPUSupported, isWebKit, isIOS, type CapabilityOptions, type DeviceCapability } from "./capabilities.js";
export type { ProgressState, ProgressPhase, DeviceCapability, CapabilityOptions, InitOptions, LoadedModel };
export type RemoveBackgroundResult = {
    blobUrl: string;
    previewUrl: string;
    width: number;
    height: number;
    processingTimeSeconds: number;
};
/**
 * Subscribe to ONNX/model loading progress.
 * Returns an unsubscribe function.
 */
export declare function subscribeToProgress(listener: (state: ProgressState) => void): () => void;
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
export declare function removeBackground(url: string): Promise<RemoveBackgroundResult>;
