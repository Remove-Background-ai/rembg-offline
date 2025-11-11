import { init } from "./init.js";
import { type ProgressState, type ProgressPhase } from "./progress.js";
import { getCapabilities, type DeviceCapability } from "./capabilities.js";
export type { ProgressState, ProgressPhase, DeviceCapability };
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
export { getCapabilities };
/**
 * Initialize the model (loads it into memory).
 * Can be called explicitly for eager loading, or will be called automatically on first removeBackground().
 *
 * The model will automatically use the best available backend (WebGPU with FP16 > WebGPU with FP32 > WASM).
 * Use getCapabilities() to check what will be used before calling init().
 */
export { init };
/**
 * Remove background from an image URL.
 * - You provide your own file/upload UI.
 * - Call this function with the selected file URL (e.g., an object URL or a web-accessible URL).
 * - Returns a blob URL of the composited transparent image and a small preview URL.
 */
export declare function removeBackground(url: string): Promise<RemoveBackgroundResult>;
