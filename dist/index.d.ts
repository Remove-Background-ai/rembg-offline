import { forceWASMMode } from "./init";
import { type ProgressState, type ProgressPhase } from "./progress";
export type { ProgressState, ProgressPhase };
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
export declare function removeBackground(url: string): Promise<RemoveBackgroundResult>;
