import { type DeviceCapability } from "./capabilities.js";
declare global {
    interface Window {
        __rembg_offline_fetch_patched__?: boolean;
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
/** The tier currently in use (null before the first successful init). */
export declare function getActiveCapability(): DeviceCapability | null;
/** Whether the active backend has produced at least one valid mask. */
export declare function isBackendVerified(): boolean;
export declare function markBackendVerified(): void;
/**
 * Drop to the next tier in the capability ladder (e.g. webgpu/fp16 ->
 * webgpu/fp32 -> wasm/q8). Used when a backend initializes but then fails or
 * produces garbage at inference time — which is exactly how broken
 * WebGPU-on-WebKit combinations manifest. Returns false when there is nothing
 * left to fall back to, the backend was forced, or it already proved itself.
 */
export declare function demoteBackend(): Promise<boolean>;
/** Reset all backend state (mainly for tests). The next init() re-detects. */
export declare function resetBackend(): Promise<void>;
export declare function init(options?: InitOptions | ((b: boolean) => void)): Promise<LoadedModel>;
