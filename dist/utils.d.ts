/**
 * Detects FP16 (half-precision) compute capability for WebGPU.
 * Returns whether shader-f16 feature is supported and available device features.
 *
 * This is the reliable way to detect FP16 support - checking for the
 * "shader-f16" feature on the GPU adapter.
 */
export declare const getPrecisionCapability: () => Promise<{
    hasFP16: boolean;
    deviceFeatures: string[];
}>;
