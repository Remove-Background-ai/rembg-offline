/**
 * Device and precision capabilities for background removal.
 */
/**
 * Check WebGPU availability and FP16 support.
 * Returns the best available device and precision for background removal.
 *
 * @returns Promise resolving to the available device capability
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
export async function getCapabilities() {
    try {
        // Check if WebGPU is available
        const gpu = globalThis.navigator?.gpu;
        if (!gpu) {
            return { device: 'wasm', dtype: 'fp32' };
        }
        // Request adapter to check features
        const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter) {
            return { device: 'wasm', dtype: 'fp32' };
        }
        // Check for FP16 support
        const hasFP16 = adapter.features.has("shader-f16");
        if (hasFP16) {
            return { device: 'webgpu', dtype: 'fp16' };
        }
        else {
            return { device: 'webgpu', dtype: 'fp32' };
        }
    }
    catch {
        // Any error defaults to WASM
        return { device: 'wasm', dtype: 'fp32' };
    }
}
