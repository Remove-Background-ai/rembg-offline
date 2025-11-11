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
    dtype: 'fp32';
};
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
export declare function getCapabilities(): Promise<DeviceCapability>;
