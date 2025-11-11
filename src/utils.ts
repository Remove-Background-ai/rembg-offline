import { env } from "@huggingface/transformers";

/**
 * Detects FP16 (half-precision) compute capability for WebGPU.
 * Returns whether shader-f16 feature is supported and available device features.
 * 
 * This is the reliable way to detect FP16 support - checking for the
 * "shader-f16" feature on the GPU adapter.
 */
export const getPrecisionCapability = async (): Promise<{ hasFP16: boolean; deviceFeatures: string[] }> => {
  try {
    // Check if WebGPU is available
    const nav = navigator as any;
    if (typeof navigator === 'undefined' || !nav.gpu) {
      console.warn("WebGPU not available - no FP16 compute support");
      return { hasFP16: false, deviceFeatures: [] };
    }

    // Try to get adapter from transformers.js env first, otherwise request new one
    //@ts-ignore
    let adapter = env.backends.webgpu?.adapter;
    
    if (!adapter) {
      adapter = await nav.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
    }

    if (!adapter) {
      console.warn("No WebGPU adapter available - falling back to FP32");
      return { hasFP16: false, deviceFeatures: [] };
    }

    // Check for shader-f16 feature - this is the official way to detect FP16 support
    const hasFP16 = adapter.features.has("shader-f16");
    const deviceFeatures = [...adapter.features];
    
    console.log("GPU Precision Capability:", {
      hasFP16,
      features: deviceFeatures
    });

    return { hasFP16, deviceFeatures };
  } catch (error) {
    console.warn("Error detecting FP16 capability:", error);
    return { hasFP16: false, deviceFeatures: [] };
  }
}