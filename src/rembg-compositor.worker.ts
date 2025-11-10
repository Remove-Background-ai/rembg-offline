// Worker to composite RGBA using provided alpha channel and generate preview
// Receives: { type: 'compose', width, height, alphaBuffer, alphaLength, bitmap, previewMax }

export type ComposeMessage =
  | {
      type: 'compose';
      width: number;
      height: number;
      alphaBuffer: ArrayBuffer;
      alphaByteOffset: number;
      alphaLength: number;
      // In workers with OffscreenCanvas support, ImageBitmap is transferable
      bitmap: ImageBitmap;
      previewMax: number;
    };

export type ComposeResultMessage =
  | {
      type: 'result';
      mainBlob: Blob;
      previewBlob: Blob;
    }
  | {
      type: 'error';
      message: string;
    };

self.onmessage = async (evt: MessageEvent<ComposeMessage>) => {
  const data = evt.data;
  if (!data || data.type !== 'compose') return;
  try {
    const { width, height, alphaBuffer, alphaByteOffset, alphaLength, bitmap, previewMax } = data;
    if (!width || !height || !bitmap) throw new Error('Invalid compose arguments');
    const alpha = new Uint8Array(alphaBuffer, alphaByteOffset, alphaLength);
    if (alpha.length !== alphaLength) throw new Error('Alpha length mismatch');

    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(bitmap, 0, 0);

    const H_CHUNK = 512;
    for (let y0 = 0; y0 < height; y0 += H_CHUNK) {
      const h = Math.min(H_CHUNK, height - y0);
      const strip = ctx.getImageData(0, y0, width, h);
      const dataRGBA = strip.data;
      const start = y0 * width;
      for (let row = 0; row < h; row++) {
        const rowStartAlpha = start + row * width;
        const rowStartRGBA = row * width * 4;
        for (let x = 0; x < width; x++) {
          dataRGBA[rowStartRGBA + x * 4 + 3] = alpha[rowStartAlpha + x];
        }
      }
      ctx.putImageData(strip, 0, y0);
    }

    // Use PNG instead of WebP - better alpha channel support in OffscreenCanvas
    const mainBlob = await canvas.convertToBlob({ type: 'image/png', quality: 0.95 });

    // Build preview
    const maxSide = Math.max(width, height);
    const scale = Math.min(1, previewMax / maxSide);
    const pW = Math.max(1, Math.round(width * scale));
    const pH = Math.max(1, Math.round(height * scale));
    const pCanvas = new OffscreenCanvas(pW, pH);
    const pCtx = pCanvas.getContext('2d', { willReadFrequently: true })!;
    pCtx.drawImage(canvas, 0, 0, pW, pH);
    const previewBlob = await pCanvas.convertToBlob({ type: 'image/png', quality: 0.6 as any });

    // Attempt to close bitmap
    try { (bitmap as any)?.close?.(); } catch {}

    const msg: ComposeResultMessage = { type: 'result', mainBlob, previewBlob };
    (self as any).postMessage(msg);
  } catch (err: any) {
    const msg: ComposeResultMessage = { type: 'error', message: err?.message || String(err) };
    (self as any).postMessage(msg);
  }
};


