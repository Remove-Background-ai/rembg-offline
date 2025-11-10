export type ComposeMessage = {
    type: 'compose';
    width: number;
    height: number;
    alphaBuffer: ArrayBuffer;
    alphaByteOffset: number;
    alphaLength: number;
    bitmap: ImageBitmap;
    previewMax: number;
};
export type ComposeResultMessage = {
    type: 'result';
    mainBlob: Blob;
    previewBlob: Blob;
} | {
    type: 'error';
    message: string;
};
