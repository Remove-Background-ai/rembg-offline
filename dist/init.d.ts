declare global {
    interface Window {
        __rembg_offline_fetch_patched__?: boolean;
    }
}
export declare function init(setModelLoaded?: (b: boolean) => void): Promise<{
    model: any;
    processor: any;
}>;
