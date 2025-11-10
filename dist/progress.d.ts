export type ProgressPhase = 'idle' | 'downloading' | 'building' | 'ready' | 'error';
export type ProgressState = {
    phase: ProgressPhase;
    progress: number;
    errorMsg?: string;
    sessionId: number;
};
type ProgressListener = (state: ProgressState) => void;
declare class OnnxProgress {
    private state;
    private listeners;
    subscribe(listener: ProgressListener): () => void;
    private emit;
    beginNewSession(): number;
    setNetworkProgress(percent: number, sessionId: number): void;
    setBuilding(sessionId: number): void;
    setReady(sessionId: number): void;
    setError(sessionId: number, errorMsg?: string): void;
    get progressPercent(): number;
    get phase(): ProgressPhase;
}
export declare const onnxProgress: OnnxProgress;
export {};
