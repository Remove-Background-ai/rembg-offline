export type ProgressPhase = 'idle' | 'downloading' | 'building' | 'ready' | 'error';

export type ProgressState = {
  phase: ProgressPhase;
  progress: number; // 0..100 (downloading/building) – ready sets to 100
  errorMsg?: string;
  sessionId: number;
};

type ProgressListener = (state: ProgressState) => void;

class OnnxProgress {
  private state: ProgressState = { phase: 'idle', progress: 0, sessionId: 0 };
  private listeners: Set<ProgressListener> = new Set();

  subscribe(listener: ProgressListener): () => void {
    this.listeners.add(listener);
    // send current immediately
    try { listener(this.state); } catch {}
    return () => { this.listeners.delete(listener); };
  }

  private emit() {
    for (const l of this.listeners) {
      try { l(this.state); } catch {}
    }
  }

  beginNewSession(): number {
    this.state = { phase: 'idle', progress: 0, sessionId: (this.state.sessionId + 1) | 0 };
    this.emit();
    return this.state.sessionId;
  }

  setNetworkProgress(percent: number, sessionId: number) {
    if (sessionId !== this.state.sessionId) return;
    const p = Math.max(0, Math.min(100, Math.floor(percent)));
    this.state = { ...this.state, phase: 'downloading', progress: p };
    this.emit();
  }

  setBuilding(sessionId: number) {
    if (sessionId !== this.state.sessionId) return;
    // Transition to building, keep progress near end but below 100 until ready
    const p = Math.max(this.state.progress, 99);
    this.state = { ...this.state, phase: 'building', progress: p };
    this.emit();
  }

  setReady(sessionId: number) {
    if (sessionId !== this.state.sessionId) return;
    this.state = { ...this.state, phase: 'ready', progress: 100 };
    this.emit();
  }

  setError(sessionId: number, errorMsg?: string) {
    if (sessionId !== this.state.sessionId) return;
    this.state = { ...this.state, phase: 'error', errorMsg, progress: 0 };
    this.emit();
  }

  get progressPercent() {
    return this.state.progress;
  }

  get phase() {
    return this.state.phase;
  }
}

export const onnxProgress = new OnnxProgress();


