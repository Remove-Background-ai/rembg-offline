class OnnxProgress {
    constructor() {
        this.state = { phase: 'idle', progress: 0, sessionId: 0 };
        this.listeners = new Set();
    }
    subscribe(listener) {
        this.listeners.add(listener);
        // send current immediately
        try {
            listener(this.state);
        }
        catch { }
        return () => { this.listeners.delete(listener); };
    }
    emit() {
        for (const l of this.listeners) {
            try {
                l(this.state);
            }
            catch { }
        }
    }
    beginNewSession() {
        this.state = { phase: 'idle', progress: 0, sessionId: (this.state.sessionId + 1) | 0 };
        this.emit();
        return this.state.sessionId;
    }
    setNetworkProgress(percent, sessionId) {
        if (sessionId !== this.state.sessionId)
            return;
        const p = Math.max(0, Math.min(100, Math.floor(percent)));
        this.state = { ...this.state, phase: 'downloading', progress: p };
        this.emit();
    }
    setBuilding(sessionId) {
        if (sessionId !== this.state.sessionId)
            return;
        // Transition to building, keep progress near end but below 100 until ready
        const p = Math.max(this.state.progress, 99);
        this.state = { ...this.state, phase: 'building', progress: p };
        this.emit();
    }
    setReady(sessionId) {
        if (sessionId !== this.state.sessionId)
            return;
        this.state = { ...this.state, phase: 'ready', progress: 100 };
        this.emit();
    }
    setError(sessionId, errorMsg) {
        if (sessionId !== this.state.sessionId)
            return;
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
