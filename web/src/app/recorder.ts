/**
 * Microphone recorder built on MediaRecorder — TypeScript port of the app's
 * static/audio.js, including microphone selection and the auto-stop toggle.
 *
 * Emits DOM events on `document`:
 *   record:start          recording began
 *   record:stop           recording ended (user, timeout or silence)
 *   record:silence        recording ended because of silence
 *   record:ready          detail: { blob }  the webm blob is available
 *   record:error          detail: { reason } one of "insecure", "denied", "notfound", "unknown"
 */
export class AudioRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private audioChunks: Blob[] = [];
  private silenceTimer: ReturnType<typeof setTimeout> | null = null;
  private maxTimer: ReturnType<typeof setTimeout> | null = null;
  private silenceThreshold: number;
  private silenceDuration: number;
  private maxDuration: number;
  private deviceId: string | null;
  private silenceDetection: boolean;
  private started = false;

  constructor({ silenceThreshold = 50, silenceDuration = 2000, maxDuration = 60000, deviceId = null, silenceDetection = true } = {}) {
    this.silenceThreshold = silenceThreshold;
    this.silenceDuration = silenceDuration;
    this.maxDuration = maxDuration;
    this.deviceId = deviceId;
    this.silenceDetection = silenceDetection;
  }

  static isSupported(): boolean {
    return Boolean(navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === "function" && window.MediaRecorder);
  }

  setDeviceId(deviceId: string | null): void {
    this.deviceId = deviceId || null;
  }

  setSilenceDetection(enabled: boolean): void {
    this.silenceDetection = Boolean(enabled);
  }

  audioConstraints(): MediaStreamConstraints {
    return { audio: this.deviceId ? { deviceId: { exact: this.deviceId } } : true };
  }

  emitRecordError(err: unknown): void {
    const name = err instanceof Error ? err.name : undefined;
    const reason = (name === 'NotAllowedError' || name === 'SecurityError') ? 'denied'
      : (name === 'NotFoundError' || name === 'OverconstrainedError') ? 'notfound'
        : 'unknown';
    this.emit('record:error', { reason });
  }

  async start(): Promise<void> {
    if (this.started) {
      this.stop();
      return;
    }

    if (!AudioRecorder.isSupported()) {
      this.emit('record:error', { reason: window.isSecureContext ? 'unknown' : 'insecure' });
      return;
    }

    try {
      // Ask for the microphone only now, never on page load
      this.stream = await navigator.mediaDevices.getUserMedia(this.audioConstraints());
    } catch (err) {
      if (err instanceof Error && err.name === 'OverconstrainedError' && this.deviceId) {
        // The chosen device is gone; fall back to the system default
        this.deviceId = null;
        try {
          this.stream = await navigator.mediaDevices.getUserMedia(this.audioConstraints());
        } catch (retryErr) {
          this.emitRecordError(retryErr);
          return;
        }
      } else {
        this.emitRecordError(err);
        return;
      }
    }

    this.audioContext = new AudioContext();
    const source = this.audioContext.createMediaStreamSource(this.stream);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 256;
    source.connect(this.analyser);

    this.audioChunks = [];
    this.mediaRecorder = new MediaRecorder(this.stream);
    this.mediaRecorder.ondataavailable = (event) => this.audioChunks.push(event.data);
    this.mediaRecorder.onstop = () => {
      const blob = new Blob(this.audioChunks, { type: 'audio/webm' });
      this.emit('record:ready', { blob });
    };

    // Only announce the recording once the recorder is really capturing: the first
    // 100-300 ms after start() are lost otherwise, and users start talking on the click.
    this.mediaRecorder.onstart = () => {
      if (this.silenceDetection) {
        requestAnimationFrame(() => this.checkSilence());
        this.maxTimer = setTimeout(() => this.stop(), this.maxDuration);
      }
      this.emit('record:start');
    };

    this.started = true;
    this.mediaRecorder.start();
  }

  stop(): void {
    if (this.maxTimer) {
      clearTimeout(this.maxTimer);
      this.maxTimer = null;
    }
    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }

    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.stop();
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      void this.audioContext.close();
    }

    if (!this.started) {
      return;
    }
    this.started = false;
    this.emit('record:stop');
  }

  isRecording(): boolean {
    return this.started;
  }

  private stopDueToSilence(): void {
    this.stop();
    this.emit('record:silence');
  }

  private checkSilence(): void {
    if (!this.started || !this.analyser) {
      return;
    }
    const data = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(data);
    let sum = 0;
    for (const value of data) sum += value;
    const average = sum / data.length;

    if (average < this.silenceThreshold) {
      if (!this.silenceTimer) {
        this.silenceTimer = setTimeout(() => this.stopDueToSilence(), this.silenceDuration);
      }
    } else if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }

    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      requestAnimationFrame(() => this.checkSilence());
    }
  }

  private emit(name: string, detail?: unknown): void {
    document.dispatchEvent(new CustomEvent(name, { detail }));
  }
}
