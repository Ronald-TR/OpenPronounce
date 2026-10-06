/// <reference lib="webworker" />
/**
 * Inference worker: keeps transformers.js + ONNX Runtime off the UI thread.
 *
 * Messages in:  { type: 'load', device, dtype }   { type: 'infer', audio: Float32Array }
 * Messages out: { type: 'progress', payload }     { type: 'ready', info }
 *               { type: 'result', payload }       { type: 'error', message }
 */
import { createSession, type Device, type Dtype, type ModelProgress, type Session, type SessionInfo } from './core/session';

let session: Session | null = null;

type LoadMessage = { type: 'load'; device: Device; dtype: Dtype };
type InferMessage = { type: 'infer'; audio: Float32Array; requestId: number };
type WorkerMessage = LoadMessage | InferMessage;

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  try {
    if (message.type === 'load') {
      session = await createSession({
        device: message.device,
        dtype: message.dtype,
        onProgress: (progress: ModelProgress) => {
          (self as unknown as Worker).postMessage({ type: 'progress', payload: progress });
        },
      });
      const info: SessionInfo = session.info;
      (self as unknown as Worker).postMessage({ type: 'ready', payload: info });
      return;
    }

    if (message.type === 'infer') {
      if (!session) {
        throw new Error('model not loaded yet');
      }
      const result = await session.run(message.audio);
      (self as unknown as Worker).postMessage({
        type: 'result',
        payload: { requestId: message.requestId, ...result },
      });
      return;
    }
  } catch (error) {
    (self as unknown as Worker).postMessage({
      type: 'error',
      message: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    });
  }
};
