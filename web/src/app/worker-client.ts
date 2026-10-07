/**
 * Promise-based RPC over the inference worker. Events (progress, ready...)
 * go to the `onEvent` callback; request/response pairs are matched by id.
 */

export interface WorkerProgress {
  status: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}

type Pending = { resolve: (value: unknown) => void; reject: (error: Error) => void };

export class WorkerClient {
  private worker: Worker;
  private pending = new Map<number, Pending>();
  private nextId = 1;
  private onEvent: (message: Record<string, unknown>) => void;

  constructor(worker: Worker, onEvent: (message: Record<string, unknown>) => void) {
    this.worker = worker;
    this.onEvent = onEvent;
    worker.onmessage = (event: MessageEvent) => {
      const data = event.data as Record<string, unknown> & { requestId?: number };
      if (typeof data.requestId === 'number' && this.pending.has(data.requestId)) {
        const pending = this.pending.get(data.requestId)!;
        this.pending.delete(data.requestId);
        if (data.type === 'error') {
          pending.reject(new Error(String(data.message)));
        } else {
          pending.resolve(data);
        }
        return;
      }
      this.onEvent(data);
    };
    worker.onerror = (event) => {
      this.onEvent({ type: 'error', message: event.message || 'worker crashed' });
    };
  }

  post(message: Record<string, unknown>, transfer?: Transferable[]): void {
    this.worker.postMessage(message, transfer ?? []);
  }

  request<T>(message: Record<string, unknown>, transfer?: Transferable[]): Promise<T> {
    const requestId = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(requestId, { resolve: resolve as (value: unknown) => void, reject });
      this.post({ ...message, requestId }, transfer);
    });
  }
}
