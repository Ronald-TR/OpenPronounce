/**
 * WebGPU capability diagnosis. WebGPU is mandatory: every model runs on the
 * GPU, so the microphone is only unlocked when an adapter is available.
 */
export interface WebGpuSupport {
  supported: boolean;
  /** Human-readable explanation, e.g. for the settings bar. */
  detail: string;
}

interface MinimalAdapter {
  info?: unknown;
  features?: Set<string>;
}

interface NavigatorWithGpu {
  gpu?: {
    requestAdapter(options?: { powerPreference?: string }): Promise<MinimalAdapter | null>;
  };
}

export async function webgpuSupport(): Promise<WebGpuSupport> {
  const nav = navigator as Navigator & NavigatorWithGpu;
  if (!nav.gpu) {
    return {
      supported: false,
      detail: 'navigator.gpu is unavailable. Use Chrome/Edge 113+, or enable hardware acceleration in the browser settings.',
    };
  }
  try {
    const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      return {
        supported: false,
        detail: 'No WebGPU adapter found. Check that hardware acceleration is enabled.',
      };
    }
    const info = adapter.info as { vendor?: string; architecture?: string } | undefined;
    const bits = [info?.vendor, info?.architecture].filter(Boolean).join(' / ');
    return {
      supported: true,
      detail: bits ? `adapter: ${bits}` : 'WebGPU adapter available',
    };
  } catch (error) {
    return {
      supported: false,
      detail: `WebGPU probe failed: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}
