/** Probe: does headless Chromium expose WebGPU (for the phase-4 measurement)? */
import { chromium } from 'playwright';

async function main(): Promise<void> {
  for (const args of [
    ['--enable-unsafe-webgpu'],
    ['--enable-unsafe-webgpu', '--use-angle=metal'],
    ['--enable-unsafe-webgpu', '--use-angle=swiftshader', '--enable-features=Vulkan'],
  ]) {
    const browser = await chromium.launch({ args });
    const page = await browser.newPage();
    await page.setContent('<title>probe</title>');
    const result = await page.evaluate(async () => {
      const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
      if (!gpu) {
        return 'no navigator.gpu';
      }
      try {
        const adapter = await gpu.requestAdapter();
        return adapter ? 'adapter available' : 'requestAdapter returned null';
      } catch (error) {
        return `error: ${String(error)}`;
      }
    });
    console.log(args.join(' '), '->', result);
    await browser.close();
  }
}

void main();
