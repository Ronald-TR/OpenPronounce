/**
 * Browser smoke test: serve the production build, load the page in headless
 * Chromium, wait for the model, run a fixture through the worker and check
 * the decoded phones.
 *
 * Usage: npm run test:browser
 */
import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { build, preview, type PreviewServer } from 'vite';

const ROOT_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const FIXTURES_DIR = path.join(ROOT_DIR, 'tests', 'fixtures');
const SCREENSHOTS_DIR = path.join(ROOT_DIR, 'tests', 'screenshots');

async function main(): Promise<void> {
  console.log('building...');
  await build({ logLevel: 'warn' });

  const server: PreviewServer = await preview({ root: ROOT_DIR, preview: { port: 5199 } });
  const url = server.resolvedUrls?.local?.[0] ?? 'http://localhost:5199/';

  await mkdir(SCREENSHOTS_DIR, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage();

  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  page.on('console', (message) => {
    if (message.type() === 'error') {
      errors.push(`console: ${message.text()}`);
    }
  });

  await page.goto(url);
  const isolated = await page.evaluate('self.crossOriginIsolated');
  console.log('crossOriginIsolated:', isolated);

  // G2P (espeak-ng WASM) renders expected phones right away, no model needed
  await page.waitForFunction(
    () => (document.getElementById('expected-phones')?.textContent ?? '').includes('/'),
    undefined,
    { timeout: 60_000 },
  );
  const g2p = await page.textContent('#expected-phones');
  console.log('g2p phones:', g2p?.trim());

  console.log('loading model (wasm/q8, first run downloads ~318 MB)...');
  await page.waitForFunction(
    () => document.getElementById('status')?.textContent?.includes('model ready'),
    undefined,
    { timeout: 25 * 60_000 },
  );
  const status = await page.textContent('#status');
  console.log('status:', status?.trim());

  await page.setInputFiles('#file-input', path.join(FIXTURES_DIR, 'en-coffee.wav'));
  console.log('fixture submitted, waiting for phones...');
  await page.waitForFunction(
    () => (document.getElementById('phones')?.children.length ?? 0) > 0,
    undefined,
    { timeout: 5 * 60_000 },
  );

  const phones = await page.locator('#phones span').allTextContents();
  const total = await page.textContent('#t-total');
  const inference = await page.textContent('#t-inference');
  const frames = await page.textContent('#t-frames');

  console.log('heard phones:', phones.join(' '));
  console.log(`timings: total ${total} · inference ${inference} · ${frames}`);

  await page.screenshot({ path: path.join(SCREENSHOTS_DIR, 'spike-browser.png'), fullPage: true });
  console.log('screenshot: web/tests/screenshots/spike-browser.png');

  await browser.close();
  await new Promise<void>((resolve) => server.httpServer?.close(() => resolve()));

  const heard = phones.join(' ');
  const ok = heard.startsWith('aɪ w ʊ d l aɪ k') && Boolean(g2p && g2p.includes('/'));
  console.log(ok ? 'SMOKE TEST PASSED' : 'SMOKE TEST FAILED');
  if (errors.length) {
    console.log('console/page errors:', errors.slice(0, 5));
  }
  process.exit(ok ? 0 : 1);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
