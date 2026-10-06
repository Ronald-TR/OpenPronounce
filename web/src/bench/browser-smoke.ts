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
  page.on('pageerror', (error) => {
    errors.push(`pageerror: ${error.message}`);
    console.log('  [pageerror]', error.message);
  });
  page.on('console', (message) => {
    if (message.type() === 'error') {
      errors.push(`console: ${message.text()}`);
      console.log('  [console.error]', message.text().slice(0, 300));
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

  // Full score: piper reference + acoustic encoder + phones + words
  console.log('scoring against piper reference (downloads voice ~63 MB + encoder ~355 MB first run)...');
  // poll status so failures are visible while waiting
  const statusPoller = setInterval(async () => {
    const status = await page.textContent('#status').catch(() => null);
    if (status && status.trim()) {
      console.log('  [status]', status.trim());
    }
  }, 10_000);
  try {
    await page.click('#score-btn');
    await page.waitForFunction(
      () => {
        const value = document.getElementById('score-value')?.textContent ?? '';
        return document.getElementById('score-box')?.classList.contains('hidden') === false && value !== '–';
      },
      undefined,
      { timeout: 15 * 60_000 },
    );
  } finally {
    clearInterval(statusPoller);
  }
  const finalStatus = await page.textContent('#status');
  console.log('final status:', finalStatus?.trim());
  const score = await page.textContent('#score-value');
  const scoreAcoustic = await page.textContent('#score-acoustic');
  const scorePhones = await page.textContent('#score-phones');
  const scoreWords = await page.textContent('#score-words');
  console.log(`SCORE: ${score} · acoustic ${scoreAcoustic} · phones ${scorePhones} · words ${scoreWords}`);

  await page.screenshot({ path: path.join(SCREENSHOTS_DIR, 'spike-browser.png'), fullPage: true });
  console.log('screenshot: web/tests/screenshots/spike-browser.png');

  await browser.close();
  await new Promise<void>((resolve) => server.httpServer?.close(() => resolve()));

  const heard = phones.join(' ');
  const scoreNumber = Number(score);
  const ok = heard.startsWith('aɪ w ʊ d l aɪ k') && scoreNumber > 75;
  console.log(ok ? 'SMOKE TEST PASSED' : `SMOKE TEST FAILED (heard ok: ${heard.startsWith('aɪ w ʊ d l aɪ k')}, score: ${scoreNumber})`);
  if (errors.length) {
    console.log('console/page errors:', errors.slice(0, 5));
  }
  process.exit(ok ? 0 : 1);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
