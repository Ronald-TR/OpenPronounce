/**
 * Browser smoke test: serve the production build, load the app in headless
 * Chromium, wait for the local models, analyze a fixture and check the score.
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
  const browser = await chromium.launch({ args: ['--disable-features=ServiceWorker'] });
  const page = await browser.newPage();

  const errors: string[] = [];
  page.on('pageerror', (error) => {
    errors.push(`pageerror: ${error.message}`);
    console.log('  [pageerror]', error.message);
  });
  page.on('console', (message) => {
    const text = message.text();
    if (message.type() === 'error') {
      errors.push(`console: ${text}`);
      console.log('  [console.error]', text.slice(0, 300));
    } else if (text.includes('[score inputs]')) {
      console.log('  [page]', text);
    }
  });

  await page.goto(url);
  const isolated = await page.evaluate('self.crossOriginIsolated');
  console.log('crossOriginIsolated:', isolated);

  console.log('waiting for local models (~830 MB on a fresh profile)...');
  await page.waitForFunction(
    () => document.getElementById('record-hint')?.textContent === 'Tap to record, then say the sentence',
    undefined,
    { timeout: 25 * 60_000 },
  );

  await page.fill('#expected-text', 'I would like a cup of coffee, please.');
  await page.setInputFiles('#file-input', path.join(FIXTURES_DIR, 'en-coffee.wav'));
  console.log('fixture submitted, clicking Analyze...');
  await page.click('#analyze-btn');

  // poll status so failures are visible while waiting
  const statusPoller = setInterval(async () => {
    const status = await page.textContent('#loading-step').catch(() => null);
    if (status && status.trim()) {
      console.log('  [status]', status.trim());
    }
  }, 15_000);
  try {
    await page.waitForFunction(
      () => document.getElementById('result')?.classList.contains('hidden') === false,
      undefined,
      { timeout: 15 * 60_000 },
    );
  } finally {
    clearInterval(statusPoller);
  }

  // wait for the score count-up animation (900 ms) to finish before reading
  await page.waitForFunction(
    () => {
      const el = document.getElementById('score-value');
      if (!el) return false;
      const current = el.textContent ?? '';
      return el.dataset.final === '1' && current.length > 0;
    },
    undefined,
    { timeout: 60_000 },
  );
  const score = await page.textContent('#score-value');
  const verdict = await page.textContent('#verdict');
  const heard = await page.textContent('#heard-phones');
  const transcript = await page.textContent('#transcript');
  const breakdown = await page.textContent('#breakdown');
  const phoneSpans = await page.locator('#heard-phones span').allTextContents();
  const chipCount = await page.locator('#word-chips button').count();

  console.log(`SCORE: ${score} · ${verdict}`);
  console.log('heard phones (spans):', JSON.stringify(phoneSpans.slice(0, 8)), `(${phoneSpans.length} spans)`);
  console.log(`chips: ${chipCount}`);
  console.log('transcript:', transcript?.trim());
  console.log('breakdown:', breakdown?.trim());

  await page.screenshot({ path: path.join(SCREENSHOTS_DIR, 'app-browser.png'), fullPage: true });
  console.log('screenshot: web/tests/screenshots/app-browser.png');

  await browser.close();
  await new Promise<void>((resolve) => server.httpServer?.close(() => resolve()));

  const heardText = (heard ?? '').replace(/\s+/g, '');
  const scoreNumber = Number(score);
  const ok = heardText.startsWith('aɪwʊdlaɪk') && phoneSpans.length >= 20 && scoreNumber > 75;
  console.log(ok ? 'SMOKE TEST PASSED' : `SMOKE TEST FAILED (score: ${scoreNumber}, spans: ${phoneSpans.length})`);
  if (errors.length) {
    console.log('errors seen:', errors.slice(0, 5));
  }
  process.exit(ok ? 0 : 1);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
