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
  page.on('pageerror', () => { });

  await page.goto(url);
  const isolated = await page.evaluate('self.crossOriginIsolated');
  console.log('crossOriginIsolated:', isolated);

  console.log('waiting for local models...');
  // when models come over the network the download popup shows progress;
  // in the local mode they load from the preview server almost instantly
  const popupEarly = await page.evaluate(() => {
    const popup = document.getElementById('download-popup');
    return popup !== null && !popup.classList.contains('hidden');
  });
  console.log('download popup visible during load:', popupEarly);

  try {
    await page.waitForFunction(
      () => {
        const hint = document.getElementById('record-hint')?.textContent ?? '';
        return hint === 'Tap to record, then say the sentence' || hint === 'Could not load the local models.';
      },
      undefined,
      { timeout: 25 * 60_000 },
    );
  } catch (error) {
    const status = await page.textContent('#status').catch(() => null);
    const hint = await page.textContent('#record-hint').catch(() => null);
    const errorMessage = await page.textContent('#error-message').catch(() => null);
    console.log('TIMEOUT state — status:', status?.trim(), '· hint:', hint?.trim(), '· error:', errorMessage?.trim());
    throw error;
  }
  // once everything is loaded, the popup is gone
  const popupHiddenAfterLoad = await page.evaluate(
    () => document.getElementById('download-popup')?.classList.contains('hidden') ?? false,
  );
  console.log('popup hidden after load:', popupHiddenAfterLoad);
  const hint = await page.textContent('#record-hint');
  if (hint !== 'Tap to record, then say the sentence') {
    const errorMessage = await page.textContent('#error-message');
    const status = await page.textContent('#status');
    console.log('BOOTSTRAP FAILED — hint:', hint?.trim(), '· error:', errorMessage?.trim(), '· status:', status?.trim());
    await browser.close();
    process.exit(1);
  }

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

  // Settings bar: pick a speak speed, then "Hear the reference"
  const settingsVisible = await page.evaluate(
    () => !document.getElementById('settings-body')?.classList.contains('hidden'),
  );
  await page.selectOption('#speak-speed-setting', '0.5');
  const speedValue = await page.evaluate(() => (document.getElementById('speak-speed-setting') as HTMLSelectElement).value);
  console.log('settings bar visible:', settingsVisible, '· speak speed set to', speedValue);

  // "Hear the reference": piper synth + playback (voice already cached by the score step)
  await page.click('#listen-btn');
  await page.waitForFunction(
    () => {
      const label = document.querySelector('[data-role="listen.label"]')?.textContent ?? '';
      return label === 'Playing…' || label === 'Reference unavailable right now';
    },
    undefined,
    { timeout: 3 * 60_000 },
  );
  const listenLabel = await page.textContent('[data-role="listen.label"]');
  console.log('listen button:', listenLabel?.trim());

  // Details section: charts must render when opened
  await page.click('#details summary');
  await page.waitForFunction(
    () => {
      const canvas = document.getElementById('phoneme-chart') as HTMLCanvasElement | null;
      return canvas !== null && canvas.width > 0;
    },
    undefined,
    { timeout: 30_000 },
  );
  const canvasWidths = await page.evaluate(() => ({
    phoneme: (document.getElementById('phoneme-chart') as HTMLCanvasElement | null)?.width ?? 0,
    f0: (document.getElementById('f0-chart') as HTMLCanvasElement | null)?.width ?? 0,
  }));
  console.log(`details charts rendered: phoneme ${canvasWidths.phoneme}px · f0 ${canvasWidths.f0}px`);

  // Word detail + viseme animation: click a chip, then "Hear it and watch the mouth"
  await page.click('#word-chips button:nth-child(2)');
  await page.waitForFunction(
    () => document.getElementById('word-detail')?.classList.contains('hidden') === false,
    undefined,
    { timeout: 10_000 },
  );
  const detailWord = await page.textContent('#detail-word');
  await page.click('#detail-listen');
  await page.waitForFunction(
    () => {
      const src = (document.getElementById('viseme-image') as HTMLImageElement | null)?.src ?? '';
      return src.includes('/assets/mouths/') && !src.endsWith('rest.png');
    },
    undefined,
    { timeout: 10_000 },
  );
  const visemeSrc = await page.evaluate(
    () => (document.getElementById('viseme-image') as HTMLImageElement).src.split('/').pop(),
  );
  console.log(`word detail: "${detailWord}" · viseme animating: ${visemeSrc}`);
  await page.waitForTimeout(3000); // let the animation finish

  await page.screenshot({ path: path.join(SCREENSHOTS_DIR, 'app-browser.png'), fullPage: true });
  console.log('screenshot: web/tests/screenshots/app-browser.png');

  await browser.close();
  await new Promise<void>((resolve) => server.httpServer?.close(() => resolve()));

  const heardText = (heard ?? '').replace(/\s+/g, '');
  const scoreNumber = Number(score);
  const ok = heardText.startsWith('aɪwʊdlaɪk') && phoneSpans.length >= 20 && scoreNumber > 75
    && listenLabel === 'Playing…' && canvasWidths.phoneme > 0 && Boolean(visemeSrc) && visemeSrc !== 'rest.png'
    && speedValue === '0.5' && popupHiddenAfterLoad;
  console.log(ok ? 'SMOKE TEST PASSED' : `SMOKE TEST FAILED (score: ${scoreNumber}, spans: ${phoneSpans.length}, listen: ${listenLabel}, chart: ${canvasWidths.phoneme}px, viseme: ${visemeSrc}, speed: ${speedValue})`);
  if (errors.length) {
    console.log('errors seen:', errors.slice(0, 5));
  }
  process.exit(ok ? 0 : 1);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
