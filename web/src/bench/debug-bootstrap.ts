/** Manual local-mode bootstrap diagnostic. */
import { chromium } from 'playwright';
import { build, preview, type PreviewServer } from 'vite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

async function main(): Promise<void> {
  await build({ logLevel: 'warn' });
  const server: PreviewServer = await preview({ root: ROOT_DIR, preview: { port: 5199 } });
  const url = server.resolvedUrls?.local?.[0] ?? 'http://localhost:5199/';

  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on('console', (m) => {
    if (['error', 'warning'].includes(m.type())) console.log('  [console]', m.type(), m.text().slice(0, 200));
  });
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message.slice(0, 200)));
  page.on('requestfailed', (req) => {
    console.log('  [reqfail]', req.url().slice(0, 110), '·', req.failure()?.errorText);
  });

  await page.goto(url);
  for (const wait of [5_000, 20_000, 60_000]) {
    await page.waitForTimeout(wait === 5_000 ? 5_000 : wait === 20_000 ? 15_000 : 40_000);
    const state = await page.evaluate(() => ({
      isolated: self.crossOriginIsolated,
      url: location.href,
      hint: document.getElementById('record-hint')?.textContent ?? null,
      loadingStep: document.getElementById('loading-step')?.textContent ?? null,
      error: document.getElementById('error-message')?.textContent ?? null,
    })).catch((e) => ({ evalError: String(e).slice(0, 120) }));
    console.log(`t+${wait / 1000}s:`, JSON.stringify(state));
  }

  await browser.close();
  await new Promise<void>((resolve) => server.httpServer?.close(() => resolve()));
}

void main();
