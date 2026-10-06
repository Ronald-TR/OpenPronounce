/**
 * Faithful port of fastdtw 0.3.4 (the exact version pinned by the Python
 * pipeline), including its tie-breaking so distances AND paths match the
 * server output. Points are n-dimensional vectors; 1-D sequences are passed
 * as [value].
 */
export type Point = number[];

function euclidean(a: Point, b: Point): number {
  let sum = 0;
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

interface BacktrackCell {
  cost: number;
  fromI: number;
  fromJ: number;
}

const INFINITY = Number.POSITIVE_INFINITY;

function dtw(x: Point[], y: Point[], window: Array<[number, number]> | null): { distance: number; path: Array<[number, number]> } {
  const lenX = x.length;
  const lenY = y.length;

  // D maps "i,j" -> [cost, fromI, fromJ] exactly like the Python defaultdict
  // of tuples; min() picks the first argument on ties (up, then left, then diag).
  const d = new Map<string, BacktrackCell>();
  const key = (i: number, j: number) => `${i},${j}`;
  d.set(key(0, 0), { cost: 0, fromI: 0, fromJ: 0 });

  const cells = window ?? allPairs(lenX, lenY);
  for (const [i0, j0] of cells) {
    const i = i0 + 1;
    const j = j0 + 1;
    const dt = euclidean(x[i0], y[j0]);
    const up = d.get(key(i - 1, j));
    const left = d.get(key(i, j - 1));
    const diag = d.get(key(i - 1, j - 1));
    const candidates: Array<[number, number, number]> = [
      [(up?.cost ?? INFINITY) + dt, i - 1, j],
      [(left?.cost ?? INFINITY) + dt, i, j - 1],
      [(diag?.cost ?? INFINITY) + dt, i - 1, j - 1],
    ];
    let best = candidates[0];
    for (const candidate of candidates) {
      if (candidate[0] < best[0]) {
        best = candidate;
      }
    }
    d.set(key(i, j), { cost: best[0], fromI: best[1], fromJ: best[2] });
  }

  const path: Array<[number, number]> = [];
  let i = lenX;
  let j = lenY;
  while (!(i === 0 && j === 0)) {
    path.push([i - 1, j - 1]);
    const cell = d.get(key(i, j));
    if (!cell) {
      // Window did not connect the corners (should not happen with fastdtw's
      // expansion); fall back to a diagonal step so we always terminate.
      i = Math.max(0, i - 1);
      j = Math.max(0, j - 1);
      continue;
    }
    i = cell.fromI;
    j = cell.fromJ;
  }
  path.reverse();
  return { distance: d.get(key(lenX, lenY))?.cost ?? INFINITY, path };
}

function* allPairs(lenX: number, lenY: number): Generator<[number, number]> {
  for (let i = 0; i < lenX; i++) {
    for (let j = 0; j < lenY; j++) {
      yield [i, j];
    }
  }
}

function reduceByHalf(x: Point[]): Point[] {
  const result: Point[] = [];
  const limit = x.length - (x.length % 2);
  for (let i = 0; i < limit; i += 2) {
    const a = x[i];
    const b = x[i + 1];
    const mid = new Array<number>(Math.max(a.length, b.length));
    for (let k = 0; k < mid.length; k++) {
      mid[k] = ((a[k] ?? 0) + (b[k] ?? 0)) / 2;
    }
    result.push(mid);
  }
  return result;
}

function expandWindow(path: Array<[number, number]>, lenX: number, lenY: number, radius: number): Array<[number, number]> {
  const expanded = new Set<string>();
  for (const [i, j] of path) {
    for (let a = i - radius; a <= i + radius; a++) {
      for (let b = j - radius; b <= j + radius; b++) {
        expanded.add(`${a},${b}`);
      }
    }
  }

  const grown = new Set<string>();
  for (const entry of expanded) {
    const [i, j] = entry.split(',').map(Number);
    grown.add(`${i * 2},${j * 2}`);
    grown.add(`${i * 2},${j * 2 + 1}`);
    grown.add(`${i * 2 + 1},${j * 2}`);
    grown.add(`${i * 2 + 1},${j * 2 + 1}`);
  }

  const window: Array<[number, number]> = [];
  let startJ: number | null = 0;
  for (let i = 0; i < lenX; i++) {
    let newStartJ: number | null = null;
    if (startJ !== null) {
      for (let j: number = startJ; j < lenY; j++) {
        if (grown.has(`${i},${j}`)) {
          window.push([i, j]);
          if (newStartJ === null) {
            newStartJ = j;
          }
        } else if (newStartJ !== null) {
          break;
        }
      }
    }
    startJ = newStartJ;
  }
  return window;
}

export function fastdtw(x: Point[], y: Point[], radius = 1): { distance: number; path: Array<[number, number]> } {
  const minTimeSize = radius + 2;
  if (x.length < minTimeSize || y.length < minTimeSize) {
    return dtw(x, y, null);
  }
  const xShrunk = reduceByHalf(x);
  const yShrunk = reduceByHalf(y);
  const shrunk = fastdtw(xShrunk, yShrunk, radius);
  const window = expandWindow(shrunk.path, x.length, y.length, radius);
  return dtw(x, y, window);
}
