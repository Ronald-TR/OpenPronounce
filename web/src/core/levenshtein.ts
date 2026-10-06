/**
 * Levenshtein distance and opcodes for token sequences, matching the semantics
 * of the Python `Levenshtein` package used by openpronounce (rapidfuzz-based).
 */

export type OpcodeTag = 'equal' | 'replace' | 'delete' | 'insert';

export interface Opcode {
  tag: OpcodeTag;
  i1: number;
  i2: number;
  j1: number;
  j2: number;
}

/** Edit distance between two token sequences. */
export function levenshteinDistance<T>(a: ArrayLike<T>, b: ArrayLike<T>): number {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;

  let previous = new Uint32Array(n + 1);
  let current = new Uint32Array(n + 1);
  for (let j = 0; j <= n; j++) {
    previous[j] = j;
  }
  for (let i = 1; i <= m; i++) {
    current[0] = i;
    for (let j = 1; j <= n; j++) {
      const substitution = previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1);
      const deletion = previous[j] + 1;
      const insertion = current[j - 1] + 1;
      current[j] = Math.min(substitution, deletion, insertion);
    }
    const swap = previous;
    previous = current;
    current = swap;
  }
  return previous[n];
}

/**
 * Opcodes describing how to turn `a` into `b`, matching Levenshtein.opcodes
 * from the Python package (rapidfuzz). rapidfuzz recovers the script with a
 * forward walk over the suffix-distance matrix; among tied optimal steps it
 * prefers insert, then replace, then delete (validated against fixtures and
 * hand-built discriminating cases).
 */
export function levenshteinOpcodes<T>(a: ArrayLike<T>, b: ArrayLike<T>): Opcode[] {
  const m = a.length;
  const n = b.length;

  // suffix[i][j] = distance between a[i:] and b[j:]
  const suffix: Uint32Array[] = [];
  for (let i = 0; i <= m; i++) {
    suffix.push(new Uint32Array(n + 1));
  }
  for (let i = 0; i <= m; i++) suffix[i][n] = m - i;
  for (let j = 0; j <= n; j++) suffix[m][j] = n - j;
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      const substitution = suffix[i + 1][j + 1] + (a[i] === b[j] ? 0 : 1);
      const deletion = suffix[i + 1][j] + 1;
      const insertion = suffix[i][j + 1] + 1;
      suffix[i][j] = Math.min(substitution, deletion, insertion);
    }
  }

  const ops: Opcode[] = [];
  let i = 0;
  let j = 0;
  while (i < m || j < n) {
    if (i < m && j < n && a[i] === b[j] && suffix[i][j] === suffix[i + 1][j + 1]) {
      ops.push({ tag: 'equal', i1: i, i2: i + 1, j1: j, j2: j + 1 });
      i += 1;
      j += 1;
      continue;
    }
    const insertion = j < n && suffix[i][j] === suffix[i][j + 1] + 1;
    const replacement = i < m && j < n && suffix[i][j] === suffix[i + 1][j + 1] + 1;
    const deletion = i < m && suffix[i][j] === suffix[i + 1][j] + 1;
    if (insertion) {
      ops.push({ tag: 'insert', i1: i, i2: i, j1: j, j2: j + 1 });
      j += 1;
    } else if (replacement) {
      ops.push({ tag: 'replace', i1: i, i2: i + 1, j1: j, j2: j + 1 });
      i += 1;
      j += 1;
    } else if (deletion) {
      ops.push({ tag: 'delete', i1: i, i2: i + 1, j1: j, j2: j });
      i += 1;
    } else {
      throw new Error('levenshtein backtrace lost the optimal path');
    }
  }

  // Merge consecutive operations of the same tag into ranges.
  const merged: Opcode[] = [];
  for (const op of ops) {
    const last = merged[merged.length - 1];
    if (last && last.tag === op.tag && last.i2 === op.i1 && last.j2 === op.j1) {
      last.i2 = op.i2;
      last.j2 = op.j2;
    } else {
      merged.push({ ...op });
    }
  }
  return merged;
}
