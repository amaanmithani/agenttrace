// The agenttrace/1 run format and the same alignment the Python diff uses.

export type Kind = 'llm' | 'tool' | 'node' | 'mcp' | 'custom';

export interface Step {
  id: number;
  kind: Kind;
  name: string;
  input?: unknown;
  output?: unknown;
  error?: string;
  parent?: number | null;
  start_ns?: number;
  end_ns?: number;
  attrs?: Record<string, unknown>;
}

export interface Run {
  meta: Record<string, unknown>;
  steps: Step[];
}

/** Stable JSON with sorted keys (matches Python's canonical form for equality). */
export function canonical(v: unknown): string {
  if (v === undefined) return 'null';
  if (v === null || typeof v !== 'object') return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(canonical).join(',')}]`;
  const o = v as Record<string, unknown>;
  return `{${Object.keys(o)
    .sort()
    .map((k) => `${JSON.stringify(k)}:${canonical(o[k])}`)
    .join(',')}}`;
}

export function signature(s: Step): string {
  return `${s.kind}:${s.name}:${canonical(s.input ?? null)}`;
}

export function parseRun(text: string): Run {
  const run: Run = { meta: {}, steps: [] };
  text.split('\n').forEach((line, i) => {
    if (!line.trim()) return;
    let obj: unknown;
    try {
      obj = JSON.parse(line);
    } catch {
      throw new Error(`line ${i + 1} is not JSON`);
    }
    if (typeof obj !== 'object' || obj === null || Array.isArray(obj)) throw new Error(`line ${i + 1} is not an object`);
    const o = obj as Record<string, unknown>;
    if ('format' in o) {
      if (o.format !== 'agenttrace/1') throw new Error(`unsupported format ${String(o.format)}`);
      const { format: _f, ...meta } = o;
      run.meta = meta;
    } else {
      if (typeof o.id !== 'number' || typeof o.kind !== 'string' || typeof o.name !== 'string')
        throw new Error(`line ${i + 1} is not a step`);
      run.steps.push(o as unknown as Step);
    }
  });
  run.steps.sort((a, b) => a.id - b.id);
  return run;
}

export const durationMs = (s: Step) => ((s.end_ns ?? 0) - (s.start_ns ?? 0)) / 1e6;

export function depthOf(run: Run): Map<number, number> {
  const byId = new Map(run.steps.map((s) => [s.id, s]));
  const depth = new Map<number, number>();
  const get = (s: Step, guard = 0): number => {
    const known = depth.get(s.id);
    if (known !== undefined) return known;
    const p = s.parent != null ? byId.get(s.parent) : undefined;
    const d = p && guard < 100 ? get(p, guard + 1) + 1 : 0;
    depth.set(s.id, d);
    return d;
  };
  run.steps.forEach((s) => get(s));
  return depth;
}

type RawOp = ['equal' | 'removed' | 'added', number, number];

/** Myers shortest edit script, O((n+m)·d). */
export function myers(a: string[], b: string[]): RawOp[] {
  const n = a.length;
  const m = b.length;
  const trace: Map<number, number>[] = [];
  let v = new Map<number, number>([[1, 0]]);
  for (let d = 0; d <= n + m; d++) {
    trace.push(new Map(v));
    for (let k = -d; k <= d; k += 2) {
      let x = k === -d || (k !== d && (v.get(k - 1) ?? -1) < (v.get(k + 1) ?? -1)) ? (v.get(k + 1) ?? 0) : (v.get(k - 1) ?? 0) + 1;
      let y = x - k;
      while (x < n && y < m && a[x] === b[y]) {
        x++;
        y++;
      }
      v.set(k, x);
      if (x >= n && y >= m) return backtrack(trace, n, m);
    }
    v = new Map(v);
  }
  throw new Error('unreachable');
}

function backtrack(trace: Map<number, number>[], n: number, m: number): RawOp[] {
  const out: RawOp[] = [];
  let x = n;
  let y = m;
  for (let d = trace.length - 1; d >= 0; d--) {
    const v = trace[d]!;
    const k = x - y;
    const pk = k === -d || (k !== d && (v.get(k - 1) ?? -1) < (v.get(k + 1) ?? -1)) ? k + 1 : k - 1;
    const px = v.get(pk) ?? 0;
    const py = px - pk;
    while (x > px && y > py) {
      x--;
      y--;
      out.push(['equal', x, y]);
    }
    if (d > 0) {
      if (x === px) out.push(['added', -1, --y]);
      else out.push(['removed', --x, -1]);
    }
  }
  return out.reverse();
}

export type Op = 'equal' | 'changed' | 'removed' | 'added';
export interface Entry {
  op: Op;
  a?: Step;
  b?: Step;
  outputDiffers: boolean;
}

export function diffRuns(A: Run, B: Run): { entries: Entry[]; first: number } {
  const raw = myers(A.steps.map(signature), B.steps.map(signature));
  const entries: Entry[] = [];
  let i = 0;
  while (i < raw.length) {
    const [op, x, y] = raw[i]!;
    if (op === 'equal') {
      const a = A.steps[x]!;
      const b = B.steps[y]!;
      entries.push({ op, a, b, outputDiffers: canonical(a.output ?? null) !== canonical(b.output ?? null) || a.error !== b.error });
      i++;
      continue;
    }
    const rem: Step[] = [];
    const add: Step[] = [];
    while (i < raw.length && raw[i]![0] !== 'equal') {
      const [o, xx, yy] = raw[i]!;
      if (o === 'removed') rem.push(A.steps[xx]!);
      else add.push(B.steps[yy]!);
      i++;
    }
    // Pair same-kind, same-name steps in order (monotone in both runs); unpaired
    // steps keep their place: A's, then B's, before the next pair.
    const pairs: [number, number][] = [];
    let j0 = 0;
    rem.forEach((r, ri) => {
      for (let j = j0; j < add.length; j++)
        if (add[j]!.kind === r.kind && add[j]!.name === r.name) {
          pairs.push([ri, j]);
          j0 = j + 1;
          return;
        }
    });
    let ri = 0;
    let aj = 0;
    for (const [pr, pa] of [...pairs, [rem.length, add.length] as [number, number]]) {
      rem.slice(ri, pr).forEach((r) => entries.push({ op: 'removed', a: r, outputDiffers: false }));
      add.slice(aj, pa).forEach((x) => entries.push({ op: 'added', b: x, outputDiffers: false }));
      if (pr < rem.length && pa < add.length) entries.push({ op: 'changed', a: rem[pr]!, b: add[pa]!, outputDiffers: false });
      ri = pr + 1;
      aj = pa + 1;
    }
  }
  const parentsA = new Set(A.steps.map((s) => s.parent).filter((p) => p != null));
  const parentsB = new Set(B.steps.map((s) => s.parent).filter((p) => p != null));
  const container = (e: Entry) => (e.a && parentsA.has(e.a.id)) || (e.b && parentsB.has(e.b.id));
  const diverges = (e: Entry) => e.op !== 'equal' || e.outputDiffers;
  let first = entries.findIndex((e) => diverges(e) && !(e.op === 'equal' && container(e)));
  if (first < 0) first = entries.findIndex(diverges);
  return { entries, first };
}

/** Line diff (LCS) for the detail panel. */
export function lineDiff(a: string, b: string): { t: ' ' | '-' | '+'; s: string }[] {
  const al = a.split('\n');
  const bl = b.split('\n');
  return myers(al, bl).map(([op, x, y]) =>
    op === 'equal' ? { t: ' ', s: al[x]! } : op === 'removed' ? { t: '-', s: al[x]! } : { t: '+', s: bl[y]! },
  );
}

export function pretty(v: unknown): string {
  if (typeof v === 'string') return v;
  if (v && typeof v === 'object' && Array.isArray((v as { messages?: unknown }).messages)) {
    const o = v as { messages: { role?: string; content?: unknown; tool_calls?: unknown }[] };
    const lines: string[] = [];
    for (const m of o.messages) {
      lines.push(`[${m.role ?? '?'}]`);
      if (typeof m.content === 'string') lines.push(...m.content.split('\n'));
      else if (m.content != null) lines.push(canonical(m.content));
      if (m.tool_calls) lines.push(`tool_calls: ${canonical(m.tool_calls)}`);
    }
    const { messages: _m, ...rest } = v as Record<string, unknown>;
    if (Object.keys(rest).length) lines.push(JSON.stringify(rest, null, 2));
    return lines.join('\n');
  }
  return JSON.stringify(v ?? null, null, 2);
}
