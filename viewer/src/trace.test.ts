import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { canonical, diffRuns, myers, parseRun } from './trace';

describe('viewer diff matches the Python diff', () => {
  it('finds the same first divergence on the example runs', () => {
    const a = parseRun(readFileSync('public/samples/before.jsonl', 'utf8'));
    const b = parseRun(readFileSync('public/samples/after.jsonl', 'utf8'));
    const expected = JSON.parse(readFileSync('../results/case-study-diff.json', 'utf8'));
    const d = diffRuns(a, b);
    const e = d.entries[d.first]!;
    expect({ op: e.op, a: e.a?.id ?? null, b: e.b?.id ?? null }).toEqual({
      op: expected.first_divergence.op,
      a: expected.first_divergence.a?.id ?? null,
      b: expected.first_divergence.b?.id ?? null,
    });
    expect(d.entries.map((x) => x.op)).toEqual(expected.entries.map((x: { op: string }) => x.op));
  });
  it('myers rebuilds both sides', () => {
    const a = 'abcabba'.split('');
    const b = 'cbabac'.split('');
    const ops = myers(a, b);
    expect(ops.filter(([o]) => o !== 'added').map(([, i]) => a[i])).toEqual(a);
    expect(ops.filter(([o]) => o !== 'removed').map(([, , j]) => b[j])).toEqual(b);
    expect(ops.filter(([o]) => o === 'equal')).toHaveLength(4);
  });
  it('canonical sorts keys and rejects bad input', () => {
    expect(canonical({ b: 1, a: [true, null] })).toBe('{"a":[true,null],"b":1}');
    expect(() => parseRun('{')).toThrow(/line 1/);
    expect(() => parseRun('{"format":"x"}')).toThrow(/unsupported/);
  });
});
