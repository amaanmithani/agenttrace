import { useEffect, useMemo, useState } from 'react';
import { depthOf, diffRuns, durationMs, lineDiff, parseRun, pretty, type Entry, type Run, type Step } from './trace';

const WHAT: Record<Entry['op'], string> = {
  changed: 'same call, different input',
  removed: 'only in run A',
  added: 'only in run B',
  equal: 'same input, different result',
};

function short(v: unknown, n = 64): string {
  const s = (typeof v === 'string' ? v : JSON.stringify(v ?? null)).replace(/\s+/g, ' ');
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

function StepLabel({ s }: { s: Step }) {
  return (
    <span className="label">
      <span className={`kind kind-${s.kind}`}>{s.kind}</span>
      <span className="name">{s.name}</span>
      <span className="id">#{s.id}</span>
      {s.error && <span className="err">error</span>}
    </span>
  );
}

function Diff({ a, b }: { a: unknown; b: unknown }) {
  const lines = lineDiff(pretty(a), pretty(b));
  return (
    <pre className="code">
      {lines.map((l, i) => (
        <div key={i} className={l.t === '-' ? 'del' : l.t === '+' ? 'ins' : undefined}>
          <span className="sign">{l.t}</span>
          {l.s}
        </div>
      ))}
    </pre>
  );
}

function Detail({ a, b }: { a?: Step; b?: Step }) {
  const s = a ?? b;
  if (!s) return <aside className="detail empty">Select a step to see what went in and what came out.</aside>;
  const both = a && b;
  return (
    <aside className="detail">
      <h2>
        <StepLabel s={s} />
      </h2>
      <p className="meta">
        {both ? `A ${durationMs(a).toFixed(1)} ms · B ${durationMs(b).toFixed(1)} ms` : `${durationMs(s).toFixed(1)} ms`}
      </p>
      {(['input', 'output'] as const).map((f) => (
        <section key={f}>
          <h3>{f === 'input' ? 'Input' : 'Output'}</h3>
          {both ? <Diff a={a[f]} b={b[f]} /> : <pre className="code">{pretty(s[f])}</pre>}
        </section>
      ))}
      {(a?.error || b?.error) && (
        <section>
          <h3>Error</h3>
          <pre className="code">{both ? `A: ${a.error ?? '—'}\nB: ${b.error ?? '—'}` : s.error}</pre>
        </section>
      )}
    </aside>
  );
}

function Single({ run, pick, picked }: { run: Run; pick: (s: Step) => void; picked?: Step }) {
  const depth = useMemo(() => depthOf(run), [run]);
  return (
    <ol className="rows single">
      {run.steps.map((s) => (
        <li key={s.id} className={picked?.id === s.id ? 'row on' : 'row'}>
          <button onClick={() => pick(s)} style={{ paddingLeft: `${0.75 + (depth.get(s.id) ?? 0) * 1.1}rem` }}>
            <span className="dot" />
            <StepLabel s={s} />
            <span className="peek">{short(s.output)}</span>
          </button>
        </li>
      ))}
    </ol>
  );
}

function Fork({ entries, first, pick, picked }: { entries: Entry[]; first: number; pick: (e: Entry) => void; picked?: Entry }) {
  return (
    <ol className="rows fork">
      {entries.map((e, i) => {
        const shared = first < 0 || i < first;
        const cls = ['row', shared ? 'shared' : 'split', i === first ? 'first' : '', picked === e ? 'on' : '', e.op].join(' ');
        return (
          <li key={i} className={cls}>
            {i === first && (
              <div className="forkmark" role="note">
                Runs part ways here: {(e.a ?? e.b)!.kind} {(e.a ?? e.b)!.name}, {WHAT[e.op]}
              </div>
            )}
            <button onClick={() => pick(e)} aria-label={`step ${(e.a ?? e.b)!.name}, ${e.op}`}>
              {shared ? (
                <span className="cell both">
                  <span className="dot" />
                  <StepLabel s={e.a!} />
                  <span className="peek">{short(e.a!.output)}</span>
                </span>
              ) : (
                <>
                  <span className="cell a">
                    <span className="dot" />
                    {e.a ? <StepLabel s={e.a} /> : <span className="none">no step</span>}
                  </span>
                  <span className="cell b">
                    <span className="dot" />
                    {e.b ? <StepLabel s={e.b} /> : <span className="none">no step</span>}
                  </span>
                  <span className="tag">{e.op === 'equal' ? (e.outputDiffers ? 'result differs' : 'same') : e.op}</span>
                </>
              )}
            </button>
          </li>
        );
      })}
    </ol>
  );
}

async function readFile(f: File): Promise<Run> {
  return parseRun(await f.text());
}

export function App() {
  const [a, setA] = useState<Run | null>(null);
  const [b, setB] = useState<Run | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [pickedStep, setPickedStep] = useState<Step>();
  const [pickedEntry, setPickedEntry] = useState<Entry>();
  const [theme, setTheme] = useState<'light' | 'dark' | null>(null);

  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
  }, [theme]);

  const diff = useMemo(() => (a && b ? diffRuns(a, b) : null), [a, b]);

  const load = async (which: 'a' | 'b', files: FileList | null) => {
    const f = files?.[0];
    if (!f) return;
    try {
      const run = await readFile(f);
      setErr(null);
      (which === 'a' ? setA : setB)(run);
      setPickedEntry(undefined);
      setPickedStep(undefined);
    } catch (e) {
      setErr(`${f.name}: ${(e as Error).message}`);
    }
  };

  const loadSample = async () => {
    try {
      const [ta, tb] = await Promise.all(
        ['samples/before.jsonl', 'samples/after.jsonl'].map((p) => fetch(p).then((r) => (r.ok ? r.text() : Promise.reject(new Error(`${p}: ${r.status}`))))),
      );
      setA(parseRun(ta!));
      setB(parseRun(tb!));
      setErr(null);
      setPickedEntry(undefined);
    } catch (e) {
      setErr(`Couldn't load the sample runs: ${(e as Error).message}`);
    }
  };

  useEffect(() => {
    if (new URLSearchParams(location.search).has('sample')) void loadSample();
  }, []);

  useEffect(() => {
    if (diff && diff.first >= 0 && !pickedEntry) setPickedEntry(diff.entries[diff.first]);
  }, [diff, pickedEntry]);

  const summary = (() => {
    if (!diff) return null;
    if (diff.first < 0) return `Identical: ${diff.entries.length} steps, same requests and same results.`;
    const e = diff.entries[diff.first]!;
    const s = (e.a ?? e.b)!;
    return `The runs agree for ${diff.first} step${diff.first === 1 ? '' : 's'}, then part ways at ${s.kind} ${s.name}: ${WHAT[e.op]}.`;
  })();

  return (
    <div className="app">
      <header>
        <h1>AgentTrace</h1>
        <div className="actions">
          <label className="file a">
            <input type="file" accept=".jsonl,.json,.txt" onChange={(e) => void load('a', e.target.files)} />
            {a ? `Run A · ${a.steps.length} steps` : 'Open run A'}
          </label>
          <label className="file b">
            <input type="file" accept=".jsonl,.json,.txt" onChange={(e) => void load('b', e.target.files)} />
            {b ? `Run B · ${b.steps.length} steps` : 'Open run B to compare'}
          </label>
          <button className="ghost" onClick={() => void loadSample()}>
            Load the example
          </button>
          <button
            className="ghost"
            aria-label="Toggle dark mode"
            onClick={() => setTheme((t) => ((t ?? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark' ? 'light' : 'dark'))}
          >
            Theme
          </button>
        </div>
      </header>
      {err && <p className="error" role="alert">{err}</p>}
      {summary && <p className="summary">{summary}</p>}
      <main>
        {!a && !b ? (
          <section className="intro">
            <h2>Where did this run go differently?</h2>
            <p>
              Open a run recorded with <code>agenttrace</code> to see every model call, tool call and graph node as a tree.
              Open a second run and the two are aligned step by step: they share one rail while they agree and split at the
              first step where they don't.
            </p>
            <p>
              <button onClick={() => void loadSample()}>Load the example</button> a support agent before and after its
              order-lookup tool changed its output format.
            </p>
          </section>
        ) : diff ? (
          <>
            <Fork entries={diff.entries} first={diff.first} pick={setPickedEntry} picked={pickedEntry} />
            <Detail a={pickedEntry?.a} b={pickedEntry?.b} />
          </>
        ) : (
          <>
            <Single run={(a ?? b)!} pick={setPickedStep} picked={pickedStep} />
            <Detail a={pickedStep} />
          </>
        )}
      </main>
    </div>
  );
}
