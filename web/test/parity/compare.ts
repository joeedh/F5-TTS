// Compares a port's tensor with the golden one, using testdata/webgpu/tolerances.json.
// Python twin: tools/webgpu/parity.py. The two must agree on check names, metrics and pass rules.
import type { CheckTolerance, GoldenCase, Tolerances } from "./case";
import type { Tensor } from "./npy";

/** Pipeline order; the report names the earliest failing stage, since later ones inherit its error. */
export const STAGE_ORDER = [
  "input",
  "prep",
  "text",
  "audio",
  "mel",
  "plan",
  "noise",
  "dit",
  "dit_internal",
  "mel_out",
  "vocoder",
  "final",
];

/** Checks that are reported but never fail (see the notes in tolerances.json). */
export const INFORMATIONAL = new Set(["audio.resample", "vocoder.head_out"]);

const LOG_MEL_MIN = Math.log(1e-4);

/**
 * Which tolerance checks apply to a golden tensor. Inputs (noise, x_k, step_cond, the time value
 * fed to a DiT call, raw audio) have none: a port receives them, it does not produce them.
 * `c*.mel_out` is only compared when the port runs the ODE itself (free running).
 */
export function checksFor(name: string): string[] {
  const n = name.replace(/^c\d+\./, "");
  if (n === "plan.t") return ["plan.t"];
  if (name === "audio.ref") return ["audio.resample"];
  if (name === "mel.ref") return ["mel.log_masked", "mel.linear"];
  if (n === "dit.text_ids") return ["exact"];
  if (/^dit\.s\d+\.out_(cond|uncond)$/.test(n)) return ["dit.out"];
  const internal = /^dit_internal\.(?:s\d+\.)?([a-z_]+?)(\d+)?(?:_uncond)?$/.exec(n);
  if (internal) return [internal[1] === "block" ? "dit_internal.block" : `dit_internal.${internal[1]}`];
  if (n === "mel_out") return ["mel_out.free_running"];
  if (n === "vocoder.backbone") return ["vocoder.backbone"];
  if (n === "vocoder.head_out") return ["vocoder.head_out", "vocoder.stft_complex"];
  if (n === "vocoder.wav") return ["vocoder.wav"];
  return [];
}

export interface Metrics {
  maxAbs: number;
  relL2: number | null; // null where the golden norm is near zero
  cos: number | null;
}

export function metrics(got: ArrayLike<number>, ref: ArrayLike<number>, mask?: (i: number) => boolean): Metrics {
  let dd = 0;
  let bb = 0;
  let aa = 0;
  let ab = 0;
  let maxAbs = 0;
  let n = 0;
  for (let i = 0; i < ref.length; i++) {
    if (mask && !mask(i)) continue;
    const a = got[i] as number;
    const b = ref[i] as number;
    const d = a - b;
    dd += d * d;
    bb += b * b;
    aa += a * a;
    ab += a * b;
    // NaN must fail, and Math.max ignores nothing: max(x, NaN) is NaN
    maxAbs = Number.isNaN(d) ? NaN : Math.max(maxAbs, Math.abs(d));
    if (Number.isNaN(maxAbs)) break;
    n++;
  }
  const nb = Math.sqrt(bb);
  const small = nb < 1e-6 * n;
  return {
    maxAbs,
    relL2: small ? null : Math.sqrt(dd) / nb,
    cos: small || aa === 0 ? null : ab / (Math.sqrt(aa) * nb),
  };
}

/** Vocos ISTFTHead: the first half of the last axis is log magnitude (exp, clipped at 100), the second phase. */
export function complexStft(headOut: Tensor): Float64Array {
  const f = headOut.shape[headOut.shape.length - 1] as number;
  const half = f / 2;
  const rows = headOut.data.length / f;
  const out = new Float64Array(headOut.data.length);
  for (let r = 0; r < rows; r++) {
    for (let k = 0; k < half; k++) {
      const mag = Math.min(Math.exp(headOut.data[r * f + k] as number), 100);
      const phase = headOut.data[r * f + half + k] as number;
      out[r * f + k] = mag * Math.cos(phase);
      out[r * f + half + k] = mag * Math.sin(phase);
    }
  }
  return out;
}

export interface Result {
  stage: string;
  name: string;
  check: string;
  pass: boolean;
  informational: boolean;
  error?: string; // "shape ..." or "dtype ..." when the tensors cannot be compared
  metrics?: Metrics;
  tolerance?: { relL2?: number; maxAbs?: number };
}

function sameShape(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

/** Compares one port output with the golden tensor of the same name; one Result per applicable check. */
export async function compare(
  golden: GoldenCase,
  tolerances: Tolerances,
  name: string,
  actual: Tensor,
): Promise<Result[]> {
  const entry = golden.entry(name);
  const checks = checksFor(name);
  if (checks.length === 0) throw new Error(`${name} is an input; there is no check for it`);
  const base = { stage: entry.stage, name };
  if (!sameShape(actual.shape, entry.shape)) {
    const error = `shape [${actual.shape.join(", ")}] but golden is [${entry.shape.join(", ")}] (${entry.axes})`;
    return checks.map((check) => ({ ...base, check, pass: false, informational: INFORMATIONAL.has(check), error }));
  }
  if (actual.dtype !== entry.dtype) {
    const error = `dtype ${actual.dtype} but golden is ${entry.dtype}`;
    return checks.map((check) => ({ ...base, check, pass: false, informational: INFORMATIONAL.has(check), error }));
  }
  const ref = await golden.tensor(name);
  return checks.map((check): Result => {
    const informational = INFORMATIONAL.has(check);
    if (check === "exact") {
      const same = actual.data.every((v, i) => v === ref.data[i]);
      return { ...base, check, pass: same, informational, ...(same ? {} : { error: "values differ" }) };
    }
    let m: Metrics;
    if (check === "mel.log_masked") m = metrics(actual.data, ref.data, (i) => (ref.data[i] as number) > LOG_MEL_MIN);
    else if (check === "mel.linear")
      m = metrics(Float64Array.from(actual.data, Math.exp), Float64Array.from(ref.data, Math.exp));
    else if (check === "vocoder.stft_complex") m = metrics(complexStft(actual), complexStft(ref));
    else m = metrics(actual.data, ref.data);
    return judge(golden, tolerances, entry.stage, name, check, m);
  });
}

/**
 * Applies a check's tolerance to measured metrics. Also used for values derived from golden tensors
 * (plan 01's dit.v and dit.x_next), which have no golden tensor of their own. Python: parity.judge.
 */
export function judge(
  golden: GoldenCase,
  tolerances: Tolerances,
  stage: string,
  name: string,
  check: string,
  m: Metrics,
): Result {
  const informational = INFORMATIONAL.has(check);
  const shared = tolerances.checks[check];
  if (!shared) throw new Error(`no tolerance for ${check} in tolerances.json`);
  // the tiny case has its own tolerance where its floor is higher than the real cases' (plan 00, R5)
  const tol: CheckTolerance = golden.name === "tiny" && shared.tiny ? { ...shared, ...shared.tiny } : shared;
  const tolerance = { relL2: tol.rel_l2, maxAbs: tol.max_abs };
  // near-zero golden: relative L2 is undefined, so max abs decides, against the rel_l2 value if there is no max_abs
  const absLimit = tol.max_abs ?? (m.relL2 === null ? tol.rel_l2 : undefined);
  const okAbs = absLimit === undefined || m.maxAbs <= absLimit;
  const okRel = tol.rel_l2 === undefined || m.relL2 === null || m.relL2 <= tol.rel_l2;
  const pass = !Number.isNaN(m.maxAbs) && okAbs && okRel;
  return { stage, name, check, pass: pass || informational, informational, metrics: m, tolerance };
}

function fmt(x: number | null | undefined): string {
  return x === null || x === undefined ? "-" : x.toExponential(2);
}

/** One line per comparison; the earliest failing stage (pipeline order) is printed last. */
export function report(results: Result[]): { text: string; firstFailingStage: string | null; failed: number } {
  const lines = results.map((r) => {
    const status = r.informational ? "INFO" : r.pass ? "ok  " : "FAIL";
    const detail = r.error
      ? r.error
      : `relL2 ${fmt(r.metrics?.relL2)} (tol ${fmt(r.tolerance?.relL2)})  maxAbs ${fmt(r.metrics?.maxAbs)} (tol ${fmt(r.tolerance?.maxAbs)})`;
    return `${status} ${r.stage.padEnd(12)} ${r.check.padEnd(26)} ${r.name.padEnd(40)} ${detail}`;
  });
  const failures = results.filter((r) => !r.pass);
  const first = failures.sort((a, b) => STAGE_ORDER.indexOf(a.stage) - STAGE_ORDER.indexOf(b.stage))[0];
  lines.push(
    first
      ? `FIRST FAILING STAGE: ${first.stage} (${first.check}, ${first.name}); ${failures.length} failure(s)`
      : `all ${results.length} comparisons pass`,
  );
  return { text: lines.join("\n"), firstFailingStage: first?.stage ?? null, failed: failures.length };
}
