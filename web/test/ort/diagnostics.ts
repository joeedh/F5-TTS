// Diagnostics shared by the browser parity runners (plans 01 and 02): node placement read from ORT's
// verbose log, and each check's error as a fraction of its tolerance.
import * as ort from "onnxruntime-web/webgpu";
import type { Result } from "../parity";

/** Node placement per session, read from ORT's verbose log while the sessions are created. */
export interface Placement {
  /** Per session in creation order: execution provider to node count. */
  sessions: Record<string, number>[];
  /** "OpType (name)" of every node placed on the CPU, per session. */
  cpuNodes: string[][];
  /** CPU nodes whose op type is not shape arithmetic; the criterion allows none. */
  offenders: string[];
  /** MemcpyTransformer's "modified" flags; 1 means copy nodes were inserted. */
  memcpyModified: number[];
}

// Ops that may sit on the CPU when they compute shapes (placement criterion, plan 01). The log gives op
// types and names, not whether the inputs are shape tensors, so the names of any CPU nodes are reported too.
const SHAPE_OPS = new Set(["Shape", "Gather", "Concat", "Range", "Slice", "Unsqueeze", "Squeeze", "Cast", "Reshape"]);

export function parsePlacement(lines: string[]): Placement {
  const sessions: Record<string, number>[] = [];
  const cpuNodes: string[][] = [];
  const memcpyModified: number[] = [];
  let ep = "";
  for (const line of lines) {
    const memcpy = /GraphTransformer MemcpyTransformer modified: (\d)/.exec(line);
    if (memcpy) memcpyModified.push(Number(memcpy[1]));
    if (line.includes("VerifyEachNodeIsAssignedToAnEp] Node placements")) {
      sessions.push({});
      cpuNodes.push([]);
      continue;
    }
    const placed = /(?:All nodes|Node\(s\)) placed on \[(\w+)\]\. Number of nodes: (\d+)/.exec(line);
    const current = sessions.at(-1);
    if (placed && current) {
      ep = placed[1] ?? "";
      current[ep] = Number(placed[2]);
      continue;
    }
    const node = /VerifyEachNodeIsAssignedToAnEp\]\s{2,}(\S+ \(.*\))\s*$/.exec(line);
    if (node && ep === "CPUExecutionProvider") cpuNodes.at(-1)?.push(node[1] ?? "");
  }
  const offenders = cpuNodes.flat().filter((n) => !SHAPE_OPS.has(n.split(" ")[0] ?? ""));
  return { sessions, cpuNodes, offenders, memcpyModified };
}

/**
 * Runs `create` (session creation) with ORT at verbose level and the console captured, since ORT prints
 * node placement there, and parses the placement. Pass logSeverityLevel 0 to the sessions it creates.
 */
export async function withPlacement<T>(create: () => Promise<T>): Promise<{ value: T; placement: Placement }> {
  const lines: string[] = [];
  const saved = {
    log: console.log,
    info: console.info,
    warn: console.warn,
    error: console.error,
    debug: console.debug,
  };
  for (const level of Object.keys(saved) as (keyof typeof saved)[]) {
    console[level] = (...args: unknown[]) => void lines.push(args.map(String).join(" "));
  }
  ort.env.logLevel = "verbose";
  try {
    const value = await create();
    return { value, placement: parsePlacement(lines) };
  } finally {
    ort.env.logLevel = "warning";
    Object.assign(console, saved);
  }
}

/** Per check: the error as a fraction of its tolerance, so near misses show as well as failures. */
export function errorSpread(results: Result[]) {
  const ratios: Record<string, { name: string; relL2: number; maxAbs: number }[]> = {};
  for (const x of results) {
    if (!x.metrics || !x.tolerance) continue;
    const rel = x.tolerance.relL2 && x.metrics.relL2 !== null ? x.metrics.relL2 / x.tolerance.relL2 : 0;
    const abs = x.tolerance.maxAbs ? x.metrics.maxAbs / x.tolerance.maxAbs : 0;
    (ratios[x.check] ??= []).push({ name: x.name, relL2: rel, maxAbs: abs });
  }
  return Object.fromEntries(
    Object.entries(ratios).map(([check, xs]) => {
      const rel = xs.map((x) => x.relL2).sort((a, b) => a - b);
      const q = (p: number) => Number((rel[Math.min(rel.length - 1, Math.floor(p * rel.length))] ?? 0).toFixed(3));
      const worst = [...xs].sort((a, b) => b.relL2 - a.relL2).slice(0, 3);
      return [check, { count: xs.length, relL2Median: q(0.5), relL2P90: q(0.9), relL2Max: q(1), worst }];
    }),
  );
}
