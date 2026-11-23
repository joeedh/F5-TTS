// Plan 04 step 6: the API's defaults are upstream's, read from utils_infer.py's module constants.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, it } from "vitest";
import { DEFAULTS } from "../../src/pipeline";

const source = readFileSync(join(import.meta.dirname, "../../../src/f5_tts/infer/utils_infer.py"), "utf8");

function constant(name: string): number | null {
  const m = new RegExp(String.raw`^${name} = (\S+)`, "m").exec(source);
  if (!m) throw new Error(`${name} not found in utils_infer.py`);
  return m[1] === "None" ? null : Number(m[1]);
}

it("equals utils_infer.py's defaults", () => {
  expect(DEFAULTS).toEqual({
    nfeStep: constant("nfe_step"),
    cfgStrength: constant("cfg_strength"),
    swaySamplingCoef: constant("sway_sampling_coef"),
    speed: constant("speed"),
    crossFadeDuration: constant("cross_fade_duration"),
    fixDuration: constant("fix_duration"),
    targetRms: constant("target_rms"),
  });
});

it("is what infer_process and infer_batch_process default to", () => {
  // both take the module constants as their defaults, apart from infer_batch_process's literal copies
  for (const fn of ["infer_process", "infer_batch_process"]) {
    const start = source.indexOf(`def ${fn}(`);
    const sig = source.slice(start, source.indexOf(")", source.indexOf("fix_duration", start)));
    const literals: [string, string][] = [
      ["nfe_step", "32"],
      ["cfg_strength", "2.0"],
      ["sway_sampling_coef", "-1"],
      ["speed", "1"],
      ["cross_fade_duration", "0.15"],
      ["fix_duration", "None"],
      ["target_rms", "0.1"],
    ];
    for (const [arg, value] of literals) {
      const literal = value.replace(".", String.raw`\.`);
      expect(sig, `${fn} ${arg}`).toMatch(new RegExp(String.raw`${arg}: [^=]+= (${arg}|${literal})\b`));
    }
  }
});
