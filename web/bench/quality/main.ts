// Plan 04 step 7 (D5): the port's side of the quality comparison. run.mjs loads this page, calls init() once,
// then generate() per utterance of data/seedtts_testset/quality_subset.json, with the utterance's seed.
import { parseWav } from "../../src/audio/wav";
import { TTS } from "../../src/pipeline";

interface Row {
  utt: string;
  prompt_text: string;
  prompt_wav: string;
  target_text: string;
  seed: number;
}

let tts: TTS | undefined;

async function init(): Promise<{ adapter: string }> {
  tts = await TTS.create({
    dit: { textUrl: "/model/dit/base/text.onnx", stepUrl: "/model/dit/base/step_mha.onnx" },
    vocoder: { url: "/model/vocos/vocos.onnx" },
    vocabUrl: "/vocab/vocab.txt",
  });
  const adapter = await navigator.gpu.requestAdapter();
  return { adapter: adapter?.info.vendor ?? "unknown" };
}

/** One utterance; the wave comes back as base64 float32 so the page hands over bytes, not a JSON array. */
async function generate(row: Row): Promise<{ wave: string; seconds: number; ms: number; live: number }> {
  if (!tts) throw new Error("init() first");
  const res = await fetch(`/seedtts/${row.prompt_wav}`);
  if (!res.ok) throw new Error(`${row.prompt_wav}: HTTP ${res.status}`);
  const t0 = performance.now();
  const out = await tts.generate({
    refAudio: parseWav(await res.arrayBuffer()),
    refText: row.prompt_text,
    genText: row.target_text,
    settings: { seed: row.seed },
  });
  const ms = Math.round(performance.now() - t0);
  const bytes = new Uint8Array(out.wave.buffer, out.wave.byteOffset, out.wave.byteLength);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return { wave: btoa(bin), seconds: out.wave.length / out.sampleRate, ms, live: tts.liveTensors };
}

Object.assign(globalThis, { qualityBench: { init, generate } });
