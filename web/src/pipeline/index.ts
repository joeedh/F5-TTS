// The generation API (plan 04 step 6, D6): load the models once, then generate from a reference and texts,
// with progress, a per-chunk callback for early playback, abort, and release. No DOM, so it runs in a worker.
import * as ort from "onnxruntime-web/webgpu";
import type { Pcm } from "../audio/pydub";
import { DiT, type DiTOptions } from "../dit";
import { parseVocab, TextFrontend, type TextFrontendLoaders } from "../text/tokens";
import { Vocoder, type VocoderOptions } from "../vocoder";
import { DEFAULTS, generateWith, type Generated, type GenerateHooks, type Models, type Settings } from "./generate";
import { TensorLedger } from "./sample";

export { DEFAULTS, type Generated, type GenerateHooks, type Progress, type Settings } from "./generate";

export interface TTSOptions {
  dit: DiTOptions;
  vocoder: VocoderOptions;
  /** URL of vocab.txt. */
  vocabUrl: string;
  /** Loaders for the jieba wasm and the pinyin dictionary; by default the files next to src/text/. */
  text?: TextFrontendLoaders;
}

export interface GenerateRequest {
  /** The reference file, decoded at its own rate (parseWav). */
  refAudio: Pcm;
  /** What the reference says. Must not be empty: upstream would transcribe it, which is not supported. */
  refText: string;
  genText: string;
  /** Upstream's defaults where left out. A missing seed is drawn at random and returned. */
  settings?: Partial<Settings>;
}

/** A random uint32, as the seed when none is given. */
function randomSeed(): number {
  return crypto.getRandomValues(new Uint32Array(1))[0] as number;
}

export class TTS {
  private busy = false;

  private constructor(private readonly models: Models) {}

  static async create(options: TTSOptions): Promise<TTS> {
    const vocab = await fetch(options.vocabUrl).then((r) => {
      if (!r.ok) throw new Error(`${options.vocabUrl}: HTTP ${r.status}`);
      return r.text();
    });
    // one after another: ORT Web refuses to create two WebGPU sessions at once (docs/debugging.md)
    const dit = await DiT.create(options.dit);
    let vocoder: Vocoder;
    try {
      vocoder = await Vocoder.create(options.vocoder);
    } catch (e) {
      await dit.release();
      throw e;
    }
    const device = await ort.env.webgpu.device;
    const models: Models = {
      dit,
      vocoder,
      vocab: parseVocab(vocab),
      text: new TextFrontend(options.text),
      device,
      ledger: new TensorLedger(),
    };
    return new TTS(models);
  }

  /** GPU tensors the pipeline holds right now: 0 between generations, unless something leaked. */
  get liveTensors(): number {
    return this.models.ledger.live;
  }

  /** One generation at a time; a second call while one runs is refused. */
  async generate(request: GenerateRequest, hooks: GenerateHooks = {}): Promise<Generated> {
    if (this.busy) throw new Error("TTS.generate: a generation is already running");
    const settings: Settings = { ...DEFAULTS, ...request.settings, seed: request.settings?.seed ?? randomSeed() };
    if (!Number.isInteger(settings.seed) || settings.seed < 0 || settings.seed > 0xffffffff)
      throw new Error("seed must be a uint32");
    this.busy = true;
    try {
      return await generateWith(this.models, request, settings, hooks);
    } finally {
      this.busy = false;
    }
  }

  async release(): Promise<void> {
    await this.models.dit.release();
    await this.models.vocoder.release();
  }
}
