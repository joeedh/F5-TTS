// Reference-audio preparation (plan 02 step 2), in upstream's order: preprocess_ref_audio_text's silence
// trim and 12 s clip (pydub, exact for 16-bit PCM), then infer_batch_process's downmix, RMS raise and
// resample to 24 kHz.
import { detectLeadingSilence, Segment, splitOnSilence, type Pcm } from "./pydub";
import { resample } from "./resample";
import { SAMPLE_RATE } from "./mel";

export const TARGET_RMS = 0.1;

/** utils_infer.remove_silence_edges: trims leading and trailing audio below -42 dBFS. */
function removeSilenceEdges(audio: Segment, silenceThreshold = -42): Segment {
  const start = detectLeadingSilence(audio, silenceThreshold);
  const trimmed = audio.slice(start);
  const end = detectLeadingSilence(trimmed.reverse(), silenceThreshold);
  return end > 0 ? trimmed.slice(0, trimmed.lengthMs - end) : trimmed;
}

/** Joins segments in order, as upstream's clipping loop does: once over 6 s, it stops before passing 12 s. */
function joinUpTo12s(segs: Segment[]): Segment {
  let wave = Segment.silent(0);
  for (const seg of segs) {
    if (wave.lengthMs > 6000 && wave.append(seg).lengthMs > 12000) break;
    wave = wave.append(seg);
  }
  return wave;
}

/**
 * preprocess_ref_audio_text's audio part: split on long silences and keep about 12 s, else on short
 * ones, else cut at 12 s; trim the edges; append pydub's 50 ms of silence. The output keeps the input's
 * channels, and its rate unless the input is below 11025 Hz. The input is the decoded file at its own
 * rate (plan 02, D4).
 */
export function trimAndClip(input: Pcm): Pcm {
  const aseg = Segment.fromPcm(input);
  let wave = joinUpTo12s(splitOnSilence(aseg, 1000, -50, 1000, 10));
  if (wave.lengthMs > 12000) wave = joinUpTo12s(splitOnSilence(aseg, 100, -40, 1000, 10));
  if (wave.lengthMs > 12000) wave = wave.slice(0, 12000);
  return removeSilenceEdges(wave).append(Segment.silent(50)).toPcm();
}

export interface Reference {
  /** 24 kHz mono, the input to the mel (golden `audio.ref`). */
  audio: Float32Array;
  /** RMS of the downmixed audio before any raise; the generated audio is rescaled by it when `raised`. */
  rms: number;
  raised: boolean;
}

/** infer_batch_process's reference handling: mean of channels, raise to RMS 0.1 if quieter, resample. */
export function downmixRmsResample(prepared: Pcm): Reference {
  const { samples, channels, sampleRate } = prepared;
  const frames = samples.length / channels;
  const mono = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let s = 0;
    for (let c = 0; c < channels; c++) s += (samples[i * channels + c] as number) / 32768;
    mono[i] = s / channels;
  }
  let sum = 0;
  for (const v of mono) sum += v * v;
  const rms = Math.fround(Math.sqrt(sum / frames));
  const raised = rms < TARGET_RMS;
  if (raised) {
    // audio * target_rms / rms in float32
    const k = Math.fround(TARGET_RMS);
    for (let i = 0; i < frames; i++) mono[i] = Math.fround(Math.fround((mono[i] as number) * k) / rms);
  }
  return { audio: resample(mono, sampleRate, SAMPLE_RATE), rms, raised };
}
