// Reads 16-bit PCM wav files: the input format that gets exact parity with upstream (plan 02, D5).
// Other formats are refused here; plan 06 decodes them and converts to int16.
import type { Pcm } from "./pydub";

const WAVE_FORMAT_PCM = 1;
const WAVE_FORMAT_EXTENSIBLE = 0xfffe;

export function parseWav(buf: ArrayBuffer): Pcm {
  const view = new DataView(buf);
  const tag = (at: number) => String.fromCharCode(...new Uint8Array(buf, at, 4));
  if (buf.byteLength < 12 || tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("not a RIFF WAVE file");
  let fmt: { channels: number; sampleRate: number; bits: number } | undefined;
  let at = 12;
  while (at + 8 <= buf.byteLength) {
    const id = tag(at);
    const size = view.getUint32(at + 4, true);
    const body = at + 8;
    if (id === "fmt ") {
      let format = view.getUint16(body, true);
      // WAVE_FORMAT_EXTENSIBLE keeps the real format in the first two bytes of its sub-format GUID
      if (format === WAVE_FORMAT_EXTENSIBLE && size >= 40) format = view.getUint16(body + 24, true);
      if (format !== WAVE_FORMAT_PCM) throw new Error(`wav format ${format} is not PCM`);
      fmt = {
        channels: view.getUint16(body + 2, true),
        sampleRate: view.getUint32(body + 4, true),
        bits: view.getUint16(body + 14, true),
      };
      if (fmt.bits !== 16) throw new Error(`${fmt.bits}-bit wav; only 16-bit PCM is read exactly`);
    } else if (id === "data") {
      if (!fmt) throw new Error("wav data chunk before fmt chunk");
      const bytes = Math.min(size, buf.byteLength - body);
      const frameBytes = 2 * fmt.channels;
      const count = Math.floor(bytes / frameBytes) * fmt.channels;
      const samples = new Int16Array(count);
      for (let i = 0; i < count; i++) samples[i] = view.getInt16(body + 2 * i, true);
      return { samples, channels: fmt.channels, sampleRate: fmt.sampleRate };
    }
    at = body + size + (size % 2); // chunks are padded to even sizes
  }
  throw new Error("wav has no data chunk");
}
