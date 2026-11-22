# Plan 03: Text front end

Status: `plan pressure-tested` (written 2026-11-18; review findings folded in, see the end of this file). D4 (test corpus) decided by the user on 2026-11-18: the Seed-TTS eval lists. Depends on: plan 00 (done). Blocks: plan 04.

Task list: [webgpu-port-tasks.md](webgpu-port-tasks.md). Golden data: [webgpu-00-scope-and-parity.md](webgpu-00-scope-and-parity.md) (the `text` stage, `manifest.chunks[i]`, and `c*.dit.text_ids`).

## What this plan delivers

- TypeScript that turns a reference text and a target text into what plan 04 needs:
  - The fixed-up reference text.
  - `max_chars`.
  - The text chunks.
  - For each chunk, the token list and the int32 token ids, identical to Python's.
- A measured mismatch rate against Python on a corpus of English and Chinese text. The target is zero.
- Text with no CJK characters handled without loading the Chinese data, if step 3 proves a pure-TS path exact. Otherwise the segmenter loads for all text.

Out of scope:
- The duration estimate and the short-chunk speed rule. They use the text's byte lengths, but they belong to plan 04's sampler. This plan exports the byte-length helper they use.
- ASR for the reference text (excluded in the task list's starting assumptions).

## What upstream does

### Strings (`infer/utils_infer.py`)

- **Reference text fix-up** (end of `preprocess_ref_audio_text`):
  - If the text ends with neither `". "` nor `"。"`, then: if it ends with `"."`, append `" "`; otherwise append `". "`.
- **`max_chars`** (`infer_process`, `utils_infer.py:415`): `int(len(ref_text.encode("utf-8")) / (n / sr) * (22 - n / sr) * speed)`.
  - `ref_text` is the fixed-up text, before the trailing-space rule (which comes later, at `:488`).
  - `n` and `sr` are the prepared reference audio's sample count and rate (plan 02's output, before resampling).
  - The TS port uses the same operation order in float64, then `Math.trunc`.
- **`chunk_text(text, max_chars)`:**
  - Splits on `(?<=[;:,.!?])\s+|(?<=[；：，。！？])`, which is whitespace after ASCII punctuation, or the empty string after CJK punctuation. JS `split` with the same regex gives the same pieces (checked: `'a，b。c. d'` gives `['a，','b。','c.','d']` in both).
  - Greedily joins sentences while the UTF-8 byte count stays within `max_chars`.
  - A sentence whose last character is 1 byte in UTF-8 gets a trailing `" "` when appended.
  - Chunks are `strip()`ped. Empty pieces are skipped.
- **Trailing-space rule** (`infer_batch_process`): if the reference text's last character is 1 byte in UTF-8, append `" "`.
  - Together with the fix-up this gives two spaces before the generated text. The golden `ref_text_spaced` ends in `".  "`.
- **Joined text per chunk:** `ref_text + chunk`.
- **Python against JS whitespace** (all code points enumerated, Python 3.15.1):
  - Python's `\s` and `strip()` (the same set) include U+001C to U+001F and U+0085, and exclude U+FEFF.
  - JS's `\s` and `trim()` include U+FEFF, and exclude U+001C to U+001F and U+0085.
  - The TS port uses an explicit character class with Python's set, never `\s` or `trim()`.

### Tokens (`model/utils.py`)

`convert_char_to_pinyin([text])`:
1. `str.translate` maps `;` to `,`, and curly quotes (`“” ‘’`) to straight ones.
2. `rjieba.cut(text)` splits into words. rjieba 0.2.2 embeds jieba-rs 0.11.0 (from its SBOM), with `hmm=True` by default.
   - English text goes through jieba too, and its splits change tokens. rjieba keeps `hello-world`, `e.g`, `foo_bar`, `C++` and `AT&T` as single segments.
   - `C++`, `C#` and `AT&T` stay whole because they are dictionary words. `dict.txt` has exactly 5 pure-ASCII entries: `AT&T C# C++ c# c++`.
   - Apostrophes split: `don't` becomes `don`, `'`, `t`.
3. For each segment, based on its UTF-8 length:
   - **ASCII only** (bytes == characters): if the list is not empty, the segment is longer than 1 byte, and the previous token is not a substring of `` :'"`` (Python `in` on a string), append `" "`. Then append each character.
     - So a split changes tokens. Segments `['hello','-','world']` give `hello- world`, while the single segment `hello-world` gives no space.
   - **All 3-byte characters** (bytes == 3 x characters; this includes CJK punctuation such as `，`):
     - `seg_ = lazy_pinyin(seg, style=TONE3, tone_sandhi=True)`, indexed per character as `seg_[i]`.
     - `lazy_pinyin` merges runs of non-Han characters into one item, so the indexing assumes it does not. No current jieba output was found that breaks this, but the port copies the indexing exactly, and the dump asserts `len(seg_) == len(seg)`.
     - Before each character in U+3100 to U+9FFF (`is_chinese`), a `" "` token is appended. Then `seg_[i]` is appended.
   - **Mixed:**
     - Characters below U+0100 are appended as they are.
     - Characters in the `is_chinese` range get `" "` and then `lazy_pinyin(c)` of that character alone.
     - Anything else is appended as it is.
- `is_chinese` (U+3100 to U+9FFF) and pypinyin's Han range (`RE_HANS`) differ:
  - `〇` (U+3007) gives `ling2` with no space before it: outside `is_chinese`, inside `RE_HANS`.
  - Bopomofo and enclosed CJK such as `ㄅ`, `㈠` and `䷀` get `" "` and then themselves.
  - U+F900 compatibility ideographs pass through unchanged: `"豈㐀"` gives `['豈', ' ', 'qiu1']`.

### pypinyin 0.55.0's `lazy_pinyin` path

- `simpleseg` first splits Han from non-Han with `RE_HANS`.
- `mmseg` runs on each Han run with `no_non_phrases=True`: forward maximum matching, joining only strings that are keys of `PHRASES_DICT` (47,111 entries). It is not a trained model (`seg/mmseg.py:120`).
- Each word is looked up in `phrases_dict` (2.5 MB of JSON), and single characters in `pinyin_dict` (0.8 MB).
- Tone sandhi (`contrib/tone_sandhi.py`) applies only within one pypinyin word:
  - The golden zh-short shows this. `版本` gives `ban3 ben3`, because mmseg splits it into 版 / 本, and `一次` gives `yi1 ci4`, not `yi2`.
  - The third-tone rule (`_third_tone`) only looks at the run of third tones that ends the word. `两点论`, `九尾狐` and `五斗柜` all stay 3-3-x.
- TONE3 formatting: a neutral tone has no digit, as in `de`, `men` and `le` in the golden data.

### Ids

- `vocab.txt`: 2545 entries, index 0 is `" "`, and the last entry is `𠮶` (outside the BMP).
- The file is stored with LF, but this checkout has `core.autocrlf=true`, so the working tree has CRLF (`git ls-files --eol`: `i/lf w/crlf`). Python's text mode hides that. A JS `split('\n')` would leave `\r` on every entry, every lookup would miss, and every id would be 0.
- Unknown tokens map to 0 (`vocab_char_map.get(c, 0)`).

## Decisions

### D1. Word segmentation (proposal: our own wasm build of jieba-rs 0.11.0)

- `jieba-wasm` 2.4.0 does not match. Its `jieba-rs` is a git submodule pointing at a fork at version 0.7.0 (2024-06-01).
  - Its `dict.txt` and `hmm.model` are byte-identical to 0.11.0's.
  - The segmentation code is not. 0.7.0's HMM skip regex is `([a-zA-Z0-9]+(?:.\d+)?%?)`. 0.11.0 replaced it with `skip_match_len`, which joins runs across `.`, `_` and `-`. That is exactly where `hello-world` and `e.g` would split, which changes tokens (see above).
- Proposal: build a small wasm package from jieba-rs 0.11.0 (the version rjieba's SBOM names) with `wasm-bindgen`, exposing `cut(text, hmm=true)`.
  - Same engine, same dictionary, same version, so it should match by construction. The corpus confirms it.
  - Cost: a Rust toolchain for whoever rebuilds it, so the built `.wasm` is committed with a script that rebuilds it and records the jieba-rs version. Plan 05's hosting can carry it instead, if it is large.
  - The dictionary (5 MB raw) is embedded in the wasm. Measure the compressed transfer size.
- Fallback: port jieba-rs 0.11.0 to TS. Port the Rust code, not Python jieba, whose Han regex differs.
  - The pieces: prefix-dictionary DAG, max-probability route, HMM for unknown words.
  - Floating-point log sums and route tie-breaking need care. V8's `Math.log` is not guaranteed to be correctly rounded, so log-frequencies are precomputed in Python and shipped as data, and ties are tested explicitly.
- `jieba-wasm` stays only as a measured comparison in step 3, to show what the version gap costs.

### D2. Text with no CJK characters (proposal: pure TS, if proven exact)

- English needs jieba's segments, because of the space rule. A pure-TS path for text with no CJK characters would have to reproduce:
  - jieba-rs 0.11.0's character-class split of non-Han text, including `skip_match_len`.
  - The 5 ASCII dictionary words.
- Proposal: write that path, and prove it against rjieba on the English corpus and the edge cases. If it matches, text with no CJK never loads the wasm. If it does not, the wasm loads for all text, and the size goes into plan 06's loading budget.

### D3. Pinyin (proposal: port pypinyin's `lazy_pinyin` path to TS, with pypinyin's own dictionaries)

- JS pinyin libraries (`pinyin-pro`, `pinyin`) use different dictionaries and rules, so they would not give the same tokens. `pinyin-pro`'s mismatch rate is measured once in step 4 as a comparison.
- The port covers only what `convert_char_to_pinyin` reaches:
  - `lazy_pinyin` with `style=TONE3`, `tone_sandhi=True`, `errors='default'`, `strict=True`.
  - `simpleseg`, `mmseg`, phrase and character lookup, and TONE3 formatting.
  - `contrib/tone_sandhi.py` ported line for line, not re-derived from the rules.
- Dictionaries: converted from pypinyin's installed `phrases_dict.json` and `pinyin_dict.json` by `tools/webgpu/export_pinyin_dict.py` into a compact form, gzip-size measured. pypinyin is MIT; its notice is kept with the converted data.
- The converted data is generated rather than committed if it is over a few MB. Plan 05's weight-hosting repo can carry it with the model files.

### D4. Test corpus (decided 2026-11-18: Seed-TTS eval lists)

Needed: enough real text to make "zero mismatches" mean something. Proposal:
- **Committed:** a hand-written edge-case file, `testdata/webgpu/text/edge_cases.txt`, about 200 lines. It covers:
  - Polyphones in context, and 一/不 sandhi, including across mmseg word boundaries (`一次`).
  - Third-tone chains, including those that stay 3-3-x (`两点论`, `九尾狐`, `五斗柜`) and words mmseg splits (`版本`).
  - Mixed Chinese and Latin, digits.
  - English that jieba splits or keeps: hyphenated words, `well-known`, `v1.2.3`, `3.5%`, `e.g.`, `U.S.A.`, `foo_bar`, `a-b-c-`, `C++`, `C#`, `AT&T`, `don't`, and `\r\n` line breaks.
  - Full-width and half-width punctuation, curly quotes, `;`.
  - The range gaps: `〇`, `ㄅ`, `㈠`, `䷀`, `豈㐀`.
  - Emoji and characters outside the BMP. JS strings are UTF-16, so code-point iteration matters.
  - Whitespace that Python and JS classify differently: U+001C to U+001F, U+0085 and U+FEFF.
  - Chunk boundaries near `max_chars`.
- **Not committed, downloaded by a script:** a larger corpus. The proposal is the Seed-TTS eval test lists (about 2,000 Chinese and 1,000 English sentences, the texts `src/f5_tts/eval/` already uses), if their licence allows use as test input. Otherwise, a Chinese Wikipedia sample.
- **Decided by the user (2026-11-18):** the Seed-TTS eval lists. Step 1 checks their licence first. If it does not allow use as test input, come back to the user before picking another corpus.

## Steps

### Step 1. Python reference dump

- `tools/webgpu/text_reference.py`: for each input line, records:
  - The `rjieba.cut` segments.
  - `lazy_pinyin` per segment, with the `len(seg_) == len(seg)` assertion in the all-3-byte branch.
  - The `convert_char_to_pinyin` tokens.
  - The ids.
  - For the string functions: the fix-up, the trailing-space rule, and `chunk_text` at a few `max_chars` values.
  - The rjieba, jieba-rs and pypinyin versions it ran with.
- For the golden cases, the dump also takes each case's prepared sample count and rate from its manifest, so `max_chars` can be reproduced.
- Output as JSON next to the corpus. The edge-case output is committed; the large corpus's is not.
- Where the golden fields are: `max_chars`, `chunks` and `ref_text_spaced` are in the `text` stage; `joined_text` and `tokens` are in `manifest.chunks[i]`.
- Done when: the dump runs on the edge cases and on the golden cases, and reproduces their `max_chars`, `chunks`, `ref_text_spaced`, `joined_text` and `tokens` exactly.
- Result, 2026-11-21:
  - `tools/webgpu/text_reference.py` (in `.venv313`) runs upstream's own `convert_char_to_pinyin` and `chunk_text`. It wraps the `rjieba.cut` and `lazy_pinyin` they call, to record each call.
    - The reference fix-up, the trailing-space rule and the `max_chars` line sit inside larger functions, so they are copied and asserted against upstream's source, as `make_golden.py` does.
    - The `len(seg_) == len(seg)` assertion never fired.
  - Golden cases: all 7 reproduce exactly: `prep.ref_text`, `max_chars`, `chunks`, `ref_text_spaced`, every chunk's `joined_text` and `tokens`, and the ids in `c*.dit.text_ids`.
  - Edge cases: `testdata/webgpu/text/edge_cases.txt`, 219 inputs in the categories D4 lists. The dump, `edge_cases.json` (241 KB), is committed and deterministic. `\u{XXXX}` in the text file stands for a code point, so invisible characters and trailing spaces survive editors.
  - Versions recorded: rjieba 0.2.2, jieba-rs 0.11.0 (read from rjieba's SBOM), pypinyin 0.55.0.
  - The dump confirms the plan's predictions:
    - `hello-world` and `U.S.A` stay single segments; `don't` splits into `don`, `'`, `t`; `C++` and `C#` stay whole.
    - `〇` gives `ling2` with no space; `ㄅ` gets a space and then itself.
    - `版本` gives `ban3 ben3`; `一次` gives `yi1 ci4`; `两点论` stays `liang3 dian3 lun4`.
    - U+F900 passes through. Its look-alike U+8C48 is an ordinary ideograph (`qi3`). Both are in the edge cases, and jieba segments them differently next to `㐀`.
    - 66 tokens across the edge cases are not in the vocab and map to 0. Examples: the neutral-tone `ge` and `bu`, tab, CR and `}`.
  - The Seed-TTS corpus: the `BytedanceSpeech/seed-tts-eval` repo has no licence file, and its README states no terms for the test set (a Google Drive download). Per D4, this went back to the user. Decided 2026-11-21: use the Seed-TTS lists anyway, locally and uncommitted, as upstream's own eval code does. The texts go in `testdata/webgpu/text/corpus/`, which git ignores.
  - The corpus: `tools/webgpu/seedtts_corpus.py <seedtts_testset.tar>` reads `en/meta.lst`, `zh/meta.lst` and `zh/hardcase.lst` from the tar without extracting it. It writes the unique prompt and target texts: 1,711 English (`seedtts_en.txt`) and 2,857 Chinese, hard cases included (`seedtts_zh.txt`). `text_reference.py --no-golden --input ...` dumps them (4 MB and 9.6 MB of JSON). The `len(seg_) == len(seg)` assertion did not fire on the corpus either.

### Step 2. String functions

- `web/src/text/strings.ts`:
  - `fixRefText` and `addTrailingSpace`.
  - `maxChars(fixedRefText, refSamples, sampleRate, speed)` and `chunkText(text, maxChars)`.
  - `utf8Length`.
- JS details to get right:
  - Iterate by code point (`for...of`), not by UTF-16 unit.
  - Whitespace with Python's set, as above.
- Tests: the golden cases' `text` stage (7 cases) and the edge-case dump.
- Done when: every case matches exactly.
- Result, 2026-11-21:
  - `web/src/text/strings.ts`: `fixRefText`, `addTrailingSpace`, `maxChars`, `chunkText`, `utf8Length`, and `pyStrip` (Python's `strip()`).
  - Python's whitespace is 29 code points, the same in Python 3.13 and 3.15. `str.isspace`, `re`'s `\s` and `strip()` all agree on it. The port spells them out in one character class.
  - Tests (`web/test/text/strings.test.ts`, 228): all 7 golden cases reproduce `prep.ref_text`, `max_chars`, `chunks`, `ref_text_spaced` and every `joined_text`. Every edge-case line matches Python's UTF-8 length, fix-up, trailing-space rule, and `chunk_text` at 10, 25, 50 and 135 bytes.
  - JS's whitespace would fail them. `trim()` fails the leading-BOM line. `\s` in the split fails the line with U+0085 after a full stop.
  - JS `split` and Python `re.split` gave the same pieces on every line, including the zero-width splits after CJK punctuation.
  - The Seed-TTS corpus (local only, skipped where absent) matches too: all 4,568 texts, every rule, all four chunk sizes.

### Step 3. Segmentation (D1, D2)

- `tools/webgpu/jieba_wasm/`: the build script for jieba-rs 0.11.0, and the built package.
- `web/src/text/segment.ts` wraps it, loaded lazily.
- `web/src/text/segment_ascii.ts`: D2's pure-TS path for text with no CJK characters.
- Compare both with rjieba's segments on the edge cases and the large corpus (Node, Vitest). Also run `jieba-wasm` 2.4.0 once as a comparison. Report mismatch counts with examples.
- Done when: our wasm build has zero mismatches (or the TS fallback is done and has zero), and D2 is decided.
- Result, 2026-11-22: D1 and D2 both hold. Zero mismatches.
  - Built: `tools/webgpu/jieba_wasm/` holds the crate, `build.py`, and `build.json`, which records rustc, jieba-rs and the hashes. The built `web/src/text/jieba.wasm` sits next to its wrapper, so Vite can resolve it with `new URL(..., import.meta.url)`.
  - Two changes from the proposal:
    - A plain C ABI (`alloc`, `cut`, `result_ptr`, `dealloc`) instead of `wasm-bindgen`. There is no JS glue, and no `wasm-bindgen` CLI version to keep in step with the crate.
    - `default-dict` is off. It unpacks the dictionary with zstd, whose C library needs a C compiler that targets wasm (none here). The same `dict.txt` is embedded uncompressed and loaded with `Jieba::with_dict`. That builds the same state as `Jieba::new()`: the file has no duplicate words (`with_dict` would overwrite a duplicate's frequency, and `new()` would not), no non-ASCII whitespace, and an integer frequency on every line.
  - Pinning: `Cargo.lock` matches rjieba's SBOM for every shared crate (jieba-rs and jieba-macros 0.11.0, bytecount 0.6.9, rustc-hash 2.1.3). 0.11.0 no longer uses `regex`.
  - Reproducible: a clean rebuild gives the same bytes. Without `--remap-path-prefix`, the panic locations embedded the build machine's cargo path, user name included. `build.py` remaps them and fails if a local path is left.
  - Our wasm against rjieba: 0 of 219 edge cases, 0 of 1,711 English corpus texts and 0 of 2,857 Chinese corpus texts differ.
    - The wrapper first stripped a leading U+FEFF from tokens. `TextDecoder` does that unless `ignoreBOM` is set, and the two BOM edge cases caught it.
  - `jieba-wasm` 2.4.0 against rjieba, once: 18 of 219 edge cases and 14 of 1,711 English texts differ, and 0 of 2,857 Chinese. Every difference is a `.`, `_` or `-` join: rjieba keeps `x-ray`, `v1.2.3`, `Wi-Fi` and `138-0013-8000` whole, and 2.4.0 splits them. Through the space rule, `x-ray` becomes `x- ray`, which changes the tokens.
  - D2, decided: text with no CJK ideograph (jieba-rs's `is_cjk`) uses `web/src/text/segment_ascii.ts` and never loads the wasm.
    - jieba-rs reduces to this for such text. Only five dictionary words are pure ASCII (`AT&T`, `C#`, `C++`, `c#`, `c++`), none can overlap another, and each beats single characters on the route. The rest of each block goes through the HMM's skip splitter, which gives `skip_match_len` runs and one token per gap. Characters outside blocks are one token each, except `\r\n`.
    - It matches rjieba on all 93 edge cases and all 1,711 English texts that have no CJK. It also matches the wasm on 20,000 random strings built from the characters that matter.
    - Mutation checks: dropping `%`, the `-` connector, three of the five dictionary words, the `\r\n` grouping, or one-token gaps each fails tests.
  - Size and init: 5.6 MB raw, 2.1 MB gzip -9, 1.7 MB brotli 11. In Node, the dictionary takes about 80 ms to build and the instance uses 46 MiB. Browser numbers come in step 6.

### Step 4. Pinyin (D3)

- Read pypinyin 0.55.0's `core.py`, `converter.py`, `seg/simpleseg.py`, `seg/mmseg.py`, `contrib/tone_sandhi.py` and `style/tone.py`. Write the exact algorithm into `docs/research/webgpu-text-frontend.md` (add it to `docs/index.md`).
- `tools/webgpu/export_pinyin_dict.py` and `web/src/text/pinyin.ts`.
- Compare per segment with the dump's `lazy_pinyin` output. Also run `pinyin-pro` once as a comparison, and record its mismatch rate.
- Done when: zero mismatches on the edge cases and the corpus, or the remaining ones are listed and judged.
- Result, 2026-11-22: zero mismatches. D3 holds.
  - The algorithm is written out in `docs/research/webgpu-text-frontend.md`: `simpleseg`, `mmseg` with `no_non_phrases`, the converter, the TONE3 and TONE2 conversions, and the sandhi quirks. For example, `"yi1".replace('4', '2')` leaves `一` before a fourth tone as `yi1`.
  - `web/src/text/pinyin.ts` mirrors pypinyin function by function. `Pinyin.load()` fetches the dictionary lazily, and `Pinyin.parse()` takes its text.
  - Data: `tools/webgpu/export_pinyin_dict.py` writes `web/src/text/pinyin_dict.txt`, committed: 41,923 characters and 47,111 phrases, first readings only, since `heteronym=False` reads nothing else. pypinyin's MIT licence is copied next to it.
    - Size: 1.6 MB raw, 662 KB gzip -9, 541 KB brotli 11.
    - In Node, parsing takes 55 to 75 ms, and 1,000 calls on a 20-character sentence take about 50 ms.
    - `.gitattributes` marks the file `-text`, so a Windows checkout keeps it byte-identical. The parser accepts CRLF as well.
  - Tests (`web/test/text/pinyin.test.ts`), all equal to pypinyin:
    - every `lazy_pinyin` call in the edge cases and in both corpus dumps;
    - `pinyin_cases.json` (`tools/webgpu/pinyin_reference.py`, local only, 119,034 cases): every dictionary character, every phrase, and 30,000 random mixes.
  - Every word `mmseg` yields is a phrase or a single character, and all of those are in the cases. So the conversion and the sandhi are proven on every input they can receive, and the random mixes and corpus cover the segmenting.
  - Mutation checks:
    - Fixing the `yi1` quirk, dropping the third-tone rule, or dropping `no_non_phrases` each fails tests.
    - Three mutants survive, and each is unreachable. The third-tone quirk before the final run occurs in no phrase. The `iou` mark rule is reached by no reading. `ü` handling inside TONE2 is undone by `tone2_to_tone`.
  - `pinyin-pro` 3.29.5, once: on all-Han calls it differs from pypinyin on 4.36% of characters in the Chinese corpus (3,016 of 69,204) and 7.11% in the edge cases. Examples: `了` alone is `liao3`, `长` is `chang2`, and `可以` stays `ke3 yi3`.

### Step 5. Tokens and ids

- `web/src/text/tokens.ts`: `convertCharToPinyin`, the vocab loader, and `textToIds` (int32).
  - The loader splits on `\r?\n` and drops the final empty line.
  - A test asserts `vocab[" "] === 0`, 2545 entries, and that the last entry is `𠮶`.
  - Unknown tokens map to 0.
- Tests:
  - Exact ids for every golden case: their `c*.dit.text_ids`, (1, n) int32 with no padding, through the parity runner's `exact` check.
  - Exact tokens for the edge cases and the corpus.
- Done when: all match.
- Result, 2026-11-22: all match.
  - `web/src/text/tokens.ts`:
    - `convertCharToPinyin(text, cut, pinyin?)` ports the three branches. It throws where Python would raise: a 3-byte segment with too few pinyin items, or Han text with no dictionary.
    - `parseVocab` follows Python's text mode: `\r\n` and `\r` become `\n`, each key is `line[:-1]`, and a later duplicate overwrites. So a CRLF checkout reads the same as LF. A test checks CRLF, LF, lone CR and an unterminated last line.
    - `textToIds` gives an int32 array, with unknown tokens as 0.
    - `TextFrontend` loads the jieba wasm only for text with a CJK ideograph, and the pinyin dictionary only for text with a `RE_HANS` character. Note that `〇` needs the dictionary but not jieba. Each loads once.
  - Tests (`web/test/text/tokens.test.ts`):
    - Vocab: 2545 entries, `" "` at 0, `𠮶` last at 2544.
    - All 7 golden cases: every chunk's tokens, and its ids through the parity runner's `exact` check against `c*.dit.text_ids`.
    - Tokens and ids for every edge case and corpus line. `TextFrontend` gives the same tokens on all of them.
    - A test counts loads, to check that English loads nothing, `〇` loads only pinyin, and Chinese loads each once.
  - Mutation checks: dropping the segment-length test in the space rule, starting `is_chinese` at U+4E00, or dropping `"` from the no-space set each fails tests.
  - The vocab is per checkpoint, so its URL comes with the model files (plan 05). The tests read upstream's `src/f5_tts/infer/examples/vocab.txt`, which `load_model` uses when no vocab is given.

### Step 6. Size and loading

- Measure, in Chrome:
  - Compressed transfer size and init time of the jieba wasm, and of the pinyin data. The size that counts is the web build's, not the npm package's (which bundles node, web and deno builds).
  - Time per typical sentence.
- If D2's pure-TS path is exact, text with no CJK characters must not fetch the Chinese assets; show it. Decide the loading strategy with plan 06 (a dynamic import on the first CJK character).
- Done when: the numbers are in the research doc, and the loading behaviour matches D2's outcome.

### Step 7. Close out

- Hand off to plan 04: the API, and the byte-length helper for the duration rule.
- Hand off to plan 06: the asset sizes and the loading rule.
- `docs/debugging.md` entries for anything surprising, at least the CRLF `vocab.txt` and the jieba-rs version gap.
- Update the task list.

## Risks

- **R1. jieba-rs version drift.** If upstream moves to a newer rjieba, our wasm build must follow. The dump records the versions, and a small Python-side CI check regenerates the edge-case dump and compares.
- **R2. pypinyin's internals are larger than they look:** phrase priority in mmseg and the sandhi rules. Porting only the path used, line for line, keeps it manageable, but the corpus is what proves it.
- **R3. Unicode:** UTF-16 surrogates, whitespace classes, and byte length versus code-point length. Each feeds a byte-length comparison in `chunk_text` or in the token branching.
- **R4. Dictionary size.** About 3.3 MB of pinyin JSON plus the jieba dictionary, before compression. Acceptable next to about 1.3 GB of weights, but it needs lazy loading. If D2 fails, the jieba part loads for English too.
- **R5. A Rust toolchain for the wasm build.** Only for rebuilding; the built package is kept. If `wasm-bindgen` output does not run in all target browsers, the TS fallback applies.

## Pressure test, 2026-11-18

A fresh agent given only this file reviewed the first draft against the code, the installed packages, the npm package and the golden data. Findings and what changed:
- **`jieba-wasm` embeds jieba-rs 0.7.0 (a fork); rjieba embeds 0.11.0.** The dictionaries are identical but the segmentation code differs, and the difference changes tokens for words like `hello-world`. Confirmed with rjieba's SBOM and its segments. D1 is now our own wasm build of 0.11.0.
- **English text depends on jieba too,** through the space rule. So "English loads none of it" was wrong. D2 now proposes a pure-TS path for text with no CJK, proven against rjieba, with loading for all text as the fallback. English edge cases are added.
- **Tone sandhi was described too loosely.** It applies only within mmseg's words, and the third-tone rule only to the run that ends a word. The golden data shows both. The sandhi module is now ported line for line, with these words as edge cases.
- **`vocab.txt` is CRLF in this checkout.** A naive JS loader would map every token to 0. The loader and a test now handle it.
- **The whitespace list was incomplete** (U+0085), and JS's `\s` and `trim()` include U+FEFF. The port uses an explicit class.
- **`is_chinese` and pypinyin's Han range differ.** The cases where they differ are documented and added.
- **`max_chars` inputs and arithmetic are pinned:** the fixed-up text before the trailing-space rule, and the same operation order. The dump takes sample counts from the manifests.
- **Smaller fixes:**
  - Where the golden text fields live.
  - The TS fallback ports jieba-rs, not Python jieba, with care for `Math.log` and ties.
  - The size that counts is the web build's.
  - The space-rule check in the ASCII branch is a substring test.
