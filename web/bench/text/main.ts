// Plan 03 step 6: the text front end's loading and speed in the browser, from a production build (run.mjs).
// It checks that text with no CJK fetches neither Chinese asset, then times the first Chinese sentence (the
// jieba wasm and the pinyin dictionary, each split into fetch and the rest) and the steady state.
import { Pinyin } from "../../src/text/pinyin";
import { Segmenter } from "../../src/text/segment";
import { TextFrontend } from "../../src/text/tokens";

const SENTENCES = {
  en: "The quick brown fox jumps over the lazy dog, and then it runs away into the forest.",
  zh: "我们今天去公园散步，看见了一只不怕人的小松鼠，它在树上跳来跳去，非常可爱。",
  mixed: "我昨天在 GitHub 上看到一个 well-known 的项目，它的 v1.2.3 版本非常好用。",
};

function assets() {
  return (performance.getEntriesByType("resource") as PerformanceResourceTiming[])
    .filter((e) => /jieba|pinyin_dict/.test(e.name))
    .map((e) => ({
      file: e.name.split("/").pop(),
      transferSize: e.transferSize,
      encodedBodySize: e.encodedBodySize,
      decodedBodySize: e.decodedBodySize,
      fetchMs: Math.round(e.responseEnd - e.startTime),
    }));
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return Math.round((s[Math.floor(s.length / 2)] as number) * 1000) / 1000;
}

/** Longest gap between animation-frame-free tasks is not observable directly; long tasks are. */
function watchLongTasks(): () => number[] {
  const tasks: number[] = [];
  const po = new PerformanceObserver((list) => list.getEntries().forEach((e) => tasks.push(Math.round(e.duration))));
  try {
    po.observe({ type: "longtask", buffered: false });
  } catch {
    // longtask is Chromium-only
  }
  return () => (po.disconnect(), tasks);
}

declare global {
  interface Window {
    runTextBench: (runs?: number) => Promise<unknown>;
  }
}

window.runTextBench = async (runs = 50) => {
  const timing: Record<string, number> = {};
  const loads = { segmenter: 0, pinyin: 0 };
  const fe = new TextFrontend({
    segmenter: async () => {
      loads.segmenter++;
      const t = performance.now();
      const s = await Segmenter.load();
      timing.segmenterLoadMs = Math.round(performance.now() - t);
      return s;
    },
    pinyin: async () => {
      loads.pinyin++;
      const t = performance.now();
      const p = await Pinyin.load();
      timing.pinyinLoadMs = Math.round(performance.now() - t);
      return p;
    },
  });

  // 1. text with no CJK: nothing Chinese is fetched
  let t = performance.now();
  const enTokens = await fe.tokens(SENTENCES.en);
  timing.firstEnglishMs = Math.round((performance.now() - t) * 10) / 10;
  const afterEnglish = { loads: { ...loads }, assets: assets() };

  // 2. the first Chinese sentence loads both, once
  const stopWatch = watchLongTasks();
  t = performance.now();
  const zhTokens = await fe.tokens(SENTENCES.zh);
  timing.firstChineseMs = Math.round(performance.now() - t);
  await new Promise((r) => setTimeout(r, 100));
  const longTasksMs = stopWatch();
  await fe.tokens(SENTENCES.mixed);
  const afterChinese = { loads: { ...loads }, assets: assets() };

  // 3. the split of each load: compile and init of the wasm, parse of the dictionary (from the HTTP cache)
  const urlOf = (part: string) => performance.getEntriesByType("resource").find((e) => e.name.includes(part))?.name;
  const wasmUrl = urlOf("jieba");
  const dictUrl = urlOf("pinyin_dict");
  if (!wasmUrl || !dictUrl) throw new Error("the Chinese assets were not fetched");
  const wasmBytes = await (await fetch(wasmUrl)).arrayBuffer();
  t = performance.now();
  const module = await WebAssembly.compile(wasmBytes);
  timing.wasmCompileMs = Math.round(performance.now() - t);
  t = performance.now();
  const instance = await WebAssembly.instantiate(module);
  (instance.exports.init as () => void)();
  timing.wasmInstantiateAndInitMs = Math.round(performance.now() - t);
  const dictText = await (await fetch(dictUrl)).text();
  t = performance.now();
  Pinyin.parse(dictText);
  timing.dictParseMs = Math.round(performance.now() - t);

  // 4. steady state per sentence
  const perSentenceMs: Record<string, number> = {};
  for (const [name, text] of Object.entries(SENTENCES)) {
    const xs: number[] = [];
    for (let i = 0; i < runs; i++) {
      const t0 = performance.now();
      await fe.tokens(text);
      xs.push(performance.now() - t0);
    }
    perSentenceMs[name] = median(xs);
  }

  return {
    userAgent: navigator.userAgent,
    afterEnglish,
    afterChinese,
    timing,
    longTasksMs,
    perSentenceMs,
    tokens: { en: enTokens.length, zh: zhTokens.length },
  };
};

document.body.dataset.ready = "1";
