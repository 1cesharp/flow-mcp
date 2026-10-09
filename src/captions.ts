import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { createWriteStream, existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { cpus, homedir, tmpdir } from "node:os";
import { basename, extname, join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { ReadableStream } from "node:stream/web";
import { promisify } from "node:util";
import { chromium, type Browser, type Page } from "playwright-core";

import { chromeExecutable, HOME_DIR } from "./chrome.js";
import { FFMPEG_HINT, IS_MAC, IS_WIN } from "./platform.js";

// Captions in the style short-form apps use, burned into the picture. The words and their timings come from
// whisper.cpp on this computer; the lettering is drawn by headless Chrome and laid over the video by ffmpeg. Chrome
// does the drawing because Homebrew's ffmpeg is built without any text filter (no ass, subtitles or drawtext), and
// Chrome is already a requirement - so the same code runs on a Mac and on Windows with nothing extra to install.

const run = promisify(execFile);
const FFMPEG = process.env.FLOW_MCP_FFMPEG ?? "ffmpeg";
const FFPROBE = process.env.FLOW_MCP_FFPROBE ?? "ffprobe";
const WHISPER = process.env.FLOW_MCP_WHISPER ?? "whisper-cli";
// Transcripts and preview stills; the panel is allowed to show pictures from here.
export const CACHE_DIR = join(HOME_DIR, "captions");
const MODEL_DIR = join(HOME_DIR, "models");
const BIG = { maxBuffer: 64 * 1024 * 1024 };
type RunError = NodeJS.ErrnoException & { stderr?: string };

export interface Word {
  text: string;
  start: number;
  end: number;
  // A line break typed before this word in the panel: start a new caption here.
  br?: boolean;
}

export interface Transcript {
  words: Word[];
  text: string;
  language: string;
  model: string;
}

export const CAPTION_STYLES = {
  bold: { label: "Bold pop", blurb: "Big capitals, a few words at a time; the word being said lights up", maxWords: 3, highlight: true, pop: true, color: "#FFE135", letters: "capitals" },
  highlight: { label: "Highlight", blurb: "The word being said sits on a coloured block", maxWords: 4, highlight: true, pop: false, color: "#7C3AED", letters: "as_said" },
  reveal: { label: "Word by word", blurb: "Each word appears the moment it is said", maxWords: 5, highlight: true, pop: false, color: "#FFFFFF", letters: "as_said" },
  clean: { label: "Clean", blurb: "Classic subtitles: easy to read, nothing moves", maxWords: 8, highlight: false, pop: false, color: "#FFFFFF", letters: "as_said" },
  boxed: { label: "Boxed", blurb: "White words on a dark band, readable over anything", maxWords: 7, highlight: false, pop: false, color: "#FFFFFF", letters: "as_said" },
} as const;
export type CaptionStyle = keyof typeof CAPTION_STYLES;
// A preset, or how far down the frame the middle of the caption sits: 0 = top edge, 100 = bottom edge.
export type CaptionPosition = "bottom" | "middle" | "top" | number;
export type CaptionSize = "small" | "medium" | "large";
// as_said keeps whisper's own capitals (sentence starts, names); capitals and lowercase are looks, and both drop
// trailing commas and full stops the way short-form captions do.
export type CaptionLetters = "as_said" | "capitals" | "lowercase";

export interface CaptionOptions {
  style: CaptionStyle;
  position: CaptionPosition;
  size: CaptionSize;
  color?: string;
  letters?: CaptionLetters;
  text?: string;
  language?: string;
}

const lettersOf = (o: { style: CaptionStyle; letters?: CaptionLetters }): CaptionLetters => o.letters ?? CAPTION_STYLES[o.style].letters;

// Letter size as a share of the frame's short side, so a 9:16 phone video and a 16:9 landscape one read the same.
const FONT = { bold: 7.2, highlight: 6.2, reveal: 6.4, clean: 4.6, boxed: 4.6 };
const CHAR_EM = { bold: 0.74, highlight: 0.56, reveal: 0.56, clean: 0.52, boxed: 0.52 };
const SIZE = { small: 0.82, medium: 1, large: 1.2 };

/* ---------- hearing the words ---------- */

const MODEL_PREFERENCE = [
  // English-only models time words best. large-v3-turbo is the most accurate listener but placed "picks" 0.6 s
  // early in a test where small.en was 30 ms out, so it comes last.
  "ggml-small.en.bin",
  "ggml-base.en.bin",
  "ggml-medium.en.bin",
  "ggml-small.bin",
  "ggml-base.bin",
  "ggml-medium.bin",
  "ggml-large-v3-turbo.bin",
  "ggml-large-v3.bin",
];

function findModel(language: string): string {
  if (process.env.FLOW_MCP_WHISPER_MODEL) return process.env.FLOW_MCP_WHISPER_MODEL;
  const dirs = [MODEL_DIR, join(homedir(), ".cache", "whisper-models"), join(homedir(), ".cache", "whisper.cpp")];
  const found = MODEL_PREFERENCE.flatMap((name) => dirs.map((d) => join(d, name))).filter((p) => existsSync(p));
  // An English-only model cannot hear another language, so prefer a multilingual one when one is asked for.
  const wantOther = language !== "auto" && language !== "en";
  const pick = (wantOther ? found.find((p) => !p.includes(".en.")) : undefined) ?? found[0];
  if (!pick)
    throw new Error(
      `Captions need a speech model to hear the words. Download ggml-small.en.bin (466 MB) or ggml-base.en.bin (148 MB) from huggingface.co/ggerganov/whisper.cpp and put it in ${MODEL_DIR}.`,
    );
  return pick;
}

const WHISPER_HINT = IS_MAC
  ? "brew install whisper-cpp"
  : IS_WIN
    ? "download whisper-cli from github.com/ggml-org/whisper.cpp/releases and put it on your PATH (or set FLOW_MCP_WHISPER)"
    : "install whisper.cpp so whisper-cli is on your PATH";

// What the panel needs to know before offering captions, and the fix for whatever is missing.
export async function captionSetup(): Promise<{ ready: boolean; whisper: boolean; whisper_hint?: string; model?: string; models_dir: string }> {
  const whisper = await run(WHISPER, ["--help"]).then(
    () => true,
    (err: RunError) => err.code !== "ENOENT",
  );
  let model: string | undefined;
  try {
    model = basename(findModel("auto"));
  } catch {
    model = undefined;
  }
  return { ready: whisper && Boolean(model), whisper, whisper_hint: whisper ? undefined : WHISPER_HINT, model, models_dir: MODEL_DIR };
}

// The one-click "Get the speech model" in the panel: base.en from whisper.cpp's own model page, 148 MB.
let fetching: Promise<{ model: string; size: number }> | null = null;
export function downloadSpeechModel(): Promise<{ model: string; size: number }> {
  fetching ??= (async () => {
    const name = "ggml-base.en.bin";
    const dest = join(MODEL_DIR, name);
    if (existsSync(dest)) return { model: name, size: statSync(dest).size };
    mkdirSync(MODEL_DIR, { recursive: true });
    const res = await fetch(`https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${name}`);
    if (!res.ok || !res.body) throw new Error(`The speech model could not be downloaded (${res.status}). Try again in a moment.`);
    const part = `${dest}.part`;
    await pipeline(Readable.fromWeb(res.body as unknown as ReadableStream), createWriteStream(part));
    const size = statSync(part).size;
    if (size < 100_000_000) {
      rmSync(part, { force: true });
      throw new Error("The speech model arrived incomplete. Try again.");
    }
    renameSync(part, dest);
    return { model: name, size };
  })().finally(() => {
    fetching = null;
  });
  return fetching;
}

async function probe(file: string): Promise<{ width: number; height: number; duration: number; audio?: string }> {
  if (!existsSync(file)) throw new Error(`No video at ${file}.`);
  const { stdout } = await run(
    FFPROBE,
    ["-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height:stream_side_data=rotation:stream_tags=rotate:format=duration", "-of", "json", file],
    BIG,
  ).catch((err: NodeJS.ErrnoException) => {
    if (err.code === "ENOENT") throw new Error(`ffmpeg is not installed. Install it with: ${FFMPEG_HINT}`);
    throw new Error(`"${basename(file)}" isn't a playable video.`);
  });
  const data = JSON.parse(stdout) as {
    streams: { codec_type: string; codec_name?: string; width?: number; height?: number; side_data_list?: { rotation?: number }[]; tags?: { rotate?: string } }[];
    format: { duration?: string };
  };
  const v = data.streams.find((s) => s.codec_type === "video");
  if (!v?.width || !v.height) throw new Error(`"${basename(file)}" has no picture to put captions on.`);
  // Phone videos are often stored sideways with a rotation flag; ffmpeg turns them upright, so measure them upright.
  const rotation = Math.abs(Number(v.side_data_list?.find((d) => d.rotation !== undefined)?.rotation ?? v.tags?.rotate ?? 0)) % 180;
  const [width, height] = rotation === 90 ? [v.height, v.width] : [v.width, v.height];
  return { width, height, duration: Number(data.format.duration ?? 0), audio: data.streams.find((s) => s.codec_type === "audio")?.codec_name };
}

// whisper.cpp's word-sized segments (-ml 1) run each word into the next, hiding every pause. Its tokens keep the real
// gaps, so words are rebuilt from those: a token that starts with a space starts a new word, anything else (", 't")
// belongs to the word before it.
function wordsFromTokens(json: { transcription: { tokens?: { text: string; offsets: { from: number; to: number } }[] }[] }): Word[] {
  const words: Word[] = [];
  for (const seg of json.transcription) {
    for (const t of seg.tokens ?? []) {
      if (/^\s*\[_/.test(t.text)) continue;
      const clean = t.text.trim();
      if (!clean) continue;
      const last = words.at(-1);
      if (last && (!/^\s/.test(t.text) || /^[.,!?;:%)\]}"'»…-]/.test(clean))) {
        last.text += clean;
        last.end = t.offsets.to / 1000;
      } else words.push({ text: clean, start: t.offsets.from / 1000, end: t.offsets.to / 1000 });
    }
  }
  // Drop what whisper writes for non-speech: [MUSIC], (laughs), ♪.
  return words.filter((w) => !/^[[(♪*].*[\])♪*]$|^♪+$/.test(w.text));
}

export async function transcribe(video: string, language = "auto"): Promise<Transcript> {
  const info = await probe(video);
  if (!info.audio) throw new Error(`"${basename(video)}" has no sound, so there are no words to caption.`);
  const model = findModel(language);
  const lang = basename(model).includes(".en.") ? "en" : language;
  const st = statSync(video);
  const key = createHash("sha1").update(`${video}|${st.size}|${st.mtimeMs}|${basename(model)}|${lang}`).digest("hex").slice(0, 16);
  const cached = join(CACHE_DIR, `${key}.json`);
  if (existsSync(cached)) return JSON.parse(readFileSync(cached, "utf8")) as Transcript;

  const work = mkdtempSync(join(tmpdir(), "flow-captions-"));
  try {
    const wav = join(work, "audio.wav");
    await run(FFMPEG, ["-v", "error", "-y", "-i", video, "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav], BIG);
    const threads = String(Math.max(2, Math.min(8, cpus().length - 1)));
    await run(WHISPER, ["-m", model, "-f", wav, "-l", lang, "-t", threads, "-ojf", "-of", join(work, "words"), "-np"], BIG).catch((err: RunError) => {
      if (err.code === "ENOENT") throw new Error(`Captions need whisper.cpp to hear the words. Install it: ${WHISPER_HINT}.`);
      throw new Error(`whisper.cpp could not read the sound: ${String(err.stderr ?? err.message).trim().split("\n").at(-1)}`);
    });
    const json = JSON.parse(readFileSync(join(work, "words.json"), "utf8")) as Parameters<typeof wordsFromTokens>[0] & { result?: { language?: string } };
    const words = wordsFromTokens(json);
    if (!words.length) throw new Error(`No speech was heard in "${basename(video)}".`);
    const out: Transcript = { words, text: words.map((w) => w.text).join(" "), language: json.result?.language ?? lang, model: basename(model) };
    mkdirSync(CACHE_DIR, { recursive: true });
    writeFileSync(cached, JSON.stringify(out));
    return out;
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

/* ---------- fixing the words ---------- */

const norm = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N}']+/gu, "");

// Lines up a corrected transcript with the timed one. Every word that still matches keeps its timing; new or changed
// words are spread evenly over the gap they fill, so fixing "flo" to "Flow" or adding a missed word just works.
export function alignText(timed: Word[], text: string): Word[] {
  const toks: { text: string; br: boolean }[] = [];
  text.split("\n").forEach((line, li) =>
    line
      .split(/\s+/)
      .filter(Boolean)
      .forEach((t, i) => toks.push({ text: t, br: li > 0 && i === 0 })),
  );
  if (!toks.length) return [];
  const a = timed.map((w) => norm(w.text));
  const b = toks.map((t) => norm(t.text));
  const n = a.length;
  const m = b.length;
  const match = new Array<number>(m).fill(-1);
  if (n * m <= 25_000_000) {
    // Longest common subsequence; 25 M cells is roughly an hour of speech on either side.
    const W = m + 1;
    const L = new Uint16Array((n + 1) * W);
    for (let i = n - 1; i >= 0; i--)
      for (let j = m - 1; j >= 0; j--) L[i * W + j] = a[i] && a[i] === b[j] ? L[(i + 1) * W + j + 1] + 1 : Math.max(L[(i + 1) * W + j], L[i * W + j + 1]);
    for (let i = 0, j = 0; i < n && j < m; ) {
      if (a[i] && a[i] === b[j]) match[j++] = i++;
      else if (L[(i + 1) * W + j] >= L[i * W + j + 1]) i++;
      else j++;
    }
  } else {
    // Very long texts: match greedily, looking a little way ahead.
    for (let i = 0, j = 0; j < m; j++) {
      const k = a.slice(i, i + 40).indexOf(b[j]);
      if (b[j] && k >= 0) {
        match[j] = i + k;
        i += k + 1;
      }
    }
  }
  const out: Word[] = toks.map((t, k) => (match[k] >= 0 ? { ...t, start: timed[match[k]].start, end: timed[match[k]].end } : { ...t, start: NaN, end: NaN }));
  for (let k = 0; k < m; ) {
    if (!Number.isNaN(out[k].start)) {
      k++;
      continue;
    }
    let e = k;
    while (e < m && Number.isNaN(out[e].start)) e++;
    const from = k > 0 ? out[k - 1].end : (timed[0]?.start ?? 0);
    const to = e < m ? out[e].start : Math.max(from + 0.35 * (e - k), timed.at(-1)?.end ?? 0);
    const step = Math.max(0.05, (to - from) / (e - k));
    for (let q = k; q < e; q++) {
      out[q].start = from + step * (q - k);
      out[q].end = from + step * (q - k + 1);
    }
    k = e;
  }
  return out;
}

/* ---------- laying out captions ---------- */

interface State {
  words: string[];
  active: number;
  scale: number;
  from: number;
  to: number;
}

function paginate(words: Word[], maxWords: number, maxChars: number): Word[][] {
  const pages: Word[][] = [];
  let cur: Word[] = [];
  let chars = 0;
  for (const w of words) {
    const prev = cur.at(-1);
    const brk =
      prev !== undefined &&
      (w.br ||
        cur.length >= maxWords ||
        chars + w.text.length > maxChars ||
        /[.!?…]["')\]]*$/.test(prev.text) ||
        /[,;:]$/.test(prev.text) ||
        w.start - prev.end > 0.6);
    if (brk) {
      pages.push(cur);
      cur = [];
      chars = 0;
    }
    cur.push(w);
    chars += w.text.length + 1;
  }
  if (cur.length) pages.push(cur);
  return pages;
}

function layout(words: Word[], style: CaptionStyle, width: number, height: number, size: CaptionSize, letters: CaptionLetters): { pages: Word[][]; states: State[] } {
  const s = CAPTION_STYLES[style];
  const u = (Math.min(width, height) / 100) * SIZE[size];
  // Capitals are wider than small letters, so fewer of them fit on a line.
  const wide = CHAR_EM[style] * (letters === "capitals" && style !== "bold" ? 1.2 : letters !== "capitals" && style === "bold" ? 0.85 : 1);
  const perLine = Math.max(8, Math.floor(boxWidth(style, width, height) / (FONT[style] * u * wide)));
  const pages = paginate(words, s.maxWords, perLine * 2);
  // Capitals and lowercase read better without commas and full stops; question and exclamation marks stay.
  const shown = (t: string) => (letters === "as_said" ? t : t.replace(/[.,;:]+$/, ""));
  const states: State[] = [];
  let clock = 0;
  const push = (st: State) => {
    st.from = Math.max(st.from, clock);
    if (st.to - st.from < 0.02) return;
    states.push(st);
    clock = st.to;
  };
  pages.forEach((p, pi) => {
    const start = p[0].start;
    const next = pages[pi + 1]?.[0].start;
    let end = p[p.length - 1].end + 0.25;
    // Close a short gap to the next caption instead of flashing an empty frame; never overlap it.
    if (next !== undefined) end = next - end < 0.5 ? next : Math.min(end, next);
    const texts = p.map((w) => shown(w.text));
    const first = states.length;
    if (!s.highlight) push({ words: texts, active: -1, scale: 1, from: start, to: end });
    else p.forEach((w, i) => push({ words: texts, active: i, scale: 1, from: i === 0 ? start : w.start, to: i < p.length - 1 ? p[i + 1].start : end }));
    // A quick grow-in for each new caption: three frames, then still.
    const lead = states[first];
    if (s.pop && lead && lead.to - lead.from > 0.2) {
      const t = lead.from;
      states.splice(
        first,
        1,
        { ...lead, scale: 0.78, from: t, to: t + 0.05 },
        { ...lead, scale: 0.94, from: t + 0.05, to: t + 0.1 },
        { ...lead, scale: 1.05, from: t + 0.1, to: t + 0.15 },
        { ...lead, from: t + 0.15 },
      );
    }
  });
  return { pages, states };
}

const boxWidth = (style: CaptionStyle, width: number, height: number) =>
  width * (height > width ? 0.86 : style === "clean" || style === "boxed" ? 0.8 : 0.72);

function textOn(hex: string): string {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.6 ? "#111111" : "#FFFFFF";
}

function pageHtml(style: CaptionStyle, o: { width: number; height: number; position: CaptionPosition; size: CaptionSize; color: string; letters: CaptionLetters; background?: string }): string {
  const { width: W, height: H } = o;
  const vertical = H > W;
  const font = FONT[style] * (Math.min(W, H) / 100) * SIZE[o.size];
  const place =
    typeof o.position === "number"
      ? `top:${Math.min(96, Math.max(4, o.position))}%;transform:translate(-50%,-50%)`
      : o.position === "middle"
      ? "top:50%;transform:translate(-50%,-50%)"
      : o.position === "top"
        ? `top:${vertical ? 13 : 7}%;transform:translateX(-50%)`
        : `bottom:${vertical ? 21 : 8}%;transform:translateX(-50%)`;
  const sans = `system-ui,-apple-system,"Segoe UI","Helvetica Neue",Arial,sans-serif`;
  // The outline is a copy of each word drawn underneath with a thick stroke, so it never eats into the letters.
  const outline = (w: string) => `.w{position:relative;display:inline-block}.w::before{content:attr(data-t);position:absolute;left:0;top:0;z-index:-1;-webkit-text-stroke:${w} #000;color:#000}`;
  const css = {
    bold: `.cap{font:900 ${font}px/1.12 "Arial Black","Arial Bold",Impact,${sans};text-transform:uppercase;letter-spacing:.01em;color:#fff}
      ${outline(".17em")}.w::before{text-shadow:0 .07em 0 rgba(0,0,0,.55)}.w{margin:0 .07em}.w.on{color:var(--hl);transform:scale(1.06)}`,
    highlight: `.cap{font:800 ${font}px/1.32 ${sans};color:#fff}
      .w{display:inline-block;padding:.02em .13em .07em;margin:0 -.04em;border-radius:.2em;text-shadow:0 .05em .2em rgba(0,0,0,.75)}.w.on{background:var(--hl);color:var(--on);text-shadow:none}`,
    reveal: `.cap{font:800 ${font}px/1.2 ${sans};color:#fff}${outline(".13em")}.later{visibility:hidden}.w.on{color:var(--hl)}`,
    clean: `.cap{font:700 ${font}px/1.25 ${sans};color:#fff;text-shadow:0 .04em .12em rgba(0,0,0,.9),0 0 .35em rgba(0,0,0,.55)}`,
    boxed: `.cap{font:700 ${font}px/1.55 ${sans};color:#fff}.inner{display:inline;background:rgba(0,0,0,.74);padding:.12em .45em;border-radius:.28em;-webkit-box-decoration-break:clone;box-decoration-break:clone}`,
  }[style];
  return `<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;width:${W}px;height:${H}px;overflow:hidden;background:${o.background ? `#000 url(${o.background}) center/100% 100% no-repeat` : "transparent"}}
.cap{position:absolute;left:50%;width:${boxWidth(style, W, H)}px;text-align:center;text-wrap:balance;--hl:${o.color};--on:${textOn(o.color)};${place};-webkit-font-smoothing:antialiased}
.inner{display:block;transform-origin:50% 60%}
${css}
.cap{text-transform:${{ as_said: "none", capitals: "uppercase", lowercase: "lowercase" }[o.letters]}}
</style></head><body><div class="cap"><div class="inner" id="inner"></div></div><script>
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
window.paint = (s) => {
  const el = document.getElementById("inner");
  el.style.transform = s.scale === 1 ? "" : "scale(" + s.scale + ")";
  el.innerHTML = s.words.map((w, i) => '<span class="w' + (i === s.active ? " on" : "") + (${style === "reveal"} && i > s.active ? " later" : "") + '" data-t="' + esc(w) + '">' + esc(w) + "</span>").join(" ");
};
</script></body></html>`;
}

/* ---------- drawing ---------- */

let browser: Promise<Browser> | null = null;
let idle: NodeJS.Timeout | undefined;

// One headless Chrome, with its own throwaway profile, kept warm for a minute and a half so previews are quick.
async function drawingBrowser(): Promise<Browser> {
  clearTimeout(idle);
  browser ??= chromium.launch({ executablePath: chromeExecutable(), headless: true, args: ["--hide-scrollbars", "--force-color-profile=srgb"] }).catch((err) => {
    browser = null;
    throw err;
  });
  return browser;
}
function releaseBrowser(): void {
  clearTimeout(idle);
  idle = setTimeout(async () => {
    const b = await browser?.catch(() => null);
    browser = null;
    await b?.close().catch(() => {});
  }, 90_000);
  idle.unref?.();
}

async function withPage<T>(width: number, height: number, html: string, use: (page: Page) => Promise<T>): Promise<T> {
  const b = await drawingBrowser();
  const ctx = await b.newContext({ viewport: { width, height }, deviceScaleFactor: 1 });
  try {
    const page = await ctx.newPage();
    await page.setContent(html, { waitUntil: "load" });
    await page.evaluate(() => document.fonts.ready.then(() => true));
    return await use(page);
  } finally {
    await ctx.close().catch(() => {});
    releaseBrowser();
  }
}

/* ---------- putting it together ---------- */

const pickColor = (o: CaptionOptions) => (o.color && /^#[0-9a-f]{6}$/i.test(o.color) ? o.color : CAPTION_STYLES[o.style].color);

async function timedWords(video: string, o: CaptionOptions): Promise<{ transcript: Transcript; words: Word[] }> {
  const transcript = await transcribe(video, o.language ?? "auto");
  const edited = o.text?.trim();
  const same = edited !== undefined && norm(edited.replace(/\s+/g, "")) === norm(transcript.text.replace(/\s+/g, "")) && !edited.includes("\n");
  return { transcript, words: edited && !same ? alignText(transcript.words, edited) : transcript.words };
}

// One still of the real video with the real captions on it, so the panel shows exactly what will be made.
export async function previewCaptions(video: string, o: CaptionOptions): Promise<{ image: string; at: number }> {
  const info = await probe(video);
  const { words } = await timedWords(video, o);
  const { pages, states } = layout(words, o.style, info.width, info.height, o.size, lettersOf(o));
  // A caption of a few words, about a third of the way in, mid-sentence.
  const want = (pages[0]?.[0].start ?? 0) + ((pages.at(-1)?.at(-1)?.end ?? 0) - (pages[0]?.[0].start ?? 0)) * 0.33;
  const page = pages.filter((p) => p.length >= Math.min(2, CAPTION_STYLES[o.style].maxWords)).sort((p, q) => Math.abs(p[0].start - want) - Math.abs(q[0].start - want))[0] ?? pages[0];
  const mid = page ? page[Math.floor((page.length - 1) / 2)] : undefined;
  const state = states.find((s) => mid && s.from <= mid.start + 0.01 && s.to > mid.start && s.scale === 1) ?? states.find((s) => s.scale === 1);
  const at = state ? (state.from + state.to) / 2 : Math.min(1, info.duration / 2);

  mkdirSync(CACHE_DIR, { recursive: true });
  const key = createHash("sha1").update(video).digest("hex").slice(0, 12);
  const frame = join(CACHE_DIR, `frame-${key}.jpg`);
  await run(FFMPEG, ["-v", "error", "-y", "-ss", at.toFixed(3), "-i", video, "-frames:v", "1", "-q:v", "3", frame], BIG);
  const background = `data:image/jpeg;base64,${readFileSync(frame).toString("base64")}`;
  const image = join(CACHE_DIR, `preview-${key}.jpg`);
  await withPage(info.width, info.height, pageHtml(o.style, { ...info, position: o.position, size: o.size, color: pickColor(o), letters: lettersOf(o), background }), async (p) => {
    if (state) await p.evaluate((s) => (window as unknown as { paint: (s: unknown) => void }).paint(s), state);
    await p.screenshot({ path: image, type: "jpeg", quality: 85 });
  });
  rmSync(frame, { force: true });
  return { image, at };
}

const stamp = (t: number) => {
  const ms = Math.max(0, Math.round(t * 1000));
  const p = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${p(Math.floor(ms / 3_600_000))}:${p(Math.floor(ms / 60_000) % 60)}:${p(Math.floor(ms / 1000) % 60)},${p(ms % 1000, 3)}`;
};

export interface CaptionResult {
  output: string;
  srt: string;
  captions: number;
  words: number;
  duration: number;
  transcript: string;
  model: string;
}

export async function burnCaptions(video: string, o: CaptionOptions, output: string): Promise<CaptionResult> {
  const info = await probe(video);
  const { transcript, words } = await timedWords(video, o);
  const { pages, states } = layout(words, o.style, info.width, info.height, o.size, lettersOf(o));
  if (!states.length) throw new Error("There are no words to put on screen.");

  const work = mkdtempSync(join(tmpdir(), "flow-captions-"));
  try {
    // Draw every distinct caption once, as a transparent picture the size of the video.
    const files = new Map<string, string>();
    await withPage(info.width, info.height, pageHtml(o.style, { ...info, position: o.position, size: o.size, color: pickColor(o), letters: lettersOf(o) }), async (p) => {
      await p.screenshot({ path: join(work, "blank.png"), omitBackground: true });
      for (const s of states) {
        const id = JSON.stringify([s.words, s.active, s.scale]);
        if (files.has(id)) continue;
        const name = `c${String(files.size + 1).padStart(5, "0")}.png`;
        await p.evaluate((st) => (window as unknown as { paint: (s: unknown) => void }).paint(st), s);
        await p.screenshot({ path: join(work, name), omitBackground: true });
        files.set(id, name);
      }
    });

    // Play those pictures as a second video track with exact timings, and lay it over the original.
    const lines = ["ffconcat version 1.0"];
    let t = 0;
    const add = (file: string, dur: number) => lines.push(`file '${file}'`, `duration ${dur.toFixed(3)}`);
    for (const s of states) {
      if (s.from - t > 0.0005) add("blank.png", s.from - t);
      add(files.get(JSON.stringify([s.words, s.active, s.scale]))!, s.to - s.from);
      t = s.to;
    }
    add("blank.png", Math.max(0.05, info.duration - t + 1));
    lines.push("file 'blank.png'");
    writeFileSync(join(work, "captions.ffconcat"), lines.join("\n") + "\n");

    await run(
      FFMPEG,
      [
        "-v", "error", "-y",
        "-i", video,
        "-f", "concat", "-safe", "0", "-i", join(work, "captions.ffconcat"),
        "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto:eof_action=pass[v]",
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", info.audio === "aac" ? "copy" : "aac", ...(info.audio === "aac" ? [] : ["-b:a", "192k"]),
        "-movflags", "+faststart",
        output,
      ],
      BIG,
    ).catch((err: RunError) => {
      throw new Error(`ffmpeg could not add the captions: ${String(err.stderr ?? err.message).trim().split("\n").at(-1)}`);
    });

    // The same words as a subtitle file, for YouTube and other players that show their own captions.
    const srt = output.replace(/\.[^.]+$/, ".srt");
    writeFileSync(
      srt,
      pages
        .map((p, i) => {
          const next = pages[i + 1]?.[0].start;
          const end = next !== undefined ? Math.min(p[p.length - 1].end + 0.25, next) : p[p.length - 1].end + 0.25;
          return `${i + 1}\n${stamp(p[0].start)} --> ${stamp(end)}\n${p.map((w) => w.text).join(" ")}\n`;
        })
        .join("\n"),
    );
    return { output, srt, captions: pages.length, words: words.length, duration: info.duration, transcript: words.map((w) => w.text).join(" "), model: transcript.model };
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

// "clip.mp4" -> "<dir>/clip-captions.mp4", then -captions-2 and so on, so nothing is ever overwritten.
export function captionedName(dir: string, video: string): string {
  const stem = basename(video, extname(video)).replace(/[^\w.-]+/g, "_");
  for (let i = 1; ; i++) {
    const out = join(dir, `${stem}-captions${i > 1 ? `-${i}` : ""}.mp4`);
    if (!existsSync(out)) return out;
  }
}
