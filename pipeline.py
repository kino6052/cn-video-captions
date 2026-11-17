#!/usr/bin/env python
"""
cn-video-captions — Chinese subtitles from burned-in captions, on CPU.

  A  audio   faster-whisper small int8 + word timestamps/probabilities
  B  flag    min(word probability) < --th
  C  screen  OCR only flagged cues, one mid-line frame each
  D  merge   char-level diff: screen fixes substitutions, audio keeps deletions
  D2 re-hear whisper medium on the flagged cues only
  E  review  optional LLM pass over the changed lines (advisory)

Usage:
    python pipeline.py --video in.mp4 --out out/
    python pipeline.py --video in.mp4 --out out/ --start 0 --end 145 --no-llm
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import wave
from difflib import SequenceMatcher

import numpy as np

# huggingface.co is unreachable from some networks and blocks model loading for minutes;
# the mirror is set by default, and HF_HUB_OFFLINE=1 skips the update check entirely.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

CJK = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]")
UI_WORDS = {"运行中", "Opening", "营业中", "bilibili"}
DEFAULTS = {
    "band": (0.85, 0.99),   # vertical slice of the frame that holds the captions
    "scale": 0.65,          # OCR input downscale
    "min_glyph": 20,        # min glyph height, at `scale`; keeps page text out
    "th": 0.35,             # word-probability threshold for "suspect"
    "max_ocr": 24,
    "asr_model": "small",
    "asr2_model": "medium",
    "beam": 5,
}

VIDEO = None
W = H = 0
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, file=sys.stderr, flush=True)


def norm(s):
    return CJK.sub("", s or "")


def sim(a, b):
    a, b = norm(a), norm(b)
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def probe(video, audio_only=False):
    """Frame size and duration. Audio-only input has no video stream to measure."""
    if audio_only:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "json", video], capture_output=True, text=True).stdout
        return 0, 0, float(json.loads(out)["format"]["duration"])
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height,duration",
                          "-of", "json", video], capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    return int(st["width"]), int(st["height"]), float(st.get("duration") or 0)


# ------------------------------------------------------------------ A
def _transcribe_span(model, span, outdir):
    """Transcribe one [start, end) window; returns cues with absolute times."""
    a, b = span
    wav = os.path.join(outdir, f"chunk_{int(a)}_{int(b)}.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a), "-to", str(b),
                    "-i", VIDEO, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
                   check=True)
    with wave.open(wav, "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    os.remove(wav)
    segments, _ = model.transcribe(
        pcm.astype(np.float32) / 32768.0, language="zh", beam_size=1, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        word_timestamps=True, condition_on_previous_text=False)
    out = []
    for s in segments:
        text = s.text.strip()
        if not norm(text):
            continue
        words = [{"w": x.word, "p": round(float(x.probability), 2),
                  "s": round(x.start + a, 2), "e": round(x.end + a, 2)}
                 for x in (s.words or [])]
        out.append({"start": round(s.start + a, 3), "end": round(s.end + a, 3),
                    "asr": text, "words": words})
    return out


def stage_audio(start, end, outdir, chunk=300, jobs=2):
    """Chunked and parallel: same total compute, but progress, lower memory, resume."""
    from faster_whisper import WhisperModel
    from concurrent.futures import ThreadPoolExecutor
    model = WhisperModel(DEFAULTS["asr_model"], device="cpu", compute_type="int8",
                         cpu_threads=max(1, (os.cpu_count() or 4) // max(1, jobs)))
    spans = [(a, min(a + chunk, end)) for a in np.arange(start, end, chunk)]
    log(f"A audio: {len(spans)} chunk(s) of {chunk}s, {jobs} in parallel")
    acc, done = [], 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for part in pool.map(lambda sp: _transcribe_span(model, sp, outdir), spans):
            acc += part
            done += 1
            log(f"A audio: chunk {done}/{len(spans)} done ({len(part)} cues)")
    acc.sort(key=lambda r: r["start"])
    # drop the overlap whisper repeats at a chunk seam
    clean = []
    for r in acc:
        if clean and r["start"] < clean[-1]["end"] - 0.4 and sim(r["asr"], clean[-1]["asr"]) > 0.6:
            continue
        clean.append(r)
    for i, r in enumerate(clean):
        r["n"] = i + 1
        r["minp"] = round(min([w["p"] for w in r["words"]] or [1.0]), 2)
    log(f"A audio: {len(clean)} cues")
    return clean


# ------------------------------------------------------------------ B
def stage_flag(rows, th):
    for r in rows:
        r["flag"] = [f"p{r['minp']}"] if r["minp"] < th else []
        r["llm_fix"] = ""
    n = sum(1 for r in rows if r["flag"])
    log(f"B flag: {n}/{len(rows)} suspects (word probability < {th})")
    return rows


# ------------------------------------------------------------------ C
class Screen:
    """Reads the caption band out of a single video frame."""

    def __init__(self):
        from rapidocr import RapidOCR
        self.engine = RapidOCR()
        self.calls = 0

    def _band(self, t):
        y0 = int(DEFAULTS["band"][0] * H)
        bh = (int(DEFAULTS["band"][1] * H) - y0) // 2 * 2
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", VIDEO, "-frames:v", "1",
             "-vf", f"crop={W}:{bh}:0:{y0}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            capture_output=True).stdout
        rows = len(raw) // (W * 3)
        if rows < 10:
            return None
        return np.frombuffer(raw[:rows * W * 3], np.uint8).reshape(rows, W, 3)

    def read(self, t):
        import cv2
        img = self._band(t)
        if img is None:
            return "", 0.0
        img = cv2.resize(img, None, fx=DEFAULTS["scale"], fy=DEFAULTS["scale"],
                         interpolation=cv2.INTER_AREA)
        res = self.engine(img)
        self.calls += 1
        if res.boxes is None or not len(res.boxes):
            return "", 0.0
        items = []
        for box, text, score in zip(res.boxes, res.txts, res.scores):
            pts = np.array(box).reshape(-1, 2)
            height = float(pts[:, 1].max() - pts[:, 1].min())
            x1, x2 = float(pts[:, 0].min()), float(pts[:, 0].max())
            if height < DEFAULTS["min_glyph"] or score < 0.80:
                continue
            if len(norm(text)) < 2 or text.strip() in UI_WORDS:
                continue
            if not (0.10 <= (x1 + x2) / 2 / img.shape[1] <= 0.90):
                continue
            items.append((float(pts[:, 1].mean()), x1, text.strip(), float(score)))
        if not items:
            return "", 0.0
        items.sort()
        return "".join(i[2] for i in items), round(max(i[3] for i in items), 3)


def stage_screen(rows, screen, cap):
    suspects = [r for r in rows if r["flag"]][:cap]
    for r in suspects:
        r["ocr"], r["ocr_sc"] = screen.read((r["start"] + r["end"]) / 2)
    log(f"C screen: {screen.calls} OCR calls, "
        f"{sum(1 for r in suspects if r.get('ocr'))} with text")
    return rows


# ------------------------------------------------------------------ D
def merge_chars(asr, ocr):
    """Screen supplies substituted characters; audio keeps words the screen lost."""
    A, B = norm(asr), norm(ocr)
    if not B:
        return asr
    out = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, A, B).get_opcodes():
        if tag == "equal":
            out.append(A[i1:i2])
        elif tag == "replace":
            out.append(B[j1:j2] if (j2 - j1) >= 0.5 * (i2 - i1) else A[i1:i2])
        elif tag == "delete":
            out.append(A[i1:i2])
    # "insert" (OCR only) is dropped: the screen hallucinates more than it finds
    return "".join(out)


def stage_fuse(rows):
    for r in rows:
        ocr = r.get("ocr", "")
        if not ocr:
            r["text"], r["status"] = r["asr"], ("audio" if not r["flag"] else "audio?")
        elif sim(r["asr"], ocr) >= 0.40:
            r["text"], r["status"] = merge_chars(r["asr"], ocr), "merged"
        else:
            r["text"], r["status"] = r["asr"], "conflict"
    log("D fuse: " + ", ".join(
        f"{k}={sum(1 for r in rows if r['status'] == k)}"
        for k in ("merged", "conflict", "audio", "audio?")))


# ------------------------------------------------------------------ D2
def stage_second_asr(rows, outdir, cap=60):
    """Re-hear the suspects with a stronger model: a third opinion on the words.

    Capped: on long audio the suspect list runs to hundreds of cues and medium
    costs ~6s each, which silently turns a 10 minute episode into an hour.
    """
    need = sorted([r for r in rows if r["flag"]], key=lambda r: r["minp"])[:cap]
    if len(need) < sum(1 for r in rows if r["flag"]):
        log(f"D2: capping re-hear at {len(need)} lowest-confidence cues")
    if not need:
        return rows
    from faster_whisper import WhisperModel
    model = WhisperModel(DEFAULTS["asr2_model"], device="cpu", compute_type="int8",
                         cpu_threads=os.cpu_count() or 4)
    for r in need:
        wav = os.path.join(outdir, f"cue_{r['n']}.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(max(0, r["start"] - 0.3)),
                        "-to", str(r["end"] + 0.3), "-i", VIDEO, "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", wav], check=True)
        with wave.open(wav, "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        segments, _ = model.transcribe(pcm.astype(np.float32) / 32768.0, language="zh",
                                       beam_size=DEFAULTS["beam"],
                                       condition_on_previous_text=True)
        r["asr2"] = "".join(s.text for s in segments).strip()
        os.remove(wav)
    log(f"D2 re-hear: {len(need)} cues with {DEFAULTS['asr2_model']}")
    return rows


def stage_fuse2(rows):
    for r in rows:
        if not r.get("asr2"):
            continue
        ocr = r.get("ocr", "")
        if ocr:
            base = max([r["asr"], r["asr2"]], key=lambda c: sim(c, ocr))
        else:
            base = r["asr2"]
        r["base"] = "medium" if base == r["asr2"] else "small"
        r["text"] = merge_chars(base, ocr) if ocr else base
        r["status"] = "merged" if ocr else "audio2"


# ------------------------------------------------------------------ E
class Reviewer:
    """Optional LLM reviewer. Disabled unless an OpenAI-compatible endpoint is set."""

    def __init__(self, base=None, model=None, key=None):
        self.base = (base or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
        self.key = key or os.environ.get("OPENAI_API_KEY") or ""
        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.on = bool(self.base and self.key)

    def ask(self, prompt, timeout=180):
        if not self.on:
            return ""
        body = json.dumps({"model": self.model, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}],
                           "reasoning_effort": "none"}).encode()
        req = urllib.request.Request(f"{self.base}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                msg = json.loads(resp.read())["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning_content") or ""
        except Exception as e:  # never let the reviewer break a run
            log(f"LLM skipped: {type(e).__name__}")
            return ""


def parse_list(text):
    raw = re.sub(r"```(?:json)?|```", "", text or "").strip()
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        value = json.loads(m.group(0))
        return value if isinstance(value, list) else []
    except Exception:
        return []


def stage_review(rows, reviewer):
    if not reviewer.on:
        log("E review: skipped (no LLM endpoint)")
        return {"description": "", "remaining": []}
    sample = "\n".join(f"{r['start']:.0f}|{r['text']}" for r in rows[:40])
    desc = reviewer.ask("下面是一段中文视频的字幕片段：\n" + sample +
                        "\n\n用 3 句话说明视频内容与结构。只输出这三句。", timeout=90).strip()
    changed = [r for r in rows if r["status"] in ("merged", "audio2", "conflict", "audio?")]
    if not changed:
        return {"description": desc, "remaining": []}
    out = reviewer.ask(
        "视频主题：" + desc + "\n\n字幕：\n"
        + "\n".join(f"{r['n']}|{r['text']}" for r in changed[:80])
        + "\n\n只报确实错误、会让读者看不懂或误解的行：同音错字、专有名词写错、"
          "缺字导致不成句。这是口语字幕，口语、重复、短句都不算错。\n"
          "不要润色、不要改语序、不要改成书面语、不要因为标点或风格而报错。\n"
          '严格输出 JSON 数组 [{"n": 序号, "issue": "问题", "fix": "最小改动"}]；'
          "fix 只能替换个别词，必须保持原句结构与长度，不得增删分句。全部通过输出 []。")
    remaining = [x for x in parse_list(out) if isinstance(x, dict)]
    # structure guard: reject any suggestion that rewrites rather than repairs
    kept = []
    for x in remaining:
        fix = str(x.get("fix", "")).strip()
        if not fix:
            kept.append(x)
            continue
        src = next((r["text"] for r in changed if str(r["n"]) == str(x.get("n"))), "")
        ratio = len(fix) / max(1, len(src))
        if 0.7 <= ratio <= 1.3 and sim(fix, src) >= 0.5:
            kept.append(x)
        else:
            log(f"E review: dropped a structural rewrite at cue {x.get('n')}")
    remaining = kept
    log(f"E review: {len(remaining)} left")
    return {"description": desc, "remaining": remaining}


# ------------------------------------------------------------------ T
def to_pinyin(text):
    from pypinyin import pinyin, Style
    return " ".join(syllable[0] for syllable in pinyin(text, style=Style.TONE))


def stage_triple(rows, reviewer, chunk=30):
    """Chinese (already there) + pinyin (local) + English (LLM, batched)."""
    for r in rows:
        r["pinyin"] = to_pinyin(r["text"])
    if reviewer.on:
        for c0 in range(0, len(rows), chunk):
            part = rows[c0:c0 + chunk]
            out = reviewer.ask(
                "把下面的中文字幕逐条翻成英文：自然、忠实原文、不增不减，不要解释，不要合并句子。\n"
                + "\n".join(f"{r['n']}|{r['text']}" for r in part)
                + '\n\n只输出 JSON 数组 [{"n": 序号, "en": "英文"}]。')
            for item in parse_list(out):
                if not isinstance(item, dict):
                    continue
                n = str(item.get("n", "")).strip()
                hit = next((r for r in part if str(r["n"]) == n), None)
                if hit:
                    hit["en"] = str(item.get("en", "")).strip()
    for r in rows:
        r.setdefault("en", "")
    log(f"T triple: pinyin for {len(rows)} cues, "
        f"english for {sum(1 for r in rows if r['en'])}")


# ------------------------------------------------------------------ F
def ts(x):
    h, m, s = int(x // 3600), int(x % 3600 // 60), x % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def emit(rows, review, outdir, video):
    rows.sort(key=lambda r: r["start"])
    for n, r in enumerate(rows, 1):
        r["n"] = n
    tripled = any(r.get("pinyin") for r in rows)
    with open(os.path.join(outdir, "fused.srt"), "w", encoding="utf-8") as fh:
        for r in rows:
            body = r["text"]
            if tripled:
                body += "\n" + r.get("pinyin", "") + "\n" + r.get("en", "")
            fh.write(f"{r['n']}\n{ts(r['start'])} --> {ts(max(r['end'], r['start'] + 0.4))}\n"
                     f"{body}\n\n")
    json.dump({"video": video, "description": review["description"],
               "cues": [{k: r.get(k) for k in ("n", "start", "end", "text", "pinyin", "en",
                                               "asr", "asr2", "base", "status", "minp",
                                               "ocr", "ocr_sc", "flag", "words")} for r in rows],
               "remaining": review["remaining"]},
              open(os.path.join(outdir, "transcript.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    count = lambda s: sum(1 for r in rows if r["status"] == s)  # noqa: E731
    lines = [f"# {os.path.basename(video)}", "",
             "## Content", "", review["description"] or "(LLM review disabled)", "",
             "## Run", "", "| stage | result |", "|---|---|",
             f"| cues | {len(rows)} |",
             f"| flagged | {sum(1 for r in rows if r['flag'])} |",
             f"| fixed by screen | {count('merged')} |",
             f"| re-heard by {DEFAULTS['asr2_model']} | {count('audio2') + count('merged')} |", "",
             "## Screen overruled audio", "", "| time | audio | final |", "|---|---|---|"]
    for r in rows:
        if r["status"] == "merged" and r["text"] != r["asr"]:
            lines.append(f"| {r['start']:.1f}s | {r['asr']} | {r['text']} |")
    lines += ["", "## Still questionable", ""]
    rem = [f"- {x.get('n')}: {x.get('issue', '')}" for x in review["remaining"]]
    lines += rem or ["- none"]
    open(os.path.join(outdir, "report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    log(f"F emit: {outdir}/transcript.json, fused.srt, report.md")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Chinese burned-in caption extractor")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=0, help="default: full video")
    ap.add_argument("--th", type=float, default=DEFAULTS["th"])
    ap.add_argument("--max-ocr", type=int, default=DEFAULTS["max_ocr"])
    ap.add_argument("--chunk", type=int, default=300, help="ASR chunk length in seconds")
    ap.add_argument("--jobs", type=int, default=2, help="chunks transcribed in parallel")
    ap.add_argument("--max-reread", type=int, default=60,
                    help="cap on cues re-heard with the medium model")
    ap.add_argument("--no-second-pass", action="store_true", help="skip whisper medium re-hear")
    ap.add_argument("--audio-only", action="store_true",
                    help="input has no video stream: skip frame probing and all screen OCR")
    ap.add_argument("--triple", action="store_true",
                    help="each cue also gets pinyin (local) and English (LLM)")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--llm-base", default=None, help="OpenAI-compatible base URL")
    ap.add_argument("--llm-model", default=None)
    args = ap.parse_args()

    global VIDEO, W, H
    VIDEO = os.path.abspath(args.video)
    W, H, duration = probe(VIDEO, audio_only=args.audio_only)
    end = args.end or duration
    os.makedirs(args.out, exist_ok=True)

    rows = stage_audio(args.start, end, args.out, args.chunk, args.jobs)
    stage_flag(rows, args.th)
    if args.audio_only:
        # no screen to consult, but keep the flags: they are what drives the medium re-hear
        log("C screen: skipped (audio-only input)")
        for r in rows:
            r["ocr"] = ""
        stage_fuse(rows)
    else:
        stage_screen(rows, Screen(), args.max_ocr)
        stage_fuse(rows)
    if not args.no_second_pass:
        rows = stage_second_asr(rows, args.out, args.max_reread)
        stage_fuse2(rows)
    reviewer = Reviewer(base=args.llm_base, model=args.llm_model)
    if args.no_llm:
        reviewer.on = False
    if args.triple:
        stage_triple(rows, reviewer)
    emit(rows, stage_review(rows, reviewer), args.out, VIDEO)


if __name__ == "__main__":
    main()
