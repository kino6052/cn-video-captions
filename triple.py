#!/usr/bin/env python
"""
Make an existing transcript triple: Chinese + pinyin + English.

Takes what the pipeline already produced and never touches audio or video again,
so a slow transcription is done once.

    python triple.py out/transcript.json              # -> out/transcript.triple.srt
    python triple.py out/fused.srt                    # plain SRT also accepted
    python triple.py out/transcript.json --limit 20   # try it on a few cues first

Pinyin is local and free (`pypinyin`). English is one LLM call per 30 cues; without
an endpoint you still get Chinese + pinyin.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from pypinyin import pinyin, Style

T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:5.1f}s]", *a, file=sys.stderr, flush=True)


def to_pinyin(text):
    return " ".join(syllable[0] for syllable in pinyin(text, style=Style.TONE))


class Translator:
    def __init__(self, base=None, model=None, key=None, timeout=180):
        self.base = (base or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
        self.key = key or os.environ.get("OPENAI_API_KEY") or ""
        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        self.on = bool(self.base and self.key)

    def ask(self, prompt):
        body = json.dumps({"model": self.model, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}],
                           "reasoning_effort": "none"}).encode()
        req = urllib.request.Request(f"{self.base}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                msg = json.loads(resp.read())["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning_content") or ""
        except Exception as e:
            log(f"LLM failed: {type(e).__name__}")
            return ""

    def english(self, cues, chunk=40, jobs=4):
        parts = [cues[i:i + chunk] for i in range(0, len(cues), chunk)]
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(self._one_chunk, parts))
        log(f"english {sum(1 for c in cues if c.get('en'))}/{len(cues)}")

    def _one_chunk(self, part):
            reply = self.ask(
                "把下面的中文字幕逐条翻成英文：自然、忠实原文、不增不减，"
                "不要解释，不要合并句子。\n"
                + "\n".join(f"{i + 1}|{c['text']}" for i, c in enumerate(part))
                + '\n\n只输出 JSON 数组 [{"n": 序号, "en": "英文"}]。')
            raw = re.sub(r"```(?:json)?|```", "", reply or "").strip()
            m = re.search(r"\[.*\]", raw, re.S)
            got = {}
            if m:
                try:
                    for item in json.loads(m.group(0)):
                        if isinstance(item, dict):
                            got[str(item.get("n"))] = str(item.get("en", "")).strip()
                except Exception:
                    log(f"chunk {start//chunk + 1}: unparsable reply")
            for i, cue in enumerate(part, 1):
                cue["en"] = got.get(str(i), "")
            log(f"chunk done: {sum(1 for c in part if c.get('en'))}/{len(part)}")


def load(path):
    """Accept the pipeline's transcript.json or a plain .srt."""
    if path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        return [{"start": c["start"], "end": c["end"], "text": c["text"]}
                for c in data["cues"]]
    text = open(path, encoding="utf-8").read().strip()
    out = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", lines[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        out.append({"start": g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                    "end": g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
                    "text": " ".join(lines[2:]).strip()})
    return out


def ts(x):
    h, m, s = int(x // 3600), int(x % 3600 // 60), x % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser(description="add pinyin and English to a transcript")
    ap.add_argument("input", help="transcript.json or .srt")
    ap.add_argument("-o", "--out", default=None, help="output .srt (default: <input>.triple.srt)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cues")
    ap.add_argument("--no-english", action="store_true")
    ap.add_argument("--llm-base")
    ap.add_argument("--llm-model")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    cues = load(args.input)
    if args.limit:
        cues = cues[:args.limit]
    for c in cues:
        c["pinyin"] = to_pinyin(c["text"])
    log(f"pinyin for {len(cues)} cues")

    if not args.no_english:
        t = Translator(base=args.llm_base, model=args.llm_model)
        if t.on:
            t.english(cues, jobs=args.jobs)
        else:
            log("no LLM endpoint: English skipped (set OPENAI_BASE_URL and OPENAI_API_KEY)")

    out = args.out or (os.path.splitext(args.input)[0] + ".triple.srt")
    with open(out, "w", encoding="utf-8") as fh:
        for n, c in enumerate(cues, 1):
            body = "\n".join([c["text"], c["pinyin"], c.get("en", "")]).rstrip("\n")
            fh.write(f"{n}\n{ts(c['start'])} --> {ts(c['end'])}\n{body}\n\n")
    log(f"wrote {out} ({len(cues)} cues)")


if __name__ == "__main__":
    main()
