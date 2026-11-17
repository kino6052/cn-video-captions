# cn-video-captions

Accurate Chinese subtitles from videos where the captions are **burned into the picture** — running on plain CPU.

A single-file pipeline: speech recognition as the backbone, on-screen OCR spent only where it changes the answer.

## Why not just OCR every frame?

Burned-in subtitle extraction looks like an OCR problem, but dense OCR is slow and unreliable:

| Approach | Time (5:44 video, 4-core CPU) | Failure mode |
|---|---|---|
| OCR every frame | ~30 min | picks up page text / UI behind the captions |
| Audio only | ~1.5 min | misreads names and homophones (`鈴鋼` for `灵感`) |
| **This pipeline** | **~3.5 min** | both sources correct each other |

Audio runs ~4× realtime and never misreads pixels. OCR reads the exact characters and punctuation the
creator wrote. Their errors are complementary, so the pipeline uses audio for text and timing, and
consults the screen only for the lines audio is unsure about.

## Pipeline

| Stage | What | Why |
|---|---|---|
| **A. Audio** | `faster-whisper small` int8, `word_timestamps=True` | 4× realtime; per-word probability is the best free error signal |
| **B. Flag** | `min(word probability) < 0.35` | real errors are almost always low-probability (`鋼` 0.26, `豆` 0.24); normal words score 0.4+ |
| **C. Screen** | OCR only flagged lines, one mid-line frame each | ~1.3 s per line vs 0.5 s × 688 frames |
| **D. Merge** | character-level diff: **substitutions** from screen, **deletions** kept from audio | screen fixes `铃铛→灵感`, audio keeps the `想都` the screen dropped |
| **D2. Re-hear** | `whisper medium` on the flagged cues only | fixes what both small models and OCR get wrong (`Exile/Excele → Excel`) |
| **E. Review** | LLM reads the changed lines, proposes fixes | advisory only, never applied; told to report real errors only (no polishing, no reordering) and a code guard drops any rewrite that leaves the 0.7–1.3 length band |

Flagging is naturally sparse (~9% of cues), which is what keeps the whole thing inside 4 minutes.

## Measured (145 s excerpt, 78 cues, i7-1165G7)

| | |
|---|---|
| Total | 105 s (audio 32 s, OCR 7 frames 12 s, medium 50 s, LLM 16 s) |
| Flagged | 7 / 78 cues → 7 OCR frames |
| Fixed by screen | `以前想都不敢想的铃铛库网站` → `…灵感库网站` (kept `想都` from audio) |
| Fixed by medium | `Exile` → `Excel`, `最后就要复许` → `最后交付成一个HTML网页` |
| Left flagged | 0 |

## Install

```bash
# ffmpeg is required for frame and audio extraction
sudo apt install ffmpeg          # or: brew install ffmpeg

python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# whisper weights come from Hugging Face; set a mirror if you need one
export HF_ENDPOINT=https://hf-mirror.com
```

## Usage

```bash
python pipeline.py --video input.mp4 --out out/

# options
python pipeline.py --video input.mp4 --out out/ --start 0 --end 145
python pipeline.py --video input.mp4 --out out/ --th 0.35 --max-ocr 24
python pipeline.py --video input.mp4 --out out/ --no-llm     # skip LLM stages
```

Outputs in `--out`:

```
transcript.json   cues with start/end/text + asr + status + similarity + word-level timing & probability
fused.srt         standard SRT, drop-in for any player
report.md         what was flagged, what the screen overruled, what still looks wrong
```

`transcript.json` is the programmatic artifact — per-word `start`/`end`/`probability` make hover,
translation and karaoke highlighting straightforward:

```json
{"n": 1, "start": 0.0, "end": 2.18, "text": "以前想都不敢想的灵感库网站",
 "asr": "以前想都不敢想的铃铛库网站", "status": "merged",
 "words": [{"w": "以前", "p": 0.74, "s": 0.0, "e": 0.48}, ...]}
```

The bundled example is under [`examples/`](examples/).

## Tuning

| Knob | Default | Effect |
|---|---|---|
| `--th` | `0.35` | lower = fewer OCR calls, more unflagged errors; 0.35 and 0.45 gave identical output in testing |
| `--max-ocr` | `24` | hard cap on OCR frames per run |
| `CROP_BAND` | `(0.85, 0.99)` of frame height | where captions live; widen for subtitles that sit higher |
| `MIN_GLYPH` | `20` px (at 0.65 scale) | rejects page text behind the caption; the main defence against UI bleed |

## Limitations

- English words and proper nouns can still be wrong; D2 helps, a dictionary would help more
- Lines with no on-screen caption can only be cross-checked against audio
- Timing comes from audio VAD, so cue boundaries differ from the creator's by ~0.2–0.5 s
- The LLM reviewer is opinionated — it flags stylistic choices as errors. Treat `report.md` as advice
- Tuned on 1080p Mandarin talking-head + screen-recording content; Cantonese and vertical video are untested


## Lessons from building it

Each of these cost real time; the number next to it is how we knew.

1. **Audio first, screen second.** OCR every frame was ~30 min and picked up page text; audio was 1.5 min and never misread pixels. The cheap modality should be the backbone.
2. **Segment-level confidence is useless, word-level is gold.** `avg_logprob` was −0.16 on a cue containing the badly wrong `鋼` (p=0.26). Per-word probability found every real error; the segment score found none.
3. **Merge characters, not lines.** The screen dropped `想都` while fixing `铃铛→灵感`. Line replacement loses one to gain the other; a char-level diff keeps both.
4. **Spend the expensive model only on suspects.** `medium` on 7 flagged cues (50 s) fixed 2/2 of the remaining errors. On all 78 cues it would have cost 6 minutes for the same two fixes.
5. **Reasoning models are slow for translation.** `reasoning_effort: "none"` took a call from 3.7 s to 1.3 s with identical output; parallel chunks made the whole transcript 3.7 s.
6. **Check CPU-vs-wall before optimising.** An "audio-only" run took 110 s wall and **1.2 s CPU** — it was blocked on an unreachable `huggingface.co`, not computing. `HF_ENDPOINT` fixed it; no algorithm change would have.
7. **Prompts need a scope guard, and so does code.** "Report errors" became "rewrite sentences" until the prompt forbade polishing *and* a length/similarity check dropped rewrites anyway. Models drift; make the invariant executable.
8. **A threshold that never changes the output is not a knob.** 0.35 and 0.45 produced byte-identical transcripts — keep the cheaper one.
9. **Gated endpoints fail transparently if you probe them properly.** 小宇宙's transcripts are real (5/5 episodes) but every path returns the same 401 and the CDN has no sidecar. Ten minutes of probing beat hours of scraping.

## License

MIT
