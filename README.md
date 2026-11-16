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
| **E. Review** | LLM reads the changed lines, proposes fixes | advisory only, never auto-applied |

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

## License

MIT
