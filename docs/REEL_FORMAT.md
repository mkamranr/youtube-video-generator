# Reel format

A per-channel switch for vertical shorts that should stop the scroll in the first two seconds,
prove what they promise, and end on one useful call to action. It was built for a tech channel
(AI tools, local LLMs, coding agents, GPU/inference, private AI) but nothing in it is specific to
one channel.

Turn it on in **Settings → Channels → edit → Reel format (9:16)**. It applies to a video when all
of these hold (`pipeline._reel`):

- the channel's `reel_format` is on (snapshotted onto the job like every channel field);
- the video is 9:16;
- it is a spoken video (not a song, not a kids story);
- its source is a topic, a description or a repo URL. A pasted transcript is spoken word for word
  and is never restructured; a YouTube URL job skips drafting.

Everything else (other channels, other aspects, songs, stories) works exactly as before.

## What changes

| Stage | Without reel format | With reel format |
|---|---|---|
| Script | one paragraph (`write_script_from_prompt`) | 5+ scored hooks, a scene-by-scene script, one CTA (`write_reel_script`) |
| Script check | extend if short | `reel.validate`, then one `tighten_reel_script` rewrite if it fails (never a faster voice) |
| Approval page | edit the paragraph | pick the hook, edit scene voiceovers, headlines and the CTA, attach footage |
| Clip prompts | equal word split per clip | each clip gets the scene spoken over it, plus that scene's visual direction; clip 1 gets an opening rule |
| Hook card | 0.2 s, `job.hook`, cover headline or SEO title | **0.0 s with no fade-in**, the hook scene's headline (a headline over 6 words is dropped); in both, the first candidate that fits in whole words, shrinking the type before falling back, never a split word or "…" |
| End card | "Subscribe for more" | the reel's own CTA sentence (wrapped) |
| Captions off | no hook card and no end card: the story board cards are the only text on the video | same |
| Captions and cards | Shorts bottom margin (14%) | Reels safe area: top 12%, bottom 22%, left 6%, right 17%; caption text ≥ 4.5% of the frame height |
| Demo footage | n/a | an uploaded screen recording or screenshot replaces a scene's AI clips |
| Post copy | YouTube SEO only | YouTube SEO **plus** Instagram, Facebook and Shorts captions |
| Cover / thumbnail copy | from the title | sells the chosen hook; a claim that needs context is asked as a question |
| Publishing | upload any time | YouTube upload locked until a person approves the finished video |

Two changes apply to **every** channel:

- the narration loses dead air at both ends (`tts.trim_silence`, Settings: *Trim narration silence*);
- every finished video is checked (`services/qa.py`, Settings: *Check finished videos*).

## The script (`plan["reel"]`)

```json
{
  "hooks": [{"text": "...", "category": "outcome|surprise|problem_solution|curiosity|comparison|demo_first|list",
             "scores": {"clarity": 1-5, "relevance": 1-5, "curiosity": 1-5, "specificity": 1-5,
                        "factual": 1-5, "accessibility": 1-5},
             "supported": true, "rejected": "" }],
  "chosen": 0,
  "scenes": [{"n": 1, "role": "hook|context|demo|payoff|cta", "voiceover": "...", "headline": "...",
              "support_text": "", "visual": "...", "asset_needed": "", "transition": "cut",
              "emphasis": [], "has_claim": false, "asset": "demo_03.png"}],
  "cta": {"type": "save|share|question|follow|repo", "text": "..."},
  "central_message": "...", "audience_benefit": "...",
  "script_version": 1, "issues": []
}
```

`plan["narration"]` is always the scene voiceovers joined with spaces. TTS, captions and
everything downstream still read only that, so nothing past the draft needed to learn about
scenes. `plan["clip_scene"]` (clip index → scene index) is written after TTS.

### Hook selection (`reel.select_hook`)

A hook is struck out (`rejected` says why) when:

- the writer marked it `supported: false`;
- its `factual` score is below 4 (factual support is mandatory, whatever the other scores say);
- it uses a hype word (revolutionary, unlimited, best ever, game changer, mind-blowing, …);
- it claims *free*, *offline*, *open-source*, *private* or *on a laptop* without support.

Of the rest, the highest total wins, then clarity, then the earlier hook. The winner becomes the
first spoken line of scene 1. Every candidate stays in the plan with its scores, so a later
experiment can try another.

### Script check (`reel.validate`)

Issues are shown on the approval page and logged. The check flags:

- fewer than 5 hooks;
- no usable hook;
- scene 1 that is not the hook;
- an opener like "Today we're…", "In this video…" or a greeting;
- no demo scene;
- anything but exactly one CTA, and the CTA not being last;
- no central message;
- a word count more than ±15% off the target (2.5 words/s × the midpoint of the video length);
- a repeated sentence.

## Demo footage

`POST /jobs/{id}/scenes/{n}/asset` (multipart `file`: mp4, mov, webm, m4v, mkv, png, jpg, webp;
up to 200 MB) works while the script awaits approval or on a finished video.

`video.clip_from_asset` handles the footage:

- It fits the whole picture over a blurred copy of itself; it never crops, because a screenshot's
  edges are usually the interface.
- A still gets a slow push-in.
- A short recording loops; a long one is trimmed.
- The result is encoded to the rendered clips' codec, size, frame rate, timescale and audio layout,
  so the join stays a stream copy.

Footage clips are built after the rendered ones, need no GPU, and the AI clip after one starts
from text rather than from a screen's last frame. On a finished video, attaching footage marks
that scene's clips stale; regenerate them (instant) and reassemble.

## Checks (`plan["qa"]`)

Each check is `{id, label, status: pass|warn|fail|manual, detail}`. The thresholds are in Settings.

| id | flags |
|---|---|
| `container` | not MP4 / H.264 / AAC |
| `aspect` | frame not the job's aspect (±1%) — **fail** |
| `resolution` | portrait below 1080×1920 — warn only; never upscale poor footage just to hit it |
| `fps` | variable frame rate |
| `duration` | off the planned clip lengths by more than `QA_DURATION_TOLERANCE` (0.6 s) |
| `first_frame` | blank or near-black first frame — **fail** |
| `black` | black stretches ≥ 0.25 s |
| `static` | picture frozen ≥ `QA_FREEZE_SECONDS` (4 s) |
| `edge_silence` | silence at the start or end > `QA_EDGE_SILENCE` (0.4 s) |
| `gap_silence` | silence inside > `QA_GAP_SILENCE` (1.5 s) |
| `clipping` | true peak > −1 dBTP (warn), > 0 dBTP (**fail**) |
| `loudness` | integrated loudness more than 2 LU off `LOUDNESS_LUFS` |
| `end_cut` | narration runs into the last 0.3 s — **fail** |
| `captions` | a caption group needing more than two lines, or one word wider than the caption area |
| `caption_size` | reel captions below `REEL_MIN_CAPTION_FONT` |
| `assets` | a missing clip or demo file — **fail** |
| `review` | always `manual`: claims vs sources, pronunciation, other platforms' watermarks |

They run after every assembly and on **Re-run checks**, and never fail a job.

## Publish approval

`Job.publish_approved` is set by **Approve for publish** on the job page
(`POST /jobs/{id}/publish-approve {"approved": true}`). It is refused while clips are stale or
the video needs a reassembly. It asks for confirmation if any check failed. Anything that changes
the video clears it: reassembly, a regenerated clip, new footage. For reel channels
`POST /jobs/{id}/youtube-upload` answers `409` until it is set; downloads are never locked.

## Settings

| Key | Default | Meaning |
|---|---|---|
| `NARRATION_TRIM_SILENCE` | on | trim dead air at both ends of the narration (all channels) |
| `NARRATION_EDGE_SILENCE` | 0.15 | seconds of quiet kept at each end |
| `REEL_SAFE_TOP` / `BOTTOM` / `LEFT` / `RIGHT` | 0.12 / 0.22 / 0.06 / 0.17 | Reels UI-safe region, fractions of the frame |
| `REEL_MIN_CAPTION_FONT` | 0.045 | smallest reel caption, fraction of the frame height |
| `QA_ENABLED` | on | run the checks |
| `QA_FREEZE_SECONDS`, `QA_EDGE_SILENCE`, `QA_GAP_SILENCE`, `QA_DURATION_TOLERANCE` | 4 / 0.4 / 1.5 / 0.6 | check thresholds |

## Migration

None needed. Channels without `reel_format` read as off, jobs without `publish_approved` read as
not approved (only reel uploads care), and plans without `reel`, `social` or `qa` render exactly
as before.

## Not done (yet)

- Instagram Insights capture, a per-reel performance dashboard, and hook/format experiment
  tracking. The plan already stores every hook with its scores, the chosen one, the CTA type and
  `script_version`, so results can be joined to it later.
- Two materially different edits (result-first vs problem-first) of the same source. Each would
  double the GPU render.
- Automatic footage capture (for example a repo page recording). Footage is uploaded by hand.
