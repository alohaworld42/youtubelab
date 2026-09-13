---
name: video-cut
description: Cut, trim, re-edit, and inspect videos in this repo — beat-aligned cuts, shot swaps, frame extraction, contact sheets. Use when the user asks to cut/trim/re-edit a video, swap or re-render a shot in a kidsong video, sync cuts to music, or inspect footage frame-by-frame.
---

# Video cutting in this repo

All tools are local: `ffmpeg`/`ffprobe` on PATH, moviepy 2.x + librosa + opencv in `./venv`, and the kidsong editing stack in `pipeline/kidsong/`.

## Core operations

**Inspect footage** (always before editing):
- `ffprobe -v error -show_entries format=duration -show_entries stream=codec_name,width,height -of default=noprint_wrappers=1 <file>`
- Extract frames to LOOK at them (Read tool renders PNGs): `ffmpeg -y -v error -i <file> -ss <t> -frames:v 1 frame.png`
- Contact sheet of any clip: `python -c "from pipeline.kidsong.review import contact_sheet; contact_sheet({'id':'x'}, '<clip>', '<outdir>')"`

**Beat-aligned re-cut** (the kidsong editor):
- `pipeline.kidsong.edit.beat_grid(audio_path)` → bpm + beat times (cached as `<audio>.beats.json`)
- `pipeline.kidsong.edit.build_cut_list(shotlist, beats, duration)` → cuts snapped to beats (±0.45 s, 1.6–4.5 s bounds)
- `pipeline.kidsong.edit.assemble(cfg, cut_list, renders, voice_path, words, duration, out_path)` → final 1080×1920 H.264
- A finished video's paper trail lives next to it: `<base>-shots.json`, `<base>-shots/` (renders), `<base>-shots/review/` (contact sheets + review_log.json). Re-cutting does NOT require re-rendering — reuse the shot mp4s.

**Swap/replace one bad shot** in a finished kidsong video:
1. Find the shot id in `<base>-shots.json`; re-render just it via `pipeline.kidsong.comfy.ComfyClient` (workflow `ltx23_t2v_toon_hires`, patch PROMPT/SEED — bump seed by +977 for a new take).
2. Update the renders dict path and re-run `edit.assemble` with the existing cut list, words, and voice track.

**Plain trims/joins** (no beat logic needed) — prefer stream-copy when not re-timing:
- Trim: `ffmpeg -ss <start> -to <end> -i in.mp4 -c copy out.mp4` (re-encode with `-c:v libx264 -c:a aac` if cutting mid-GOP looks glitchy)
- Join same-codec clips: concat demuxer with a file list; different codecs → re-encode.
- Burn a crossfade between two clips: `ffmpeg -i a.mp4 -i b.mp4 -filter_complex xfade=transition=fade:duration=0.3:offset=<a_dur-0.3> out.mp4`

## Rules
- Never overwrite the source video — write a new timestamped file in `output/`.
- After any edit: ffprobe the result AND extract 2-3 frames at cut points to visually verify.
- Keep audio from the original voice/song track; video-only edits must not touch the audio timeline (cut video to audio, never the reverse).
- Only true junk (extracted check-frames, module test artifacts) gets deleted after verification. Generated media (takes, cuts, songs, superseded videos) are CHANNEL INVENTORY — never delete them.
