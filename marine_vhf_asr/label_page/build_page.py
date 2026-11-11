"""
Build the data files for label_page/radio-log-check.html from real_clips_job.py output:
150 clips per held-out feed (seed 0, interleaved), WhisperX words as a second draft,
audio as base64 MP3 in chunks of 30. Needs ffmpeg.

    python marine_vhf_asr/label_page/build_page.py <dir with real/candidates.jsonl, real/clips/> <WhisperX transcript root>

Writes <dir>/page/{clips.json,audio_N.json,picked.json}; publish them beside the HTML page.
"""
import json, os, random, subprocess, base64, sys, re
S, TX = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(f"{S}/real/candidates.jsonl")]
rng = random.Random(0)
by = {}
for r in rows: by.setdefault(r["feed"], []).append(r)
picked = []
for feed, v in sorted(by.items()):
    rng.shuffle(v); by[feed] = v[:150]
for a, b in zip(by["17329"], by["26694"]): picked += [a, b]   # interleave so any prefix is balanced
def whisper(r):
    p = f"{TX}/{r['feed']}/{os.path.basename(r['file'])[:-4]}.json"
    if not os.path.exists(p): return ""
    words = [w["word"] for s in json.load(open(p)) for w in s.get("words", []) if "start" in w and r["start"] <= (w["start"] + w.get("end", w["start"])) / 2 <= r["end"]]
    return re.sub(r"\s+", " ", " ".join(words)).strip()
os.makedirs(f"{S}/page", exist_ok=True)
clips, chunk, k = [], {}, 0
for i, r in enumerate(picked):
    mp3 = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{S}/real/clips/{r['id']}.wav", "-ac", "1", "-ar", "16000", "-b:a", "32k", "-f", "mp3", "-"], capture_output=True, check=True).stdout
    chunk[r["id"]] = base64.b64encode(mp3).decode()
    clips.append({"id": r["id"], "feed": r["feed"], "start": r["start"], "end": r["end"], "draft": r["draft"].lower(), "whisper": whisper(r), "chunk": k})
    if len(chunk) == 30 or i == len(picked) - 1:
        json.dump(chunk, open(f"{S}/page/audio_{k}.json", "w")); chunk = {}; k += 1
json.dump(clips, open(f"{S}/page/clips.json", "w"))
json.dump(picked, open(f"{S}/page/picked.json", "w"))
d = [c["end"] - c["start"] for c in clips]
print(len(clips), "clips", k, "chunks", f"{sum(d)/60:.1f} min audio", "empty v6 drafts", sum(not c["draft"] for c in clips), "empty whisper", sum(not c["whisper"] for c in clips))
for c in clips[:6]: print(c["feed"], f"{c['end']-c['start']:.1f}s |", c["draft"], "|", c["whisper"])
