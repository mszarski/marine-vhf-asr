# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.14.1", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0",
#     "huggingface_hub==2.2.0", "mistral-common[audio]==1.12.0",
#     "soundfile==0.14.0", "scipy==1.18.1", "numpy", "kagglehub", "requests",
# ]
# ///
"""
Build candidate clips for a hand-corrected real test set from the Kaggle marine
radio chatter recordings: download the picked files, cut transmissions with the
energy gate (segment.py), draft a transcript with an adapter, and upload clips +
candidates.jsonl to a private Hub dataset. Kaggle is unreachable from the dev
container, hence a job. The audio is Broadcastify-derived: internal use only.

OUT_REPO must already hold picks.json: {"<feed>": ["Marine_audio/<feed>/<file>.mp3", ...]}.

    hf jobs uv run --flavor a10g-small --timeout 90m --secrets HF_TOKEN \\
        --env CODE_REPO=<user>/marine-vhf-asr-code --env ADAPTER=<user>/voxtral-3B-marine-vhf-v6 \\
        --env OUT_REPO=<user>/marine-vhf-real-test --env PER_FILE=5 \\
        marine_vhf_asr/jobs/real_clips_job.py
"""
import json, os, sys
import kagglehub, numpy as np, soundfile as sf
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
sys.path.insert(0, snapshot_download(os.environ["CODE_REPO"]))
from marine_vhf_asr import evaluate
from marine_vhf_asr.segment import find_transmissions
from marine_vhf_asr.sources import SR, load_audio

KAGGLE = "linogova/marine-radio-chatter-bridge-2-bridge-communication"
out_repo, per_file = os.environ["OUT_REPO"], int(os.environ.get("PER_FILE", "5"))
pad = 0.25  # seconds of audio kept around each burst
picks = json.load(open(hf_hub_download(out_repo, "picks.json", repo_type="dataset")))
rng = np.random.default_rng(0)
os.makedirs("/tmp/out/clips", exist_ok=True)
rows, audios = [], []
for feed, names in sorted(picks.items()):
    for name in names:
        try:
            x = load_audio({"path": kagglehub.dataset_download(KAGGLE, path=name)})
        except Exception as e:
            print("SKIP", name, repr(e)[:200], flush=True)
            continue
        found = find_transmissions(x, SR)
        keep = sorted(rng.choice(len(found), size=min(per_file, len(found)), replace=False)) if found else []
        stem = os.path.basename(name)[:-4]
        for i in keep:
            a, b = max(0.0, found[i][0] - pad), min(len(x) / SR, found[i][1] + pad)
            clip = x[int(a * SR):int(b * SR)]
            cid = f"{stem}_{int(a * 10):05d}"
            sf.write(f"/tmp/out/clips/{cid}.wav", clip, SR, subtype="PCM_16")
            rows.append({"id": cid, "feed": feed, "file": name, "start": round(a, 2), "end": round(b, 2)})
            audios.append(clip)
        print(name, f"{len(x) / SR:.0f}s", len(found), "bursts, kept", len(keep), flush=True)

model, processor = evaluate.load_model(adapter=os.environ["ADAPTER"])
for i in range(0, len(rows), 8):
    for row, hyp in zip(rows[i:i + 8], evaluate.transcribe(model, processor, evaluate.BASE_MODEL, audios[i:i + 8])):
        row["draft"] = hyp
    print("transcribed", min(i + 8, len(rows)), "/", len(rows), flush=True)
with open("/tmp/out/candidates.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
HfApi().upload_folder(repo_id=out_repo, repo_type="dataset", folder_path="/tmp/out",
                      commit_message=f"Real test candidates: {len(rows)} clips, drafts by {os.environ['ADAPTER']}")
print("UPLOADED", len(rows), flush=True)
