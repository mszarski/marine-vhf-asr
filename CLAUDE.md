# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `HANDOFF.md` first for current state, model lineage, latest numbers,
history and next steps.** `marine_vhf_asr/README.md` holds every run's results
tables; new results get recorded there. This file holds only durable rules.

## Project Overview

`marine_vhf_asr` builds a marine VHF radio transcription model (port control /
VTS, pilots, coastguard, bridge-to-bridge): an SMCP dialogue generator, a VHF
channel simulator, data loaders, and LoRA fine-tune/eval scripts for
`mistralai/Voxtral-Mini-3B-2507`, following the recipe of
`pphilip/voxtral-3B-atc-transcribe` (the ATC equivalent). Goals, in order:
**general, not port-specific** (must work for any port and for vessels on no
AIS list); usable accuracy on real radio, vessel names above all (a wrong ship
is useless or dangerous); honest numbers.

## Working with the user

- Messages are short and casual. Reply with a clear verdict first, then a
  compact results table. Always say **which model** a number belongs to (the
  user once read the untrained baseline's 69% WER as our model's result).
- Wants general solutions; rejected a port-specific (NY-trained) plan.
- Values honesty over flattering numbers. Say when a test is weak; correct
  earlier claims proactively; don't present judgment as measurement (e.g.
  calling unverified output "plausible" with no reference to check it against).
- Spending: HF Jobs spending is authorized for this project. Give a cost
  estimate before each round and the actual cost after. **Ask before launching
  a new training run**; cheap evals/data jobs (<$5) can go without asking.
- For choices, give numbered options + a recommendation ("lets do 1 and 2").
- Asked to be told when a long run finishes: keep watching, report unprompted.
- Git: commit and push your work; no PR unless asked. No model names in repo
  files or commit message text (beyond the required attribution trailer).
- Session state goes in `HANDOFF.md`, not in this file.

## Commands

```bash
pip install -r marine_vhf_asr/requirements.txt          # generation + metrics (numpy, scipy)
pip install -r marine_vhf_asr/requirements-train.txt    # GPU training / eval (torch, transformers, peft, datasets, ...)

pytest marine_vhf_asr/tests/                            # CPU only; optional-dep tests importorskip
python -m marine_vhf_asr.synthesize --out data/text --dialogues 50 --text-only
python -m marine_vhf_asr.train --train ... --eval ... --wer-eval ... --output runs/x
python -m marine_vhf_asr.evaluate --data vhf8k:test@400 martts:test --adapter <adapter>
python -m marine_vhf_asr.harbor --ais-root data/ais --regions new_york --dialogues 300 --out ...
hf upload mszarski/marine-vhf-asr-code marine_vhf_asr marine_vhf_asr --private --exclude "tests/*" --exclude "__pycache__/*"
```

Run modules with `python -m` from the repo root; `data/`, `runs/` gitignored.
`.claude/hooks/session-start.sh` installs `requirements.txt` + pytest in remote
sessions. Full v6 training / Nemotron eval launch commands: `HANDOFF.md` §8.

## Architecture

- **Text**: `gazetteer.py` (ports, stations, `AIS_REGIONS`) -> `smcp.py`
  `DialogueGenerator` (turns written as spoken). `harbor.py` fills dialogues
  with real vessel names from AIS snapshots (`ais.py`) and records candidate
  lists per transmission: `moving_a` (class A, moved >1 kn, within 20 nm, last
  10 min) and `all`.
- **Audio**: `tts.py` backends (Kokoro, Piper, Chatterbox, Parler) ->
  `vhf_channel.py` `simulate_vhf` -> `synthesize.py` writes 16 kHz WAVs + JSONL
  manifest (`{"audio", "text", "slots", ...}`; real recordings use the same format).
- **Data loading**: `sources.py` turns specs into `datasets.Dataset`s (`audio,
  text, slots, context, decoys, source`). Spec syntax `name:split[#context][@max]`,
  e.g. `vhf8k:train#mix@2000`, `martts:train+clean`, `path/manifest.jsonl#moving_a`.
  Context modes: `#none`, `#supported[:N]`, `#decoys[:N]`, `#mix`, `#mixlong`,
  or a manifest `contexts` key.
- **Prompting**: `prompting.py` is shared by train and inference. Candidate
  names are spliced after the audio, before `[/INST]`; prompts are left-padded;
  rows without context get exactly the stock prompt; loss only on transcript
  tokens + EOS. Also holds runaway guards and optional two-pass `verify_names`.
- **Training / eval**: `train.py` (LoRA r=32/alpha=64 on LM, `--lora-audio`
  adds the audio encoder; pushes adapter before eval; `--resume` from Hub
  `last-checkpoint`), `evaluate.py` (WER per source, slot accuracy,
  decoy-copy / wrong-name rate, name CER via `metrics.py`; `--verify`).
- **HF Jobs**: `jobs/*.py` are `uv run` scripts with inline pinned deps;
  `jobs/nemotron/` is a Docker-image shell job (vLLM + NeMo Speech).

## Decisions (rules, with why)

- Voxtral-Mini-3B + LoRA: same audio type as ATC, fits one 24 GB GPU; a 30B
  zero-shot model (Nemotron) did not beat the fine-tuned 3B.
- Mix in real ATC radio (`jacktol/ATC-ASR-Dataset`) + ATCOSIM: the only real
  VHF acoustics available. MARTTS (CC-BY) is OK to use (user's decision).
- Transcripts follow ATC conventions so sources can mix: uppercase, no
  punctuation, digits spoken singly, isolated letters -> NATO words except
  `A`/`I`, `NINER` -> `NINE`; all in `normalize.py`. Score WER only after
  normalizing both sides (so our numbers aren't comparable to the ATC card's).
  Slot/name accuracy matters more than WER.
- Shuffle candidate lists (VHF-8K's own prompt puts the true name last: leak).
- Keep the speaker off many training lists (`DROP_SPEAKER=0.5`) and cap lists
  at `MAX_TRAIN_LIST=120`: short always-correct lists teach "answer is on the
  list" (21.5% wrong names); ~350-name lists made training 2x slower and timed
  out (v4). Fewer confidently wrong names beats a few points of name accuracy.
- Held-out evaluation: New York and Øresund regions are excluded entirely
  (`--exclude-names`), ~17% of test speakers are unknown vessels. Never train
  on a test port's own traffic (memorizes its fleet).
- Train and test on different TTS engines (Kokoro held-out voices alone
  overstated gains ~3x). MARTTS is Chatterbox audio, so Chatterbox isn't unseen;
  use Parler (Whisper large-v3 filter: >=80% non-name words recovered) or real audio.
- Two-pass name verification stays off by default (2x inference, -1-3 pts names).
- Kaggle "Marine Radio Chatter" (Broadcastify-scraped): internal research
  only; don't republish audio or build a product on it without checking terms.

## Gotchas

- **HF Jobs run code from the Hub, not this checkout**: re-upload the package
  to `mszarski/marine-vhf-asr-code` before every job or it runs stale code.
  `{DATA}` in job args becomes a snapshot of `DATA_REPO`. `ast.parse` job
  scripts before launching (a heredoc once broke a paid job).
- `hf jobs` timeouts look like `90m`/`5h` ("1h30m" invalid). `hf jobs wait`
  returns early: poll `hf jobs inspect`. Background shell tasks cap at 2 h, so
  re-arm watchers. `a10g-large`/`l4` stalled in the scheduler; `a10g-small` works.
- Training: `hub_strategy="checkpoint"`, push adapter before long eval,
  `--resume` after timeouts (a v4 checkpoint upload failed silently).
- transformers 5.x: no `warmup_ratio` (use `warmup_steps=0.03`); `bf16` must
  be `torch.cuda.is_available()` or CPU runs crash.
- Job dependency pins are deliberate: Chatterbox needs numpy 1.26,
  `pyarrow<18`, `datasets<4`, `setuptools<81` (own job file); Nemotron needs
  nvcc/crt/**nvvm** all pinned to CUDA 13.0, `VLLM_USE_FLASHINFER_SAMPLER=0`,
  and the plain (non-model-card) config on A100 (card kernels need sm_90).
  Nemotron returns empty output on many clips without context terms.
- Runaway guards (`max_new_tokens_for`, `trim_runaway`, `collapse_loops`)
  exist because of real loops ("37000...", "VTS VTS VTS"); they never alter references.
- `ais.snapshot` needs data sorted by `t` (`ais.load` sorts). Danish waters need
  `harbor.py --min-listed 5`. `ais.spoken_name` reads vowel-less 2-4 letter
  tokens as letters except `LETTER_ACRONYMS`. Verifier matching must be
  letter-insensitive ("H T M S" vs "HTMS").
- VHF-8K numbers are digits; ambiguous ones (3+ digit runs, times, coords) are
  dropped on load. Manifest `slots` come back as sparse structs; `sources.py` rebuilds them.
- Kokoro needs `espeak-ng` + Python <= 3.12 (TTS runs on HF Jobs). Parler is slow: shard it.
- Dev container: no GPU; network blocks `coast.noaa.gov` and `kaggle.com`, so
  pull AIS/Kaggle data in jobs (`ais_extract_job.py`, `kaggle_sample_job.py`;
  never download the full ~100 GB Kaggle set). Use `datasets.disable_caching()`
  locally (cache once filled 15 GB). Hub downloads of many small files: `max_workers=32`.
  `hf upload-large-folder` uploads to repo root unless a path is set.
- Kaggle WhisperX transcripts are machine-made ("thank you" hallucinations);
  real traffic is casual, not SMCP. Don't trust tiny eval samples (ATC 16 clips
  35.7% vs 300 clips 69.1%). Kill by PID, not `pkill -f` (it killed its shell).

## Infrastructure pointers

All HF repos are under `mszarski` and private.
- Code mirror for jobs: `marine-vhf-asr-code`. Adapters:
  `voxtral-3B-marine-vhf` (v1), `-context` (v2), `-v3` ... `-v6`.
- Datasets: `marine-vhf-ais` (AIS parquet, 9 regions), `marine-vhf-ais-synth`
  (current train/test audio, exclude/unknown name lists, `diagnostics/`),
  `marine-radio-chatter-sample` (Kaggle sample). Older: `marine-vhf-ais-nyc`,
  `marine-vhf-nyc-synth`.
- Trackio Space `mszarski/trackio`, project `marine-vhf-asr`.
- Flavors: `a100-large` ($2.50/h) train/eval, `a10g-small` TTS/cheap evals,
  `cpu-upgrade` data pulls. Eval outputs: `OUT_REPO` `diagnostics/<NAME>.json`.
- External data (VHF-8K, MARTTS, ATC, ATCOSIM, NOAA / Danish AIS, Kaggle): links in `HANDOFF.md` section 8.
