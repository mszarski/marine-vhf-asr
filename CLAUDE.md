# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`marine_vhf_asr` builds a marine VHF radio transcription model: an SMCP
dialogue generator, a VHF channel simulator, data loaders, and LoRA
fine-tune/eval scripts for `mistralai/Voxtral-Mini-3B-2507`. It follows the
recipe of `pphilip/voxtral-3B-atc-transcribe` (the ATC equivalent).
`marine_vhf_asr/README.md` is the main document: datasets, usage, and the
results of every training run (v1-v6, Nemotron comparison). New results get
recorded there.

## Commands

```bash
pip install -r marine_vhf_asr/requirements.txt          # generation + metrics (numpy, scipy)
pip install -r marine_vhf_asr/requirements-train.txt    # GPU training / eval (torch, transformers, peft, datasets, ...)

pytest marine_vhf_asr/tests/                            # CPU only; torch/datasets/pandas/soundfile tests importorskip
python -m marine_vhf_asr.synthesize --out data/text --dialogues 50 --text-only
python -m marine_vhf_asr.train --train ... --eval ... --wer-eval ... --output runs/x
python -m marine_vhf_asr.evaluate --data vhf8k:test@400 martts:test --adapter <adapter>
python -m marine_vhf_asr.harbor --ais-root data/ais --regions new_york --dialogues 300 --out ...
```

Modules are run with `python -m` from the repo root (relative imports).
`data/` and `runs/` are gitignored.

## Architecture

- **Text**: `gazetteer.py` (ports, stations, vessel words) -> `smcp.py`
  `DialogueGenerator` (turns written as spoken, numbers as words). `harbor.py`
  instead fills dialogues with real vessel names from AIS snapshots (`ais.py`)
  and records candidate lists per transmission (`moving_a`, `all`).
- **Audio**: `tts.py` backends (Kokoro, Piper, Chatterbox, Parler) ->
  `vhf_channel.py` `simulate_vhf` -> `synthesize.py` writes 16 kHz WAVs + JSONL
  manifest (`{"audio", "text", "slots", ...}`; real recordings use the same format).
- **Data loading**: `sources.py` turns source specs into `datasets.Dataset`s with
  columns `audio, text, slots, context, decoys, source`.
  Spec syntax: `name:split[#context][@max]`, e.g. `vhf8k:train#mix@2000`,
  `martts:train+clean`, `path/manifest.jsonl#moving_a`. Context modes:
  `#none`, `#supported[:N]`, `#decoys[:N]`, `#mix`, `#mixlong`, or a manifest
  `contexts` key.
- **Prompting**: `prompting.py` is shared by train and inference. Candidate
  vessel names are spliced into Voxtral's transcription prompt after the audio;
  rows without context must get exactly the stock prompt. Also holds the
  runaway guards (`collapse_loops`, `trim_runaway`, `max_new_tokens_for`) and
  two-pass `verify_names`.
- **Training / eval**: `train.py` (LoRA on LM, `--lora-audio` adds the audio
  encoder; pushes adapter before eval; `--resume` from Hub `last-checkpoint`),
  `evaluate.py` (WER per source, slot accuracy, decoy-copy rate, name CER via
  `metrics.py`).
- **HF Jobs**: `jobs/*.py` are `uv run` scripts with inline, pinned
  dependencies; `jobs/nemotron/` is a vLLM + NeMo Speech shell job plus client.

## Conventions

- Transcripts follow the ATC dataset conventions so sources can be mixed:
  uppercase, no punctuation, digits spoken one by one (`16` -> `ONE SIX`;
  `numbers="cardinal"` for sources read as words), isolated letters -> NATO
  words except `A`/`I`, `PAN PAN`, `SECURITE`, `NINER` -> `NINE`. All of this
  lives in `normalize.py`; targets are `normalize_transcript(text)`.
- Score WER only after normalizing both sides; slot accuracy (vessel names,
  channels, positions, MMSI) matters more than WER for safety.
- Commit messages are prefixed `marine_vhf_asr: ...`.

## Gotchas

- **HF Jobs run code from the Hub, not this checkout.** Jobs
  `snapshot_download(CODE_REPO)` and import from it, so after editing the
  package, re-run `hf upload <user>/marine-vhf-asr-code marine_vhf_asr marine_vhf_asr --private --exclude "tests/*" --exclude "__pycache__/*"`
  before launching a job. `{DATA}` in job args is replaced with a snapshot of `DATA_REPO`.
- Job dependency pins are deliberate (e.g. Chatterbox needs numpy 1.26 with
  pinned pyarrow/datasets; Nemotron pins nvcc/nvvm to the CUDA runtime). Don't
  loosen them casually; Chatterbox and Parler have their own job files for this reason.
- Data-leak hygiene for honest test sets: MARTTS scenarios 9-10 are the test
  split (a/b/c variants share scripts); NY and Oresund are held-out regions with
  their vessel names excluded via `--exclude-names`; test audio uses held-out
  Kokoro voices (`_kv`) or engines unseen in training. MARTTS itself was made
  with Chatterbox, so Chatterbox is not a clean unseen engine; Parler is.
- Synthetic-speech results overstate real-world accuracy; the model learns TTS
  pronunciations. Prefer cross-engine test numbers when judging changes.
- `sources.py` constants `MAX_TRAIN_LIST=120` and `DROP_SPEAKER=0.5` came from
  the v4 failure (slow training, over-copying decoys); change them knowingly.
- VHF-8K transcripts with ambiguous numbers (3+ digit runs, coordinates) are
  dropped on load because the TTS reading is unknown.
- Kokoro needs `espeak-ng` and Python <= 3.12, so TTS normally runs on HF Jobs (`jobs/harbor_tts_job.py`).
