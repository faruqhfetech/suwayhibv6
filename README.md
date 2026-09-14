# Suwayhib v6

Offline Qur'an-recitation practice companion for under-resourced Qur'an
schools. A learner recites a known verse; words turn green as they're
confirmed. It never marks a word "wrong" -- on an error it just stalls,
waiting for the word to be repeated correctly.

Technically: streaming forced alignment with rejection. A frozen
self-supervised encoder (wav2vec2-XLS-R-300M) feeds a small trained phone
head, whose posteriors are decoded against the *known* expected phone
sequence by a streaming CTC lattice. The same trained head is also entered
in the IQRA 2026 shared task (phoneme recognition).

Read `suwayhibv6.txt` (a full paper draft covering architecture, results,
and the iterations that failed along the way) for the real detail.
`docs/handover.txt` carries the v1-v5 project history and lessons learned.

## Layout

```
suwayhib/            the package
  core/               phones.py, reftext.py, textscore.py, bootstrap.py, model.py
  pipeline/           extract.py, align_phonemes.py, harness.py, score.py, decode.py, demo.py
  train/              train.py, train_stream.py
  tools/              probe.py, injector.py (archival -- only needed to rebuild testset/)

docs/                 handover.txt (current), archive/ (superseded v4/v5 docs), logs/ (captured run output)
suwayhibv6.txt        the paper draft -- current, authoritative project writeup

manifest/             verified word-segment manifest + IQRA alias config
texts/                Qur'an text variants (quran-simple-plain.txt is the one in use)
reference_library/    professional-reciter audio + forced-alignment timings (gitignored)
recs/                 real (non-synthetic) learner recordings for eval (gitignored)
testset/              fixed synthetic eval benchmark -- do not regenerate casually (gitignored)
runs/                 trained checkpoints (gitignored)
cache/                extracted-feature caches, regeneratable (gitignored)

leaderboard.csv, submission.csv    IQRA 2026 shared-task leaderboard + current submission
```

## Running things

Everything is a real Python package now (`suwayhib/__init__.py` etc.), so
invoke modules with `-m` from the repo root, e.g.:

```bash
python -m suwayhib.pipeline.decode selftest
python -m suwayhib.pipeline.score testset --ckpt runs/head/best.pt --by-error-type
python -m suwayhib.pipeline.demo --file Recording_4.m4a
python -m suwayhib.train.train run --cache cache/feats --epochs 20
```

See `requirements.txt` for Python dependencies; `ffmpeg`/`ffplay` must also
be on `PATH` (used for audio decode/playback via subprocess).
