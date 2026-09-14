#!/usr/bin/env python3
"""
Suwayhib v6 -- streaming trainer over the full IqraEval corpus.

    python -m suwayhib.train.train_stream stage1 --out runs/stream --hours 200
    python -m suwayhib.train.train_stream stage2 --out runs/stream --init runs/stream/stage1_best.pt
    python -m suwayhib.train.train_stream resume --out runs/stream

WHY THIS EXISTS ALONGSIDE train.py
-----------------------------------
train.py reads pre-extracted features from extract.py's cache. That is
the right design when the same features are reused across many runs, but
it does not scale: caching 6 layers of XLS-R at 50Hz costs ~2.2 GB per
hour of audio, so the full corpus (~160h across Iqra_train + Iqra_TTS)
would be ~350 GB before a single epoch runs.

This trainer streams instead: pull audio from HF, run the frozen encoder
inline, keep only the layers the head consumes, train, discard. Nothing
touches disk. The cost is re-encoding every epoch (XLS-R forward is
~20-25x realtime on a 4 GB card, measured), which is real but bounded --
and on Kaggle's P100/T4 it is considerably faster.

TWO THINGS THE LITERATURE CHANGED ABOUT OUR APPROACH
------------------------------------------------------
1. WE WERE TRAINING ON THE WRONG LABELS.

   Every IqraEval training set ships BOTH phoneme_ref (canonical: what
   SHOULD have been said) and phoneme_mis (verbatim: what WAS said).
   train.py reads only phoneme_ref. For Iqra_train the two are identical
   -- native speakers, no errors -- so it never mattered. But it makes
   Iqra_TTS's ~80 hours of DELIBERATE mispronunciations actively
   harmful: the model would be taught to emit the correct sequence for
   audio containing an error, which is worse than not training on it.

   The MDD task, and the leaderboard metric, are defined on what was
   actually said. So: phoneme_mis, and Iqra_TTS becomes usable.

   This plausibly explains word_repetition stuck at 27.5% -> 29.4%
   across two training runs despite a 3x boundary-loss reweighting. The
   model had never once seen an error paired with an error label.

2. THE TWO-STAGE CURRICULUM IS THE MOST-VALIDATED FINDING IN THE
   CHALLENGE REPORT.

   Systems incorporating REAL mispronounced speech -- by targeted
   fine-tuning, label curation, or two-stage curriculum -- outperformed
   those relying only on synthetic augmentation, because authentic
   errors carry acoustic characteristics TTS cannot replicate even when
   the synthesis is conditioned on erroneous transcripts. The gap held
   across the whole leaderboard.

   stage1: Iqra_train + Iqra_TTS (breadth, ~160h)
   stage2: Iqra_Extra_IS26 (~2h of REAL human errors, gated dataset --
           request access early, approval is someone else's schedule)

DOES THIS SERVE BOTH THE PRODUCT AND THE LEADERBOARD?
-------------------------------------------------------
Yes, and the reason is worth stating because it is not obvious: the
split between the two use cases is at DECODE time, not training time.

The head is a frame-level phoneme posterior model. The leaderboard reads
those posteriors through a free decode ("what was said"). Suwayhib reads
the SAME posteriors through decode.py's constrained lattice ("does this
match the word we are expecting"). Training on phoneme_mis improves both
-- the leaderboard directly, and Suwayhib because a model that
faithfully transcribes an error gives the lattice better evidence to
stall on.

The one place they diverge is anti-phone modelling (extending the label
set with a #phone "mispronounced" variant per canonical phone, which the
literature reports beats a single catch-all reject symbol). That helps
leaderboard diagnosis but doubles the symbol set the decoder graph must
handle. Left as --anti-phone, off by default, deliberately unexplored
until the simpler thing is measured.

THE BOUNDARY HEAD PROBLEM, AND WHY LOCAL DATA IS STILL MIXED IN
-----------------------------------------------------------------
The boundary head predicts per-frame word-END positions and is trained
from manifest_clean.csv's verified word spans. It is the one part of the
model that sees timing, and raising its weight moved madd_lengthening
from 89.0% to 96.5%.

Streamed HF data has NO word boundaries. Training purely on the stream
would silently drop that supervision entirely -- the loss would still be
computed, on an all-zero mask, contributing nothing. So local reference
library items are interleaved into the stream at --local-ratio, purely
to keep boundary supervision alive. Items without boundaries contribute
CTC only; the mask handles it (see collate).

TRAINING RECIPE, FROM THE wav2vec2/CTC LITERATURE RATHER THAN GUESSED
----------------------------------------------------------------------
  - TRI-STAGE LR: warm up 10% of updates, hold constant 40%, decay 50%.
    This is the wav2vec2 fine-tuning schedule, used consistently across
    the CTC-on-frozen-SSL papers. Replaces train.py's OneCycleLR, which
    was a reasonable default but not the one this setting is tuned for.
  - HEAD-ONLY WARMUP: the first --warmup-head steps train only the
    output projection, to stabilise CTC alignment before the whole head
    starts moving. Standard practice; cheap insurance against the early
    collapse-to-blank that CTC is prone to.
  - LENGTH BUCKETING: sequences grouped by length before batching, so a
    1900-frame item is not padded alongside 250-frame ones. Standard,
    and our own length histogram (p50 258 frames, p99 906, max 1906)
    says the waste without it is severe.
  - SpecAugment: kept from train.py. wav2vec2 reports it "delays
    overfitting and significantly improves the final error rate", which
    matches what we measured.

KAGGLE
------
Sessions cap at ~9-12h, so this checkpoints every --ckpt-every steps and
`resume` restarts from the last one, including optimizer and scheduler
state. Losing a 9-hour session to a missing scheduler state is a
recoverable mistake to make once, not twice.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import bootstrap as B
from ..core import model as M
from ..core import phones as P
from ..core import reftext as RT
from ..pipeline import extract as E
from . import phase2 as PH2

SR = 16000
FRAMES_PER_SEC = 50


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

def build_vocab(extra_symbols=()):
    """Phone symbol -> index, 0 reserved for CTC blank.

    Built from phones.py's measured inventory unioned with whatever the
    streamed data actually contains. The union matters: our G2P and
    IqraEval's phonetiser agree to ~1.5% (measured over 2000 sentences)
    but not perfectly, and a symbol present in their labels but absent
    from our inventory would otherwise silently drop every item
    containing it.
    """
    base = set(P.iqra_inventory()) | set(extra_symbols)
    syms = sorted(base)
    return {s: i + 1 for i, s in enumerate(syms)}


def scan_symbols(dataset, split, n=2000, label_col="phoneme_mis"):
    """Collect the symbol set actually present in a streamed dataset.

    Cheap (text column only, no audio decode) and worth doing once
    rather than discovering an OOV mid-run.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split, streaming=True)
    seen = Counter()
    for i, r in enumerate(ds):
        if i >= n:
            break
        lab = r.get(label_col) or r.get("phoneme_ref") or ""
        seen.update(lab.split())
    return seen


# --------------------------------------------------------------------------
# streaming sources
# --------------------------------------------------------------------------

def stream_hf(dataset, split, label_col="phoneme_mis", max_hours=None,
              shuffle_buffer=2000, seed=0):
    """Yield {audio, phones, source} from a streamed HF dataset.

    label_col defaults to phoneme_mis -- what was ACTUALLY said. See the
    module docstring for why this is not phoneme_ref. Falls back to
    phoneme_ref when phoneme_mis is absent, which is correct for
    Iqra_train (native speech, the two are identical by construction).

    Audio is decoded via extract.py's _decode_audio_field (ffmpeg-based),
    NOT by `datasets` itself -- see that function's docstring: `datasets`'
    own torchcodec-based auto-decode crashes hard on a torch/torchcodec
    version mismatch that is not exotic (it is what plain `pip install`
    produced on this project). cast_column(..., decode=False) hands back
    raw bytes instead of asking `datasets` to decode them.
    """
    from datasets import Audio, load_dataset
    ds = load_dataset(dataset, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    if shuffle_buffer:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    budget = max_hours * 3600 if max_hours else None
    total = 0.0
    for r in ds:
        lab = r.get(label_col) or r.get("phoneme_ref")
        if not lab:
            continue
        wav, sr = E._decode_audio_field(r["audio"])
        if sr != SR:
            raise SystemExit(f"{dataset}: sample rate {sr} != {SR}")
        dur = len(wav) / SR
        if budget and total + dur > budget:
            return
        total += dur
        item = {"audio": wav, "phones": lab.split(), "ends": None,
                "source": dataset}
        # Phase 2 (train/phase2.py) supervision: phoneme_ref is the
        # CANONICAL target regardless of which column trains the CTC
        # head above. Where it equals the CTC target, this audio
        # genuinely supports it (verify_label=1). Where it differs
        # (Iqra_TTS's deliberate mispronunciations, and real errors in
        # Iqra_Extra_IS26), the audio does NOT genuinely support the
        # canonical sequence it deviates from (verify_label=0) -- a
        # free negative example for the SAME accept/reject question
        # decode.py asks at inference, requiring no extra data.
        ref = r.get("phoneme_ref")
        if ref:
            item["verify_phones"] = ref.split()
            item["verify_label"] = 0 if ref != lab else 1
        yield item


def _fetch_everyayah(path, cache_dir=None, timeout=30):
    """Fetch one ayah's audio directly from everyayah.com's own public
    CDN, instead of reading reference_library/'s local copy.

    Why this is safe to rely on: reference_library/ATTRIBUTION.txt
    already documents that its audio comes from everyayah.com and its
    word timings come separately from cpfair/quran-align's own
    published release (18MB, tracked in git under
    reference_library/timings/ -- NOT something this project computed).
    The only thing that was ever local-only was the ~12GB of audio
    BYTES themselves, and everyayah.com serves those same files
    publicly. Verified directly (HTTP 200, BunnyCDN-backed, CORS-open)
    for all 8 reciters this project's manifest uses, at exactly the
    {reciter}/{surah:03d}{ayah:03d}.mp3 URL shape below -- this means a
    Kaggle run never needs reference_library/'s audio uploaded at all,
    only the already-tracked manifest + timings.

    `path` is a manifest audio_path like "Alafasy_128kbps/078001.wav"
    -- only the reciter folder and the surah+ayah digits are taken from
    it; everyayah.com itself serves .mp3, regardless of whatever
    extension the local ingested copy ended up with.

    cache_dir, if given, saves each fetched file once. stream_local is
    wrapped in cycle() for a multi-hour run, so without this every
    pass through the ~4,400 local items would re-fetch every file from
    scratch.
    """
    import urllib.request
    reciter = Path(path).parent.name
    fname = f"{Path(path).stem}.mp3"
    if cache_dir:
        cached = Path(cache_dir) / reciter / fname
        if cached.exists():
            wav, _sr = E._decode_audio_field({"bytes": cached.read_bytes()})
            return wav
    url = f"https://everyayah.com/data/{reciter}/{fname}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
    if cache_dir:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    wav, _sr = E._decode_audio_field({"bytes": data})
    return wav


def local_ayah_keys(manifest):
    """-> sorted list of (reciter, surah, ayah) keys in the local
    manifest, in the SAME deterministic order stream_local iterates
    them. This is the basis for a fixed, leakage-free train/validation
    split of the local reference library -- see run_training's
    val_keys, which takes the LAST --val-items of this list."""
    return sorted(B.group_ayahs(B.load_manifest(manifest)).keys())


def stream_local(manifest, lib, text_path, audio_cache=None, max_items=None,
                 everyayah_cache=None, include_keys=None, exclude_keys=None):
    """Yield reference-library items WITH word-end frames.

    This is the only source of boundary supervision -- see the module
    docstring. Uses our own G2P (phones.py) rather than a shipped label
    column, because the reference library has no phoneme annotations of
    its own; it has verified word TIMINGS, which is exactly the thing
    the streamed data lacks.

    Audio resolution order per item: audio_cache (a prebuilt local
    AudioCache, if given) -> a local file under `lib` (the normal path
    when reference_library/'s audio happens to be present) -> a live
    fetch from everyayah.com (see _fetch_everyayah), cached to
    `everyayah_cache` so repeat passes don't re-fetch. This means the
    exact same call works unmodified whether reference_library/'s
    audio is present locally (a dev machine that has it) or absent
    (a fresh Kaggle clone that only has the tracked manifest +
    timings) -- nothing needs to know which situation it's in.

    include_keys/exclude_keys: mutually exclusive (reciter, surah,
    ayah) filters, used to carve the local library into a training
    stream (exclude_keys=val_keys, so validation items never train the
    model) and a fixed validation set (include_keys=val_keys, called
    once to materialise it -- see run_training).
    """
    rt = RT.RefText(text_path)
    cache = None
    if audio_cache and (Path(audio_cache) / "index.json").exists():
        cache = B.AudioCache(audio_cache)

    ayahs = B.group_ayahs(B.load_manifest(manifest))
    n = 0
    for key, words in sorted(ayahs.items()):
        if include_keys is not None and key not in include_keys:
            continue
        if exclude_keys is not None and key in exclude_keys:
            continue
        if max_items and n >= max_items:
            return
        rec, surah, ayah = key
        toks = rt.words(surah, ayah)
        if len(toks) != len(words):
            continue
        phones, ends = [], []
        for w, tok in zip(words, toks):
            ph = P.word_to_phones_iqra(tok)
            if not ph:
                phones = []
                break
            phones.extend(ph)
            ends.append(w["end_ms"] * FRAMES_PER_SEC // 1000)
        if not phones:
            continue
        path = words[0]["audio_path"]
        local_path = Path(lib) / path
        if cache:
            wav = cache.ayah(path)
        elif local_path.exists():
            wav = B.decode(local_path)
        else:
            wav = _fetch_everyayah(path, cache_dir=everyayah_cache)
        n += 1
        # Professional reciters, verified word segments: this audio
        # always genuinely supports its own canonical G2P sequence, so
        # it is a (less informative but free) verify_label=1 example --
        # see stream_hf's comment for what these fields feed into.
        yield {"audio": wav, "phones": phones, "ends": ends,
               "source": f"ref/{rec}", "reciter": rec,
               "verify_phones": phones, "verify_label": 1}


def cycle(make_stream):
    """Restart a finite stream forever.

    FOUND BY TESTING interleave(): weights only hold while BOTH streams
    are live. Once one exhausts it is dropped and the remainder is drawn
    entirely from the survivor -- so a 0.15 local ratio measured 0.50 in
    a two-stream test where both were the same length.

    That matters here specifically: the local reference library is ~4,400
    items and will exhaust within the first fraction of a 160-hour run,
    after which boundary supervision would silently vanish for the entire
    remainder while the printed --local-ratio still claimed 0.15. Cycling
    the finite source keeps the ratio honest for the whole run.

    Takes a FACTORY, not a generator, because a generator cannot be
    restarted once consumed.
    """
    while True:
        n = 0
        for item in make_stream():
            n += 1
            yield item
        if n == 0:
            # nothing to cycle; stop rather than spin forever
            return


def interleave(streams, weights, seed=0):
    """Round-robin with weights, so a 2-hour real-error set is not
    drowned by 160 hours of everything else. Exhausted streams drop out
    rather than ending the whole mixture -- but see cycle() above: a
    finite stream that should hold a steady share must be wrapped, or
    its share silently goes to zero once it runs out."""
    rng = np.random.default_rng(seed)
    live = [(s, w) for s, w in zip(streams, weights)]
    while live:
        p = np.array([w for _, w in live], dtype=float)
        p /= p.sum()
        i = int(rng.choice(len(live), p=p))
        try:
            yield next(live[i][0])
        except StopIteration:
            live.pop(i)


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------

class BucketBatcher:
    """Group by length before batching.

    Padding to the longest item in a randomly-formed batch is wasteful
    when lengths span 40 to 1906 frames (measured on our own cache: p50
    258, p95 563, p99 906). Grouping by length is standard in the
    wav2vec2 line of work for exactly this reason. A pool is filled,
    sorted, cut into batches, shuffled, and emitted -- so batches are
    length-homogeneous but their ORDER is still random, which matters
    for optimisation.
    """

    def __init__(self, source, batch_size, pool_batches=24, seed=0):
        self.source = source
        self.bs = batch_size
        self.pool = batch_size * pool_batches
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        buf = []
        for item in self.source:
            buf.append(item)
            if len(buf) >= self.pool:
                yield from self._flush(buf)
                buf = []
        if len(buf) >= self.bs:
            yield from self._flush(buf)

    def _flush(self, buf):
        buf.sort(key=lambda d: len(d["audio"]))
        batches = [buf[i:i + self.bs] for i in range(0, len(buf), self.bs)]
        batches = [b for b in batches if len(b) == self.bs]
        self.rng.shuffle(batches)
        yield from batches


def encode_batch(enc, batch, layers, device, max_frames):
    """Frozen-encoder forward + collate into padded tensors."""
    waves = [b["audio"] for b in batch]
    feats = enc(waves, layers)                    # list of (T, L, D) fp16

    T = min(max(f.shape[0] for f in feats), max_frames)
    B = len(feats)
    L, D = feats[0].shape[1], feats[0].shape[2]
    x = torch.zeros(B, T, L, D)
    lens = torch.zeros(B, dtype=torch.long)
    bnd = torch.zeros(B, T)
    bnd_mask = torch.zeros(B, T)
    seqs, slens = [], []

    verify_ids, verify_labels, verify_present = [], [], []

    for i, (f, item) in enumerate(zip(feats, batch)):
        t = min(f.shape[0], T)
        x[i, :t] = torch.from_numpy(f[:t].astype(np.float32))
        lens[i] = t
        seqs.append(item["_ids"])
        slens.append(len(item["_ids"]))
        if item["ends"] is not None:
            # only items WITH real boundary labels contribute to the
            # boundary loss; everything streamed from HF has ends=None
            # and is masked out entirely rather than being taught that
            # "no boundary anywhere" is the truth
            bnd_mask[i, :t] = 1.0
            for e in item["ends"]:
                if 0 <= e < t:
                    bnd[i, e] = 1.0
        vids = item.get("_verify_ids")
        if vids is not None:
            # position INDEX into this batch, not a separate counter --
            # phase2.verification_loss needs to read the matching row
            # of the model's OWN logits later, from THIS batch's x/lens.
            verify_present.append(i)
            verify_ids.append(vids)
            verify_labels.append(item.get("verify_label", 1))

    return {"x": x, "lens": lens, "seq": torch.cat(seqs),
            "slens": torch.tensor(slens), "bnd": bnd, "bnd_mask": bnd_mask,
            "verify_present": verify_present, "verify_ids": verify_ids,
            "verify_labels": verify_labels}


# --------------------------------------------------------------------------
# losses / augmentation (shared with train.py's semantics)
# --------------------------------------------------------------------------

def ctc_loss(logits, lens, seq, slens):
    """zero_infinity is correct HERE (training) and wrong for scoring --
    it maps an impossible alignment to loss 0, which read as a score is
    the best possible value. Never reuse this to score."""
    lp = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    return F.ctc_loss(lp, seq, lens, slens, blank=0, reduction="mean",
                      zero_infinity=True)


def boundary_loss(pred, target, mask, pos_weight):
    if mask.sum() == 0:
        return torch.zeros((), device=pred.device)
    l = F.binary_cross_entropy_with_logits(
        pred.float(), target, reduction="none",
        pos_weight=torch.tensor(pos_weight, device=pred.device))
    return (l * mask).sum() / mask.sum().clamp(min=1)


def compute_val_loss(model, enc, val_items, layers, device, max_frames,
                     lam_bnd, pos_weight, no_boundary):
    """CTC + boundary loss (mean, same formula as training) over a
    FIXED, held-out slice of local reference-library items that never
    enter the training stream (see run_training's val_keys / build_
    streams' exclude_keys). This is the "best" tracking signal this
    trainer previously had none of -- deliberately narrow (local
    reference-library items only, since that's the one source with
    real word-level ground truth available to hold out at all; the
    streamed HF sources have no fixed, re-drawable validation slice by
    construction) rather than absent.

    Runs the WHOLE val_items list as one batch -- val_items is small by
    design (--val-items, default 64) specifically so this is cheap
    enough to run at every checkpoint interval without meaningfully
    slowing the run down.
    """
    if not val_items:
        return None
    model.eval()
    with torch.no_grad():
        b = encode_batch(enc, val_items, layers, device, max_frames)
        x = b["x"].to(device)
        out = model(x, chunk=None, left_chunks=-1)  # full context, matches
        # sample_chunk(training=False, ...)'s own (None, -1) return
        lc = ctc_loss(out["logits"], b["lens"], b["seq"], b["slens"])
        lb = torch.zeros((), device=device)
        if not no_boundary and out.get("boundary") is not None:
            lb = boundary_loss(out["boundary"], b["bnd"].to(device),
                               b["bnd_mask"].to(device), pos_weight)
        val = float(lc.item() + lam_bnd * float(lb.item()))
    model.train()
    return val


def spec_augment(x, lens, bnd_mask, n_time=2, time_frac=0.05,
                 n_feat=2, feat_width=16, rng=None):
    """SpecAugment with one project-specific constraint.

    Time-masked frames are REMOVED from bnd_mask. The boundary head is
    being weighted heavily precisely to sharpen word-end timing; masking
    erases that timing evidence. Without this, a masked span containing
    a true word end would teach the model to predict a boundary from
    audio that is no longer there -- label noise aimed at the one head
    we are trying to strengthen. Frequency masking removes channels, not
    time, so it needs no such treatment.
    """
    if rng is None:
        rng = np.random
    B, T, L, D = x.shape
    x = x.clone()
    bnd_mask = bnd_mask.clone()
    for i in range(B):
        n = int(lens[i].item())
        if n <= 2:
            continue
        max_t = max(1, int(n * time_frac))
        for _ in range(n_time):
            w = int(rng.randint(0, max_t + 1))
            if w <= 0:
                continue
            t0 = int(rng.randint(0, max(1, n - w)))
            x[i, t0:t0 + w] = 0.0
            bnd_mask[i, t0:t0 + w] = 0.0
        for _ in range(n_feat):
            w = int(rng.randint(0, feat_width + 1))
            if w <= 0:
                continue
            f0 = int(rng.randint(0, max(1, D - w)))
            x[i, :, :, f0:f0 + w] = 0.0
    return x, bnd_mask


def tristage_lr(step, total, peak, warm=0.10, hold=0.40, final_scale=0.05):
    """wav2vec2's tri-state schedule: warm up 10%, hold 40%, decay 50%.

    Used consistently across the CTC-on-frozen-SSL literature. Replaces
    OneCycleLR, which is a fine general default but is not the schedule
    this particular setting was tuned with -- and OneCycle's terminal
    near-zero LR is a poor fit for a run that may be interrupted and
    resumed on a session-limited machine.
    """
    w, h = int(total * warm), int(total * (warm + hold))
    if step < w:
        return peak * step / max(w, 1)
    if step < h:
        return peak
    frac = (step - h) / max(total - h, 1)
    return peak * (1.0 - frac * (1.0 - final_scale))


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def build_streams(a, vocab, stage, val_keys=None):
    """-> (list of generators, list of weights)"""
    streams, weights = [], []

    if stage == 1:
        streams.append(stream_hf("IqraEval/Iqra_train", "train",
                                 max_hours=a.hours, seed=a.seed))
        weights.append(a.w_train)
        if a.tts:
            streams.append(stream_hf("IqraEval/Iqra_TTS", "train",
                                     max_hours=a.hours, seed=a.seed + 1))
            weights.append(a.w_tts)
    else:
        streams.append(stream_hf(a.stage2_dataset, a.stage2_split,
                                 max_hours=None, seed=a.seed))
        weights.append(1.0)
        if a.stage2_replay > 0:
            # a little stage-1 data mixed back in, to limit catastrophic
            # forgetting during a fine-tune on ~2 hours
            streams.append(stream_hf("IqraEval/Iqra_train", "train",
                                     max_hours=a.stage2_replay_hours,
                                     seed=a.seed + 2))
            weights.append(a.stage2_replay)

    if a.local_ratio > 0 and Path(a.manifest).exists():
        # WRAPPED IN cycle(): the local library is ~4,400 items and will
        # exhaust early in a 160-hour run. Without cycling, boundary
        # supervision would silently disappear for the entire remainder
        # while the banner still printed the configured ratio. See
        # cycle()'s docstring for the measurement that caught this.
        streams.append(cycle(
            lambda: stream_local(a.manifest, a.lib, a.text, a.audio_cache,
                                 everyayah_cache=a.everyayah_cache,
                                 exclude_keys=val_keys)))
        weights.append(a.local_ratio)

    return streams, weights


def prepare_item(item, vocab):
    """Attach integer label ids; drop items with OOV symbols.

    verify_phones (if present) is converted the same way, but an OOV
    there only drops the VERIFY signal for this item, not the item
    itself -- the ordinary CTC target already passed its own check
    above, and there is no reason to discard a perfectly good training
    example just because its (separate, optional) verification target
    happens to contain a rare symbol.
    """
    ids = [vocab[p] for p in item["phones"] if p in vocab]
    if len(ids) != len(item["phones"]) or not ids:
        return None
    item["_ids"] = torch.tensor(ids, dtype=torch.long)

    vp = item.get("verify_phones")
    item["_verify_ids"] = None
    if vp:
        vids = [vocab[p] for p in vp if p in vocab]
        if len(vids) == len(vp) and vids:
            item["_verify_ids"] = torch.tensor(vids, dtype=torch.long)
    return item


def save_ckpt(path, model, opt, scaler, step, vocab, cfg, extra=None):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step,
                "vocab": vocab, "config": cfg, **(extra or {})}, path)


def _checkpoint(a, stage, out_dir, resume_path, model, opt, scaler, step,
                vocab, cfg, verify_head, enc, val_prepared, device,
                best_val, final=False):
    """Compute val loss (if there's a held-out slice to compute it on),
    always overwrite resume.pt/stage{stage}_last.pt, and additionally
    write stage{stage}_best.pt whenever val loss improves -- this is
    the "save both last and best, auto-resume from last" pattern:
    -> best_val (unchanged if no val slice or no improvement this call,
    so the caller always has the correct running value to pass back in
    next time, across resumes too).
    """
    val = compute_val_loss(model, enc, val_prepared, a.layers, device,
                           a.max_frames, a.lam_bnd, a.pos_weight,
                           a.no_boundary)
    improved = val is not None and val < best_val
    if improved:
        best_val = val

    vh_state = {"best_val": best_val}
    if verify_head is not None:
        vh_state["verify_head"] = verify_head.state_dict()
    save_ckpt(resume_path, model, opt, scaler, step, vocab, cfg, vh_state)
    save_ckpt(out_dir / f"stage{stage}_last.pt", model, opt, scaler, step,
              vocab, cfg, vh_state)
    if improved:
        save_ckpt(out_dir / f"stage{stage}_best.pt", model, opt, scaler,
                  step, vocab, cfg, {**vh_state, "val": val})

    if not final:
        val_str = (f"  val {val:.4f} (best {best_val:.4f})"
                  if val is not None else "")
        print(f"  checkpointed at step {step}{val_str}", flush=True)
    return best_val


def run_training(a, stage):
    device = torch.device(a.device)

    extra = ()
    if a.scan_symbols:
        print("scanning label symbols in the stream...")
        seen = scan_symbols("IqraEval/Iqra_train", "train", n=a.scan_n)
        extra = tuple(seen)
        print(f"  {len(seen)} distinct symbols observed")
    vocab = build_vocab(extra)
    n_phones = len(vocab)

    enc = E.Encoder(device=a.device, fp16=(device.type == "cuda"),
                    max_layer=max(a.layers))
    model = M.PhoneHead(
        n_phones=n_phones, n_layers=len(a.layers), dim=a.dim,
        blocks=a.blocks, heads=a.heads, kernel=a.kernel,
        ff_mult=a.ff_mult, dropout=a.dropout, fusion=a.fusion).to(device)

    cfg = {"dim": a.dim, "blocks": a.blocks, "heads": a.heads,
           "kernel": a.kernel, "ff_mult": a.ff_mult, "fusion": a.fusion,
           "n_layers": len(a.layers), "layers": a.layers,
           "n_phones": n_phones}

    verify_head = PH2.VerificationHead().to(device) if a.verify_loss else None
    params = list(model.parameters())
    if verify_head is not None:
        params += list(verify_head.parameters())
    start_step = 0
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / "resume.pt"

    if a.init:
        ck = torch.load(a.init, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"initialised from {a.init} (step {ck.get('step', '?')})")
    best_val = float("inf")
    if a.resume and resume_path.exists():
        ck = torch.load(resume_path, map_location=device,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_step = ck["step"]
        # best_val MUST come back from the checkpoint, not reset to inf:
        # otherwise the very next checkpoint interval after a resume
        # would treat whatever val loss it sees as automatically "best"
        # and overwrite a genuinely better stage{stage}_best.pt from
        # before the interruption.
        best_val = ck.get("best_val", float("inf"))
        if verify_head is not None and ck.get("verify_head"):
            verify_head.load_state_dict(ck["verify_head"])
        print(f"resumed from {resume_path} at step {start_step} "
              f"(best_val so far: {best_val:.4f})")

    print(f"\nstage      : {stage}")
    print(f"vocab      : {n_phones} phones + blank")
    print(f"layers     : {a.layers}")
    print(f"parameters : {model.n_params()/1e6:.2f} M")
    print(f"labels     : {a.label_col}  "
          f"(phoneme_mis = what was ACTUALLY said; using phoneme_ref "
          f"here would make Iqra_TTS actively harmful)")
    print(f"boundary   : lambda {a.lam_bnd}, supervised only by local "
          f"items (local_ratio {a.local_ratio})")
    if verify_head is not None:
        print(f"verify     : ON, lambda {a.lam_verify} -- train/phase2.py's "
              f"decoding-aware loss, gradient flows into the phone head "
              f"itself (not a post-hoc classifier, see phase1.py for that)")
    else:
        print(f"verify     : off (--verify-loss to enable)")
    # VALIDATION SPLIT. The last a.val_items local-library keys (sorted,
    # deterministic) are held out of TRAINING entirely (exclude_keys
    # below) and materialised ONCE here as a fixed set never re-drawn
    # or reshuffled -- "best" must be measured against the SAME items
    # every time, or a lucky/unlucky draw could look like real
    # progress. Only possible when the local library is actually in
    # use; streamed HF sources have no such fixed slice to hold out.
    val_keys, val_prepared = None, []
    if a.local_ratio > 0 and Path(a.manifest).exists() and a.val_items > 0:
        keys = local_ayah_keys(a.manifest)
        val_keys = set(keys[-a.val_items:]) if keys else set()
        val_raw = stream_local(a.manifest, a.lib, a.text, a.audio_cache,
                               everyayah_cache=a.everyayah_cache,
                               include_keys=val_keys)
        val_prepared = [x for x in (prepare_item(i, vocab) for i in val_raw)
                        if x]
        print(f"validation : {len(val_prepared)} held-out local items "
              f"(never trained on)\n")

    streams, weights = build_streams(a, vocab, stage, val_keys=val_keys)
    src = interleave(streams, weights, seed=a.seed)
    src = (x for x in (prepare_item(i, vocab) for i in src) if x)
    batcher = BucketBatcher(src, a.batch, a.pool_batches, seed=a.seed)

    model.train()
    step = start_step
    t0 = time.time()
    run_ctc, run_bnd, run_verify, nb = 0.0, 0.0, 0.0, 0
    deadline = t0 + a.max_hours_wall * 3600 if a.max_hours_wall else None

    for batch in batcher:
        if step >= a.total_steps:
            break
        if deadline and time.time() > deadline:
            print(f"\nwall-clock budget reached at step {step}")
            break

        lr = tristage_lr(step, a.total_steps, a.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        b = encode_batch(enc, batch, a.layers, device, a.max_frames)
        bx, bnd_mask = b["x"], b["bnd_mask"]
        if a.specaug:
            bx, bnd_mask = spec_augment(
                bx, b["lens"], bnd_mask, n_time=a.spec_time_masks,
                time_frac=a.spec_time_frac, n_feat=a.spec_feat_masks,
                feat_width=a.spec_feat_width)

        x = bx.to(device, non_blocking=True)
        chunk, left = M.sample_chunk(True, a.dynamic, int(b["lens"].max()))
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x, chunk=chunk, left_chunks=left)
            lc = ctc_loss(out["logits"], b["lens"], b["seq"], b["slens"])
            lb = torch.zeros((), device=device)
            if not a.no_boundary and out.get("boundary") is not None:
                lb = boundary_loss(out["boundary"], b["bnd"].to(device),
                                   bnd_mask.to(device), a.pos_weight)
            lv = torch.zeros((), device=device)
            if verify_head is not None and b["verify_present"]:
                # Restrict to the batch rows that actually carry a
                # verify target -- everything else (an item whose
                # verify_phones had an OOV symbol, or no phoneme_ref at
                # all) sits out of this loss term entirely, same
                # principle as bnd_mask above but at whole-ROW rather
                # than per-frame granularity, since a target sequence
                # can't be partially supervised.
                idx = torch.tensor(b["verify_present"], device=device)
                sub_lp = F.log_softmax(
                    out["logits"][idx].float(), dim=-1).transpose(0, 1)
                sub_lens = b["lens"].to(device)[idx]
                vtgt = torch.cat(b["verify_ids"]).to(device)
                vtlen = torch.tensor([len(v) for v in b["verify_ids"]],
                                     device=device)
                vlabel = torch.tensor(b["verify_labels"], dtype=torch.float32,
                                      device=device)
                lv, _ = PH2.verification_loss(verify_head, sub_lp, vtgt,
                                              sub_lens, vtlen, vlabel)
            loss = lc + a.lam_bnd * lb + a.lam_verify * lv

        if not torch.isfinite(loss):
            step += 1
            continue

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        # HEAD-ONLY WARMUP: for the first --warmup-head steps, zero the
        # gradient of everything except the output projection. CTC on a
        # freshly-initialised head is prone to collapsing to all-blank
        # early; letting the classifier settle first is standard, cheap
        # insurance against losing the first hour of a long run.
        if step < a.warmup_head:
            for n_, p_ in model.named_parameters():
                if not n_.startswith("out."):
                    if p_.grad is not None:
                        p_.grad.zero_()
        clip_params = list(model.parameters())
        if verify_head is not None:
            clip_params += list(verify_head.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, a.clip)
        scaler.step(opt)
        scaler.update()

        run_ctc += lc.item()
        run_bnd += float(lb.detach())
        run_verify += float(lv.detach())
        nb += 1
        step += 1

        if step % a.log_every == 0:
            el = time.time() - t0
            print(f"step {step:7d}/{a.total_steps}  "
                  f"ctc {run_ctc/max(nb,1):7.4f}  "
                  f"bnd {run_bnd/max(nb,1):7.4f}  "
                  f"verify {run_verify/max(nb,1):7.4f}  lr {lr:.2e}  "
                  f"{el/60:6.1f}m  {(step-start_step)/max(el,1):.2f} steps/s",
                  flush=True)
            run_ctc, run_bnd, run_verify, nb = 0.0, 0.0, 0.0, 0

        if step % a.ckpt_every == 0:
            best_val = _checkpoint(a, stage, out_dir, resume_path, model,
                                   opt, scaler, step, vocab, cfg,
                                   verify_head, enc, val_prepared, device,
                                   best_val)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    best_val = _checkpoint(a, stage, out_dir, resume_path, model, opt,
                           scaler, step, vocab, cfg, verify_head, enc,
                           val_prepared, device, best_val, final=True)
    print(f"\nstopped at step {step} -> {out_dir}/stage{stage}_last.pt"
          f"{f' (best: {out_dir}/stage{stage}_best.pt, val {best_val:.4f})' if val_prepared else ''}")
    print("NOTE: val loss above is CTC+boundary on a small (--val-items)")
    print("held-out slice of the LOCAL reference library only -- streamed")
    print("HF sources have no fixed slice to hold out. Useful for picking")
    print("a checkpoint across a long run, but the honest, full evaluation")
    print("for this project still lives elsewhere:")
    print("  python -m suwayhib.pipeline.score testset --ckpt <ckpt> --by-error-type")
    print("  python -m suwayhib.pipeline.decode sweep --ckpt <ckpt> --source testset "
          "--split holdout")
    return 0


# --------------------------------------------------------------------------

def add_common(p):
    p.add_argument("--out", default="runs/stream")
    p.add_argument("--device", default="cuda")
    p.add_argument("--layers", type=int, nargs="+",
                   default=[14, 15, 16, 17, 18, 19])
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kernel", type=int, default=15)
    p.add_argument("--ff-mult", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fusion", default="weighted_sum")

    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--pool-batches", type=int, default=24)
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--total-steps", type=int, default=120000)
    p.add_argument("--warmup-head", type=int, default=500)
    p.add_argument("--dynamic", action="store_true", default=True)
    p.add_argument("--no-dynamic", dest="dynamic", action="store_false")

    p.add_argument("--lam-bnd", type=float, default=1.0)
    p.add_argument("--pos-weight", type=float, default=20.0)
    p.add_argument("--no-boundary", action="store_true")

    p.add_argument("--verify-loss", action="store_true",
                   help="train/phase2.py's decoding-aware verification "
                        "loss, off by default. Uses phoneme_ref vs "
                        "phoneme_mis pairs already in the stream (no "
                        "extra data) to teach the head to make correct "
                        "vs mispronounced audio SEPARABLE on the same "
                        "constrained-alignment score decode.py "
                        "thresholds at inference, not just to "
                        "transcribe accurately.")
    p.add_argument("--lam-verify", type=float, default=0.5,
                   help="weight on the verify loss when --verify-loss "
                        "is set. Unvalidated starting point -- this "
                        "project's priority order is a plain retrain "
                        "first, then re-measuring this weight, not "
                        "picking it in the abstract.")

    p.add_argument("--specaug", action="store_true", default=True)
    p.add_argument("--no-specaug", dest="specaug", action="store_false")
    p.add_argument("--spec-time-masks", type=int, default=2)
    p.add_argument("--spec-time-frac", type=float, default=0.05)
    p.add_argument("--spec-feat-masks", type=int, default=2)
    p.add_argument("--spec-feat-width", type=int, default=16)

    p.add_argument("--label-col", default="phoneme_mis")
    p.add_argument("--hours", type=float, default=None,
                   help="cap per streamed dataset; None = all of it")
    p.add_argument("--tts", action="store_true", default=True,
                   help="include Iqra_TTS (~80h of deliberate errors). "
                        "Only safe BECAUSE we train on phoneme_mis.")
    p.add_argument("--no-tts", dest="tts", action="store_false")
    p.add_argument("--w-train", type=float, default=1.0)
    p.add_argument("--w-tts", type=float, default=1.0)

    p.add_argument("--local-ratio", type=float, default=0.15,
                   help="share of batches drawn from the local reference "
                        "library. This is the ONLY source of word-boundary "
                        "supervision -- streamed HF data has none. Set 0 "
                        "to disable, which silently disables the boundary "
                        "head's learning signal too.")
    p.add_argument("--val-items", type=int, default=64,
                   help="local-library items held out of TRAINING "
                        "entirely and used as a fixed validation slice "
                        "at every --ckpt-every, so stage{N}_best.pt "
                        "means something -- see compute_val_loss. Set "
                        "0 to disable (only stage{N}_last.pt is saved, "
                        "as before this existed).")
    p.add_argument("--manifest", default="manifest/manifest_clean.csv")
    p.add_argument("--lib", default="reference_library")
    p.add_argument("--text", default="texts/quran-simple-plain.txt")
    p.add_argument("--audio-cache", default="cache/audio")
    p.add_argument("--everyayah-cache", default="cache/everyayah",
                   help="local cache dir for audio fetched live from "
                        "everyayah.com when reference_library/'s local "
                        "copy of a file is absent (e.g. on a fresh "
                        "Kaggle clone that never uploaded the ~12GB of "
                        "audio -- only the tracked manifest + timings "
                        "are needed). Set empty/None to disable caching "
                        "and re-fetch every time (not recommended: "
                        "stream_local repeats via cycle() for the "
                        "whole run).")

    p.add_argument("--scan-symbols", action="store_true",
                   help="scan the stream's label symbols before building "
                        "the vocab, so an OOV is found now rather than "
                        "mid-run")
    p.add_argument("--scan-n", type=int, default=2000)

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--max-hours-wall", type=float, default=None,
                   help="stop after this many wall-clock hours. Set it "
                        "below the Kaggle session limit so the final "
                        "checkpoint is written before the session dies.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=None)
    p.add_argument("--resume", action="store_true")


def main():
    ap = argparse.ArgumentParser(description="streaming trainer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("stage1", help="Iqra_train + Iqra_TTS")
    add_common(s1)
    s1.set_defaults(func=lambda a: run_training(a, 1))

    s2 = sub.add_parser("stage2", help="adapt on real mispronunciations")
    add_common(s2)
    s2.add_argument("--stage2-dataset", default="IqraEval/Iqra_Extra_IS26")
    s2.add_argument("--stage2-split", default="train")
    s2.add_argument("--stage2-replay", type=float, default=0.25,
                    help="weight of stage-1 data mixed back in, to limit "
                         "forgetting while fine-tuning on ~2 hours")
    s2.add_argument("--stage2-replay-hours", type=float, default=10.0)
    s2.set_defaults(func=lambda a: run_training(a, 2))

    rs = sub.add_parser("resume")
    add_common(rs)
    rs.set_defaults(func=lambda a: run_training(a, 1))

    a = ap.parse_args()
    if a.cmd == "resume":
        a.resume = True
    if a.cmd == "stage2":
        # stage2 defaults that only make sense there
        if a.total_steps == 120000:
            a.total_steps = 8000
        if a.lr == 3e-4:
            a.lr = 5e-5      # gentle: ~2h of data, do not overwrite stage1
    rc = a.func(a) or 0
    sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
