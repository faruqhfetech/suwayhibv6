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

def _decode_audio(a):
    """datasets' audio column has changed representation across versions
    -- dict in older releases, torchcodec AudioDecoder in newer. Handle
    both, fail loudly on anything else rather than guessing, since a
    silently mis-decoded waveform yields features that look entirely
    plausible and are entirely wrong."""
    if isinstance(a, dict):
        return np.asarray(a["array"], dtype=np.float32), int(
            a.get("sampling_rate", SR))
    if hasattr(a, "get_all_samples"):
        s = a.get_all_samples()
        d = s.data
        if hasattr(d, "numpy"):
            d = d.numpy()
        d = np.asarray(d, dtype=np.float32)
        if d.ndim == 2:
            d = d.mean(axis=0)
        return d, int(s.sample_rate)
    raise SystemExit(f"unrecognised audio field type {type(a)!r}")


def stream_hf(dataset, split, label_col="phoneme_mis", max_hours=None,
              shuffle_buffer=2000, seed=0):
    """Yield {audio, phones, source} from a streamed HF dataset.

    label_col defaults to phoneme_mis -- what was ACTUALLY said. See the
    module docstring for why this is not phoneme_ref. Falls back to
    phoneme_ref when phoneme_mis is absent, which is correct for
    Iqra_train (native speech, the two are identical by construction).
    """
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split, streaming=True)
    if shuffle_buffer:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    budget = max_hours * 3600 if max_hours else None
    total = 0.0
    for r in ds:
        lab = r.get(label_col) or r.get("phoneme_ref")
        if not lab:
            continue
        wav, sr = _decode_audio(r["audio"])
        if sr != SR:
            raise SystemExit(f"{dataset}: sample rate {sr} != {SR}")
        dur = len(wav) / SR
        if budget and total + dur > budget:
            return
        total += dur
        yield {"audio": wav, "phones": lab.split(), "ends": None,
               "source": dataset}


def stream_local(manifest, lib, text_path, audio_cache=None, max_items=None):
    """Yield reference-library items WITH word-end frames.

    This is the only source of boundary supervision -- see the module
    docstring. Uses our own G2P (phones.py) rather than a shipped label
    column, because the reference library has no phoneme annotations of
    its own; it has verified word TIMINGS, which is exactly the thing
    the streamed data lacks.
    """
    rt = RT.RefText(text_path)
    cache = None
    if audio_cache and (Path(audio_cache) / "index.json").exists():
        cache = B.AudioCache(audio_cache)

    ayahs = B.group_ayahs(B.load_manifest(manifest))
    n = 0
    for (rec, surah, ayah), words in sorted(ayahs.items()):
        if max_items and n >= max_items:
            return
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
        wav = (cache.ayah(path) if cache
               else B.decode(Path(lib) / path))
        n += 1
        yield {"audio": wav, "phones": phones, "ends": ends,
               "source": f"ref/{rec}", "reciter": rec}


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

    return {"x": x, "lens": lens, "seq": torch.cat(seqs),
            "slens": torch.tensor(slens), "bnd": bnd, "bnd_mask": bnd_mask}


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

def build_streams(a, vocab, stage):
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
            lambda: stream_local(a.manifest, a.lib, a.text, a.audio_cache)))
        weights.append(a.local_ratio)

    return streams, weights


def prepare_item(item, vocab):
    """Attach integer label ids; drop items with OOV symbols."""
    ids = [vocab[p] for p in item["phones"] if p in vocab]
    if len(ids) != len(item["phones"]) or not ids:
        return None
    item["_ids"] = torch.tensor(ids, dtype=torch.long)
    return item


def save_ckpt(path, model, opt, scaler, step, vocab, cfg, extra=None):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step,
                "vocab": vocab, "config": cfg, **(extra or {})}, path)


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

    enc = E.Encoder(device=a.device, fp16=(device.type == "cuda"))
    model = M.PhoneHead(
        n_phones=n_phones, n_layers=len(a.layers), dim=a.dim,
        blocks=a.blocks, heads=a.heads, kernel=a.kernel,
        ff_mult=a.ff_mult, dropout=a.dropout, fusion=a.fusion).to(device)

    cfg = {"dim": a.dim, "blocks": a.blocks, "heads": a.heads,
           "kernel": a.kernel, "ff_mult": a.ff_mult, "fusion": a.fusion,
           "n_layers": len(a.layers), "layers": a.layers,
           "n_phones": n_phones}

    start_step = 0
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / "resume.pt"

    if a.init:
        ck = torch.load(a.init, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"initialised from {a.init} (step {ck.get('step', '?')})")
    if a.resume and resume_path.exists():
        ck = torch.load(resume_path, map_location=device,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_step = ck["step"]
        print(f"resumed from {resume_path} at step {start_step}")

    print(f"\nstage      : {stage}")
    print(f"vocab      : {n_phones} phones + blank")
    print(f"layers     : {a.layers}")
    print(f"parameters : {model.n_params()/1e6:.2f} M")
    print(f"labels     : {a.label_col}  "
          f"(phoneme_mis = what was ACTUALLY said; using phoneme_ref "
          f"here would make Iqra_TTS actively harmful)")
    print(f"boundary   : lambda {a.lam_bnd}, supervised only by local "
          f"items (local_ratio {a.local_ratio})")
    print(f"schedule   : tri-stage, peak {a.lr}, {a.total_steps} steps\n")

    streams, weights = build_streams(a, vocab, stage)
    src = interleave(streams, weights, seed=a.seed)
    src = (x for x in (prepare_item(i, vocab) for i in src) if x)
    batcher = BucketBatcher(src, a.batch, a.pool_batches, seed=a.seed)

    model.train()
    step = start_step
    t0 = time.time()
    run_ctc, run_bnd, nb = 0.0, 0.0, 0
    best = float("inf")
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
            loss = lc + a.lam_bnd * lb

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        scaler.step(opt)
        scaler.update()

        run_ctc += lc.item()
        run_bnd += float(lb.detach())
        nb += 1
        step += 1

        if step % a.log_every == 0:
            el = time.time() - t0
            print(f"step {step:7d}/{a.total_steps}  "
                  f"ctc {run_ctc/max(nb,1):7.4f}  "
                  f"bnd {run_bnd/max(nb,1):7.4f}  lr {lr:.2e}  "
                  f"{el/60:6.1f}m  {(step-start_step)/max(el,1):.2f} steps/s",
                  flush=True)
            run_ctc, run_bnd, nb = 0.0, 0.0, 0

        if step % a.ckpt_every == 0:
            save_ckpt(resume_path, model, opt, scaler, step, vocab, cfg)
            save_ckpt(out_dir / f"stage{stage}_last.pt", model, opt,
                      scaler, step, vocab, cfg)
            # free anything the last window left behind before the next
            # one allocates; on a memory-constrained box this was the
            # difference between finishing and an OOM kill
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  checkpointed at step {step}", flush=True)

    save_ckpt(resume_path, model, opt, scaler, step, vocab, cfg)
    save_ckpt(out_dir / f"stage{stage}_last.pt", model, opt, scaler, step,
              vocab, cfg)
    print(f"\nstopped at step {step} -> {out_dir}/stage{stage}_last.pt")
    print("NOTE: this trainer has no held-out val loop -- streaming makes")
    print("a clean split awkward and the honest evaluation for this")
    print("project lives elsewhere. Evaluate the checkpoint with:")
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
    p.add_argument("--manifest", default="manifest/manifest_clean.csv")
    p.add_argument("--lib", default="reference_library")
    p.add_argument("--text", default="texts/quran-simple-plain.txt")
    p.add_argument("--audio-cache", default="cache/audio")

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
