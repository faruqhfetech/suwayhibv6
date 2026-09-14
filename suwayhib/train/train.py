#!/usr/bin/env python3
"""
Suwayhib v6 -- train the phone head.

    python -m suwayhib.train.train labels --cache cache/feats --n 5
    python -m suwayhib.train.train run --cache cache/feats --epochs 20
    python -m suwayhib.train.train run --cache cache/feats,cache/feats_iqra \\
        --dim 128 --blocks 8 --ff-mult 2 --epochs 30

WHAT IS BEING TRAINED
---------------------
Only the head from model.py. The XLS-R encoder is frozen and already
cached to disk by extract.py, so an epoch is array slicing rather than a
forward pass through 300M parameters -- which is the whole reason for the
cache. v5 measured >40 min/epoch spawning a subprocess per word.

TWO LABEL SOURCES, TWO DIFFERENT ROLES
---------------------------------------
CTC does not need frame-level alignment. It needs the phone SEQUENCE per
utterance and marginalises over alignments itself. That is both simpler
and more honest than the probe's uniform split, which had to invent
within-word boundaries.

    reference_library  sequence (from phones.py) AND frame-level word
                       boundaries (from the verified manifest). Feeds
                       both the CTC loss and the boundary head.
    IqraEval           sequence only, taken from the dataset's own
                       phoneme_ref column -- THEIR labels, not ours, so
                       CTC targets here are ground truth rather than our
                       G2P's opinion. No boundaries, so these items
                       contribute nothing to the boundary loss.

Mixing them is deliberate: the reference library is 8 professional voices
(too few for speaker invariance, per v4a) while Iqra_train is thousands of
crowd voices with expert-vowelised transcripts. The frozen encoder already
supplies invariance; the mixture supplies label diversity.

THE BOUNDARY HEAD IS THE POINT
-------------------------------
v5's five decoders all had to SEARCH for word ends, because nothing in the
model ever reported one. Section 5 of the handover names this directly:
"BOUNDARY SUPERVISION -- so the model reports where the word ENDS rather
than the decoder searching for it." The manifest has verified ends for
every word in the reference library, so this supervision is free. The
graph decoder in Phase 3 consumes it.

Weighted BCE, because word-end frames are ~1 in 50 and an unweighted loss
would learn to predict "no boundary" everywhere and score 98%.

DYNAMIC CHUNK TRAINING HAPPENS HERE, NOT IN THE MODEL
------------------------------------------------------
model.py accepts a chunk size but never samples one. That is deliberate --
sampling is a training policy, not an architectural property. If this loop
forgets to call sample_chunk, the model trains full-context-only and will
degrade in streaming while every offline number looks fine. The run banner
prints the chunk policy for exactly that reason.
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import bootstrap as B
from ..core import model as M
from ..core import phones as P
from ..pipeline import extract as E

FRAMES_PER_SEC = 50


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------

def reference_labels(manifest, text_path, vocab):
    """-> {cache_key: {"seq": [...], "ends": [frame, ...], "n_words": k}}

    Sequence from our G2P; word-end frames from the verified manifest.
    """
    text, _ = P.load_words_for_g2p(text_path)
    ayahs = B.group_ayahs(B.load_manifest(manifest))

    out, skipped = {}, Counter()
    for (rec, surah, ayah), words in ayahs.items():
        key = "ref/" + words[0]["audio_path"].replace("/", "__").replace(
            ".wav", "")
        t = text.get((surah, ayah))
        if not t:
            skipped["no_text"] += 1
            continue
        toks = t.split()
        if len(toks) != len(words):
            skipped["word_count_mismatch"] += 1
            continue

        seq, ends, bad = [], [], False
        for w, tok in zip(words, toks):
            ph = P.word_to_phones_iqra(tok)
            if not ph:
                bad = True
                break
            unk = [p for p in ph if p not in vocab]
            if unk:
                skipped["oov"] += 1
                bad = True
                break
            seq.extend(vocab[p] for p in ph)
            ends.append(w["end_ms"] * FRAMES_PER_SEC // 1000)
        if bad or not seq:
            continue
        out[key] = {"seq": seq, "ends": ends, "n_words": len(words)}
    return out, skipped


def iqra_labels(meta_path, vocab):
    """-> {cache_key: {"seq": [...], "ends": None}}

    Uses the dataset's OWN phoneme_ref, stored by extract.py into
    meta.jsonl. Deliberately not re-derived with our G2P: on IqraEval data
    their labels are ground truth and ours are an approximation
    (align_phonemes.py measured ~1.5% disagreement, which is small but is
    not zero and would be pure label noise here).
    """
    out, skipped = {}, Counter()
    p = Path(meta_path)
    if not p.exists():
        return out, skipped
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        ref = r.get("phoneme_ref")
        if not ref:
            skipped["no_phoneme_ref"] += 1
            continue
        syms = ref.split()
        unk = [s for s in syms if s not in vocab]
        if unk:
            skipped["oov"] += 1
            continue
        out[r["key"]] = {"seq": [vocab[s] for s in syms], "ends": None,
                         "n_words": None}
    return out, skipped


def collect_oov(meta_path, vocab, limit=40):
    """Symbols in the data that our inventory lacks. Reported, not hidden:
    a silently dropped symbol class is a silently wrong model."""
    c = Counter()
    p = Path(meta_path)
    if not p.exists():
        return c
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        for s in (r.get("phoneme_ref") or "").split():
            if s not in vocab:
                c[s] += 1
    return c


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class Items(torch.utils.data.Dataset):
    def __init__(self, caches, labels, max_frames=1500):
        self.caches = caches
        self.keys = []
        for ci, c in enumerate(caches):
            for k in c.index:
                if k in labels:
                    lo, hi = c.index[k]
                    if hi - lo >= 8:
                        self.keys.append((ci, k))
        self.labels = labels
        self.max_frames = max_frames

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        ci, k = self.keys[i]
        x = self.caches[ci].get(k)                 # (T, L, D)
        lab = self.labels[k]
        T = x.shape[0]
        if T > self.max_frames:
            # Crop from the START. Cropping mid-utterance would leave the
            # CTC target describing audio that is no longer present, which
            # teaches the model that phones can vanish.
            x = x[:self.max_frames]
            T = self.max_frames
        ends = None
        if lab["ends"] is not None:
            ends = [e for e in lab["ends"] if e < T]
        return {"x": torch.from_numpy(x), "seq": torch.tensor(lab["seq"]),
                "ends": ends, "T": T, "key": k}


def collate(batch):
    B = len(batch)
    T = max(b["T"] for b in batch)
    L, D = batch[0]["x"].shape[1:]
    x = torch.zeros(B, T, L, D)
    lens = torch.zeros(B, dtype=torch.long)
    bnd = torch.zeros(B, T)
    bnd_mask = torch.zeros(B, T)
    seqs, slens = [], []
    for i, b in enumerate(batch):
        x[i, :b["T"]] = b["x"]
        lens[i] = b["T"]
        seqs.append(b["seq"])
        slens.append(len(b["seq"]))
        if b["ends"] is not None:
            bnd_mask[i, :b["T"]] = 1.0     # only supervised items count
            for e in b["ends"]:
                if 0 <= e < T:
                    bnd[i, e] = 1.0
    return {"x": x, "lens": lens, "seq": torch.cat(seqs),
            "slens": torch.tensor(slens), "bnd": bnd, "bnd_mask": bnd_mask,
            "keys": [b["key"] for b in batch]}


def spec_augment(x, lens, bnd_mask, n_time=2, time_frac=0.05,
                 n_feat=2, feat_width=16, rng=None):
    """SpecAugment on cached XLS-R features, with one project-specific
    adjustment that matters.

    x:        (B, T, L, D) cached features, modified IN PLACE on a clone
    lens:     (B,) true frame counts -- masking must respect padding, or
              it wastes its budget on zeros and under-regularises the
              real audio
    bnd_mask: (B, T) which frames count toward the boundary loss;
              time-masked frames are REMOVED from it -- see below

    THE SYNERGY CONSTRAINT, made explicit because these two changes
    actively fight each other otherwise:

    lam_bnd was just raised 0.3 -> 1.0 so the model attends harder to
    precise word-END timing, because the decoder's held-out breakdown
    showed every failure concentrated in duration/timing categories
    (word_repetition 72%, madd_lengthening 74%, madd_shortening 89%,
    while every segmental category sat at 100%). Time masking erases
    exactly that timing evidence. Applied naively, a masked span that
    happens to contain a true word end would be teaching the model to
    predict a boundary from audio that is no longer there -- pure label
    noise, aimed squarely at the one head we are trying to strengthen.

    So time-masked frames are zeroed out of bnd_mask: the boundary head
    is simply not asked about frames whose evidence was destroyed. CTC
    still sees the masked input and still has to produce the full label
    sequence, which is where the regularisation comes from. The two
    changes then pull in the same direction instead of opposing.

    Frequency masking has no such conflict -- it removes feature
    channels, not time -- so it is left unconditioned.

    Defaults are conservative on TIME (2 masks, 5% of length each) and
    ordinary on FREQUENCY. Standard SpecAugment recipes are far more
    aggressive on time, but they are tuned for ASR where nothing
    downstream depends on exact boundary timing. Ours does.
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
            # the whole point: do not grade the boundary head on frames
            # whose evidence we just deleted
            bnd_mask[i, t0:t0 + w] = 0.0

        for _ in range(n_feat):
            w = int(rng.randint(0, feat_width + 1))
            if w <= 0:
                continue
            f0 = int(rng.randint(0, max(1, D - w)))
            x[i, :, :, f0:f0 + w] = 0.0

    return x, bnd_mask


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def ctc_loss(logits, lens, seq, slens):
    """logits (B, T, C) -> scalar.

    zero_infinity=True IS CORRECT FOR TRAINING and wrong for scoring.
    Handover section 6 records this: it turns an impossible alignment
    (input shorter than target) into a loss of ZERO, which read as a score
    is the BEST POSSIBLE value and won a search in v5. Here it only stops
    a degenerate batch item from producing inf gradients. Never reuse this
    function to score.
    """
    lp = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)   # (T, B, C)
    return F.ctc_loss(lp, seq, lens, slens, blank=0, reduction="mean",
                      zero_infinity=True)


def boundary_loss(pred, target, mask, pos_weight):
    if mask.sum() == 0:
        return torch.zeros((), device=pred.device)
    l = F.binary_cross_entropy_with_logits(
        pred.float(), target, reduction="none",
        pos_weight=torch.tensor(pos_weight, device=pred.device))
    return (l * mask).sum() / mask.sum().clamp(min=1)


def evaluate(model, loader, device, n_batches=None):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(loader):
            if n_batches and i >= n_batches:
                break
            out = model(b["x"].to(device))
            l = ctc_loss(out["logits"], b["lens"], b["seq"], b["slens"])
            if torch.isfinite(l):
                tot += l.item()
                n += 1
    model.train()
    return tot / max(n, 1)


def cmd_run(a):

    # DERIVED from actual G2P output over the whole corpus, not from a
    # hand-maintained constant. The first training attempt used
    # build_vocab() and dropped 76% of the reference library as OOV
    # because 'l' -- the most frequent consonant in Arabic -- was missing
    # from PHONE_SET. Measuring the inventory removes that whole class of
    # silent failure.
    vocab = P.build_vocab_observed(a.text)
    n_phones = len(vocab)
    device = torch.device(a.device)

    caches = [E.FeatureCache(c) for c in a.cache.split(",")]
    labels = {}
    for c_path, c in zip(a.cache.split(","), caches):
        ref, sk = reference_labels(a.manifest, a.text, vocab)
        labels.update({k: v for k, v in ref.items() if k in c.index})
        iq, sk2 = iqra_labels(Path(c_path) / "meta.jsonl", vocab)
        labels.update({k: v for k, v in iq.items() if k in c.index})
        oov = collect_oov(Path(c_path) / "meta.jsonl", vocab)
        if oov:
            print(f"OOV symbols in {c_path}: {dict(oov.most_common(12))}")
            print("  Items containing these are DROPPED. If the counts are")
            print("  large, widen the vocabulary via build_vocab(extra=...)")
            print("  rather than losing the data.")

    ds = Items(caches, labels, a.max_frames)

    # RECITER-HELD-OUT SPLIT, replacing a random 90/10 over pooled items.
    #
    # The random split was a real methodological hole: testset/meta.json
    # declares holdout_reciters = [Husary, Minshawy] and every evaluation
    # in this project honours that, but TRAINING did not -- so those two
    # reciters' audio was in the training data, and every val number
    # reported so far was optimistic by an unknown amount. It also
    # allowed ayah-level leakage: the same ayah by a different reciter
    # could sit on both sides of the split.
    #
    # Keys are "ref/<Reciter>__<surah><ayah>" (reference library) or
    # "<dataset>/<id>" (IqraEval). Only the former carry a reciter;
    # IqraEval items are assigned by a STABLE digest of the key.
    # Python's builtin hash() is salted per process and would give a
    # different split on every run, so md5 is used deliberately.
    holdout_reciters = [r.strip() for r in a.holdout_reciters.split(",")
                        if r.strip()]
    tr_idx, va_idx, va_is_ref = [], [], []
    seen_reciters = Counter()
    for i, (ci, key) in enumerate(ds.keys):
        if key.startswith("ref/"):
            body = key[4:]
            reciter = body.split("__")[0] if "__" in body else ""
            seen_reciters[reciter] += 1
            is_val = any(h in reciter for h in holdout_reciters)
        else:
            h = hashlib.md5(key.encode("utf-8")).hexdigest()
            is_val = (int(h[:8], 16) % 10) == 0
        if is_val:
            va_idx.append(i)
            va_is_ref.append(key.startswith("ref/"))
        else:
            tr_idx.append(i)

    if not va_idx or not tr_idx:
        raise SystemExit(
            f"split produced train={len(tr_idx)} val={len(va_idx)} -- one "
            f"side is empty. Check --holdout-reciters against the reciter "
            f"names actually present: {sorted(seen_reciters)[:8]}")

    tr = torch.utils.data.Subset(ds, tr_idx)
    va = torch.utils.data.Subset(ds, va_idx)

    # FOUND WHILE THE FIRST v2 RUN WAS ALREADY IN PROGRESS: "val" is a
    # BLEND, and not an even one. Roughly 2/8 of the reference library
    # (Husary + Minshawy) lands in val, but so does ~10% of Iqra_train --
    # and Iqra_train supplies the large majority of every item count in
    # this project (18,241 of 22,622 total). Estimated from the printed
    # counts: ~63% of val is the Iqra random-holdout slice (easy: same
    # source and distribution as 83% of TRAIN), and only ~37% is the
    # actually hard reciter-generalisation test this split exists for.
    # "Best" selected on the blended loss is therefore majority-graded on
    # the easier slice, which can silently diverge from what is best for
    # the domain the product actually ships in: professional Qur'anic
    # recitation, i.e. reference_library's distribution, not Common
    # Voice-style read speech.
    #
    # Tracked here rather than fixed by re-splitting, because the
    # already-running v2 training should not be restarted for this --
    # see cmd_run below for how it is used (a second, reciter-only val
    # loss reported alongside the blended one, and a THIRD checkpoint
    # selected on it specifically).
    va_ref_idx = [va_idx[j] for j, r in enumerate(va_is_ref) if r]
    va_iqra_idx = [va_idx[j] for j, r in enumerate(va_is_ref) if not r]
    va_ref = (torch.utils.data.Subset(ds, va_ref_idx) if va_ref_idx
             else None)

    print(f"split      : held out {holdout_reciters}")
    print(f"             train {len(tr)}  val {len(va)}  "
          f"(ref-only {len(va_ref_idx)} / {len(va_ref_idx)/max(len(va),1):.0%}"
          f", iqra-random {len(va_iqra_idx)} / "
          f"{len(va_iqra_idx)/max(len(va),1):.0%})")
    if len(va_iqra_idx) > 2 * max(len(va_ref_idx), 1):
        print("             WARNING: val is majority Iqra random-holdout,")
        print("             not the reciter test -- 'best' below is a")
        print("             blend; see val_ref for the harder number.")
    # WORKER / MEMORY POLICY. Three OOM kills happened on this machine
    # during earlier runs; every one traced to memmap pages accumulating
    # faster than they were reclaimed, not to model memory. The rules
    # that came out of it, kept deliberately conservative:
    #
    #   persistent_workers OFF everywhere. It saves re-opening a 44 GB
    #   memmap each epoch, but it also means those handles and their
    #   page-cache footprint are NEVER released for the whole run.
    #   Respawning each epoch costs seconds and guarantees the pages get
    #   dropped -- the right trade on a memory-constrained box.
    #
    #   pin_memory only when actually on CUDA. Pinned memory is
    #   page-locked and cannot be swapped out, which is precisely the
    #   wrong property here if it is not buying overlapped transfer.
    pin = (device.type == "cuda")
    dl_tr = torch.utils.data.DataLoader(
        tr, batch_size=a.batch, shuffle=True, collate_fn=collate,
        num_workers=a.workers, drop_last=True,
        persistent_workers=False, pin_memory=pin)
    # BUG FOUND MID-RUN: persistent_workers=True on the VAL loader meant
    # its workers stayed resident for the WHOLE run instead of tearing
    # down after each brief once-per-epoch use -- doubling concurrent
    # worker count from 4 to 8 (confirmed via `ps aux`: two live PID
    # groups holding memmap pages simultaneously). Swap climbed to 3.5 GB
    # within 3 minutes of epoch 2 starting, faster than any prior run,
    # which was heading toward an EARLIER OOM than the epoch-6/7 kills
    # this was meant to fix. Validation runs at most --val-batches items,
    # so it gets no benefit from persistence or extra workers; num_workers
    # is dropped to 0 outright rather than just removing persistence, to
    # not pay worker-spawn cost for a loader used this briefly.
    # val loader: num_workers=0 deliberately. It runs at most
    # --val-batches items once per epoch, so workers are pure overhead;
    # when they were ALSO persistent they doubled the live worker count
    # (confirmed via ps aux: two live PID groups holding memmap pages),
    # and swap hit 3.5 GB three minutes into epoch 2.
    dl_va = torch.utils.data.DataLoader(
        va, batch_size=a.batch, shuffle=False, collate_fn=collate,
        num_workers=0, pin_memory=pin)
    dl_va_ref = (torch.utils.data.DataLoader(
        va_ref, batch_size=a.batch, shuffle=False, collate_fn=collate,
        num_workers=0, pin_memory=pin) if va_ref is not None else None)

    model = M.PhoneHead(
        n_phones=n_phones, n_layers=len(caches[0].layers), dim=a.dim,
        blocks=a.blocks, heads=a.heads, kernel=a.kernel, ff_mult=a.ff_mult,
        dropout=a.dropout, fusion=a.fusion).to(device)

    print(f"\nvocab      : {n_phones} phones + blank")
    print(f"items      : {len(ds)}  (train {len(tr)}, val {len(va)})")
    print(f"layers     : {caches[0].layers}")
    print(f"parameters : {model.n_params()/1e6:.2f} M")
    print(f"chunk policy: {'DYNAMIC (60% chunked / 40% full)' if a.dynamic else 'FULL CONTEXT ONLY -- streaming will degrade'}")
    print(f"boundary   : {'on' if not a.no_boundary else 'off'}"
          f"  lambda {a.lam_bnd}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)
    steps = a.epochs * max(1, len(dl_tr))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_ref = [float("inf")]     # list so it is mutable inside the loop
    step = 0
    since_best = 0

    for ep in range(a.epochs):
        # Release anything left over from the previous epoch BEFORE
        # allocating this one's workers. gc first (drops Python-side
        # references to last epoch's batches, worker objects and the
        # DataLoader's internal iterator), then empty_cache (returns
        # torch's cached CUDA blocks to the driver). Neither frees
        # memory that is genuinely still referenced, so this cannot
        # break a live tensor -- it only stops dead objects from
        # sitting on memory across an epoch boundary, which on this
        # machine was the difference between finishing and an OOM kill.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        t0 = time.time()
        run_ctc, run_bnd, nb = 0.0, 0.0, 0
        for bi, b in enumerate(dl_tr):
            if a.max_batches and bi >= a.max_batches:
                break
            chunk, left = M.sample_chunk(True, a.dynamic, int(b["lens"].max()))
            bx, bnd_mask = b["x"], b["bnd_mask"]
            if a.specaug:
                # applied on CPU before the H2D copy: the masked tensor
                # is what gets transferred, so no bandwidth is spent
                # moving values that are about to be zeroed
                bx, bnd_mask = spec_augment(
                    bx, b["lens"], bnd_mask,
                    n_time=a.spec_time_masks, time_frac=a.spec_time_frac,
                    n_feat=a.spec_feat_masks, feat_width=a.spec_feat_width)
            x = bx.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x, chunk=chunk, left_chunks=left)
                lc = ctc_loss(out["logits"], b["lens"], b["seq"], b["slens"])
                lb = torch.zeros((), device=device)
                if not a.no_boundary and out.get("boundary") is not None:
                    lb = boundary_loss(out["boundary"], b["bnd"].to(device),
                                       bnd_mask.to(device), a.pos_weight)
                loss = lc + a.lam_bnd * lb
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            # GradScaler SKIPS the optimizer step when it finds inf/nan
            # in the gradients. Stepping the scheduler regardless would
            # advance the LR schedule further than the number of steps
            # actually taken, so read the scale before and after and only
            # advance when the step really happened. (This is also what
            # the "lr_scheduler.step() before optimizer.step()" warning
            # was pointing at.)
            prev_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= prev_scale:
                sched.step()
            step += 1
            run_ctc += lc.item()
            run_bnd += lb.detach().item()
            nb += 1

        vl = evaluate(model, dl_va, device, a.val_batches)
        vl_ref = (evaluate(model, dl_va_ref, device, a.val_batches)
                  if dl_va_ref is not None else float("nan"))
        el = time.time() - t0
        per_b = el / max(nb, 1)
        print(f"epoch {ep+1:3d}/{a.epochs}  ctc {run_ctc/max(nb,1):7.4f}  "
              f"bnd {run_bnd/max(nb,1):7.4f}  val {vl:7.4f}  "
              f"val_ref {vl_ref:7.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {el:5.1f}s "
              f"({nb} batches, {per_b:.2f}s/batch)")
        if a.max_batches:
            # Extrapolate, since a capped epoch is a timing probe rather
            # than a real epoch. Stated explicitly because "1 epoch took
            # 40s" is badly misleading when the epoch was 100 of 548
            # batches.
            print(f"  -> full epoch would be {len(dl_tr)} batches "
                  f"~= {per_b * len(dl_tr) / 60:.1f} min")

        config = {"dim": a.dim, "blocks": a.blocks, "heads": a.heads,
                  "kernel": a.kernel, "ff_mult": a.ff_mult,
                  "fusion": a.fusion, "n_layers": len(caches[0].layers),
                  "layers": caches[0].layers, "n_phones": n_phones}

        # LAST, unconditionally, every epoch. Val is a BLEND -- see the
        # split construction above -- roughly 63% easy Iqra random-
        # holdout and only ~37% the hard reciter-generalisation test
        # this split exists for. "Best" selected on that blend can
        # diverge from what actually generalises to reference_library's
        # distribution, the domain the product ships in. LAST is not a
        # fix for that on its own, but it is the natural-endpoint
        # checkpoint uncontaminated by any val-selection bias, and it
        # costs nothing to keep alongside best.pt for comparison.
        torch.save({"model": model.state_dict(), "vocab": vocab,
                   "config": config, "val": vl, "val_ref": vl_ref,
                   "epoch": ep + 1}, out_dir / "last.pt")

        improved = vl < best
        if improved:
            best = vl
            since_best = 0
            torch.save({"model": model.state_dict(), "vocab": vocab,
                       "config": config, "val": vl, "val_ref": vl_ref,
                       "epoch": ep + 1}, out_dir / "best.pt")
        else:
            since_best += 1

        # BEST ON THE HARD SLICE SPECIFICALLY. Selected independently
        # from best.pt: a blended-val improvement and a reciter-only
        # improvement are not the same event, and conflating them is
        # exactly the distortion this checkpoint exists to avoid.
        if dl_va_ref is not None and vl_ref < best_ref[0]:
            best_ref[0] = vl_ref
            torch.save({"model": model.state_dict(), "vocab": vocab,
                       "config": config, "val": vl, "val_ref": vl_ref,
                       "epoch": ep + 1}, out_dir / "best_ref.pt")

        if not improved:
            # EARLY STOPPING on the BLENDED val, unchanged from before.
            # The previous run bottomed at epoch 18 then drifted up for
            # twelve wasted epochs; on a machine that has OOM-killed
            # three earlier runs, that wasted exposure is itself a risk.
            if a.patience and since_best >= a.patience:
                print(f"\nearly stop: {since_best} epochs without "
                      f"improvement (best {best:.4f} at epoch "
                      f"{ep + 1 - since_best})")
                break

    print(f"\nbest (blended) val CTC {best:.4f} -> {out_dir/'best.pt'}")
    if dl_va_ref is not None:
        print(f"best (reciter-only) val CTC {best_ref[0]:.4f} -> "
              f"{out_dir/'best_ref.pt'}")
    print(f"last epoch                        -> {out_dir/'last.pt'}")
    print("\nThree checkpoints, three different questions:")
    print("  best.pt      lowest BLENDED val -- majority-weighted toward")
    print("               the easier Iqra random-holdout slice")
    print("  best_ref.pt  lowest RECITER-ONLY val -- the harder test,")
    print("               closer to the deployment domain")
    print("  last.pt      wherever training ended, no val selection at")
    print("               all -- compare against both above")
    if a.fusion == "weighted_sum":
        w = model.fusion.weights()
        print("\nlearned layer weights:")
        for L, wi in zip(caches[0].layers, w):
            print(f"  layer {L:2d}  {wi:.4f}  {'#' * int(wi * 60)}")
        print("Compare against probe.py's ranking. Broad agreement is")
        print("reassuring; sharp disagreement is worth understanding")
        print("before trusting either.")
    return 0


def cmd_lengths(a):
    """Frame-length distribution of the cached items.

    --max-frames exists to bound memory, but it CROPS FROM THE START while
    keeping the full phone sequence, so any item longer than the cap ends
    up with a CTC target describing audio that is no longer present. That
    teaches the model that phones can simply vanish, which is the opposite
    of what a detector needs. So the cap should be set from the data, not
    guessed -- and if a meaningful fraction of items exceed it, the honest
    fix is a smaller batch rather than a smaller cap.
    """
    caches = [E.FeatureCache(c) for c in a.cache.split(",")]
    lens = []
    for c in caches:
        for k, (lo, hi) in c.index.items():
            lens.append(hi - lo)
    lens = np.array(sorted(lens))
    print(f"items: {len(lens)}")
    print(f"frames  min {lens.min()}  max {lens.max()}  "
          f"mean {lens.mean():.0f}")
    print(f"seconds min {lens.min()/FRAMES_PER_SEC:.1f}  "
          f"max {lens.max()/FRAMES_PER_SEC:.1f}  "
          f"mean {lens.mean()/FRAMES_PER_SEC:.1f}\n")
    for q in (0.5, 0.75, 0.9, 0.95, 0.99, 1.0):
        v = np.quantile(lens, q)
        print(f"  p{int(q*100):3d}  {v:6.0f} frames  "
              f"{v/FRAMES_PER_SEC:5.1f} s")
    print(f"\n{'cap':>7} {'kept whole':>12} {'CROPPED':>9}")
    for cap in (400, 600, 800, 1000, 1200, 1500, 2000):
        n_crop = int((lens > cap).sum())
        print(f"{cap:>7} {len(lens)-n_crop:11d} {n_crop:9d} "
              f"({n_crop/len(lens):.1%})")
    print("\nPick a cap that crops ~nothing. Memory scales with")
    print("batch x cap, so trade against --batch, not against this.")
    return 0


def cmd_labels(a):
    """Inspect labels before training on them."""
    vocab = P.build_vocab_observed(a.text)
    ref, sk = reference_labels(a.manifest, a.text, vocab)
    print(f"reference items: {len(ref)}   skipped: {dict(sk)}")
    for k in list(ref)[:a.n]:
        r = ref[k]
        inv = {v: s for s, v in vocab.items()}
        print(f"\n{k}")
        print(f"  words {r['n_words']}  phones {len(r['seq'])}  "
              f"ends {r['ends'][:8]}")
        print(f"  seq   {' '.join(inv[i] for i in r['seq'][:24])} ...")
    return 0


def main():
    ap = argparse.ArgumentParser(description="train the v6 phone head")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cache", default="cache/feats")
        p.add_argument("--manifest", default="manifest/manifest_clean.csv")
        p.add_argument("--text", default="texts/quran-simple-plain.txt")

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--out", default="runs/head")
    r.add_argument("--device", default="cuda")
    r.add_argument("--dim", type=int, default=128)
    r.add_argument("--blocks", type=int, default=8)
    r.add_argument("--heads", type=int, default=4)
    r.add_argument("--kernel", type=int, default=15)
    r.add_argument("--ff-mult", type=int, default=2)
    r.add_argument("--fusion", default="weighted_sum")
    r.add_argument("--epochs", type=int, default=20)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--lr", type=float, default=3e-4)
    r.add_argument("--weight-decay", type=float, default=0.01)
    # Dropout was hardcoded at 0.1 inside model.py and therefore not
    # tunable without editing code. Exposed rather than raised: the
    # previous run's train/val gap (train CTC 0.27 vs val 0.53) is not
    # yet attributable, because that "val" came from a CONTAMINATED
    # random split with Husary/Minshawy inside training. How much
    # genuine overfitting survives the split fix is an open question,
    # so the knob is available but the default is unchanged.
    r.add_argument("--dropout", type=float, default=0.1)
    r.add_argument("--specaug", action="store_true", default=True,
                   help="SpecAugment masking on cached features")
    r.add_argument("--no-specaug", dest="specaug", action="store_false")
    r.add_argument("--spec-time-masks", type=int, default=2)
    r.add_argument("--spec-time-frac", type=float, default=0.05,
                   help="max width of each time mask, as a fraction of "
                        "utterance length. DELIBERATELY conservative: "
                        "time masking erases the very timing evidence "
                        "the raised --lam-bnd is trying to exploit. "
                        "Masked frames are excluded from the boundary "
                        "loss (see spec_augment) so the two do not "
                        "fight, but the budget is still kept small.")
    r.add_argument("--spec-feat-masks", type=int, default=2)
    r.add_argument("--spec-feat-width", type=int, default=16,
                   help="max width of each frequency mask, in feature "
                        "channels of 1024. No conflict with the boundary "
                        "head -- this removes channels, not time.")
    r.add_argument("--clip", type=float, default=5.0)
    # RAISED FROM 0.3. The decoder's per-error-type breakdown on the
    # held-out reciters showed a sharp, specific pattern at the best
    # operating point (tau=-1.40): every SEGMENTAL category was at or
    # near ceiling -- word_substitution 100%, word_deletion 100%,
    # near_miss_substitution 100%, truncation 100%, control_seam 100% --
    # while the three DURATION/TIMING categories lagged badly:
    # word_repetition 72.0%, madd_lengthening 74.1%, madd_shortening
    # 89.3%. Nothing else sat below 89%.
    #
    # That is explainable rather than mysterious: CTC is deliberately
    # duration-invariant. It marginalises over all alignments, so a
    # phone held twice as long and one held half as long collapse to
    # the SAME label sequence. The main training objective structurally
    # discards exactly the information these three categories depend
    # on. word_repetition fails for the sibling reason -- a repeated
    # word is a VALID phone sequence with extra material CTC happily
    # absorbs.
    #
    # The boundary head is the one part of the model that does see
    # timing: it predicts per-frame word-END positions from the
    # manifest's verified spans, and duration errors ARE boundary-
    # timing errors. Weighting it more heavily is the cheapest
    # available lever on the actual failing categories. 0.3 was a guess
    # made before any of this evidence existed.
    r.add_argument("--lam-bnd", type=float, default=1.0)
    r.add_argument("--patience", type=int, default=6,
                   help="early-stop after this many epochs with no val "
                        "improvement. 0 disables.")
    r.add_argument("--holdout-reciters",
                   default="Husary,Minshawy",
                   help="comma-separated substrings; any reference-library "
                        "reciter matching one goes to val, never train. "
                        "Matches testset/meta.json's own convention.")
    r.add_argument("--pos-weight", type=float, default=20.0,
                   help="word ends are ~1 frame in 50; without this the "
                        "boundary head predicts 'no' everywhere")
    r.add_argument("--no-boundary", action="store_true")
    r.add_argument("--dynamic", action="store_true", default=True)
    r.add_argument("--no-dynamic", dest="dynamic", action="store_false")
    r.add_argument("--max-frames", type=int, default=1500)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--val-batches", type=int, default=40)
    r.add_argument("--max-batches", type=int, default=0,
                   help="stop each epoch after N batches. A TIMING PROBE, "
                        "not a training mode -- the LR schedule is still "
                        "sized for full epochs, so a capped run's loss "
                        "curve is not comparable to a real one.")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=cmd_run)

    l = sub.add_parser("labels")
    common(l)
    l.add_argument("--n", type=int, default=5)
    l.set_defaults(func=cmd_labels)

    g = sub.add_parser("lengths")
    common(g)
    g.set_defaults(func=cmd_lengths)

    a = ap.parse_args()
    rc = a.func(a) or 0
    sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
