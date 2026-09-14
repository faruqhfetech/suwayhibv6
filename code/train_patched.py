#!/usr/bin/env python3
"""
Suwayhib v6 -- train the phone head.

    python code/train.py labels --cache cache/feats --n 5
    python code/train.py run --cache cache/feats --epochs 20
    python code/train.py run --cache cache/feats,cache/feats_iqra \\
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
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
FRAMES_PER_SEC = 50


def _load(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------

def reference_labels(manifest, text_path, vocab):
    """-> {cache_key: {"seq": [...], "ends": [frame, ...], "n_words": k}}

    Sequence from our G2P; word-end frames from the verified manifest.
    """
    B = _load("bootstrap")
    P = _load("phones")
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
    """
    MEMORY NOTE (why this isn't just `self.keys = [...]`, `self.labels = {...}`
    the way it started): with num_workers>0 AND persistent_workers=True,
    worker processes are forked once and live for the whole run instead of
    respawning every epoch. Every time a worker reads an element out of a
    Python list or dict, CPython bumps that object's refcount, and Linux's
    copy-on-write fork implementation responds by duplicating the touched
    memory page into the worker's own private memory -- even though nothing
    was actually written. Persistent workers never re-fork, so this never
    resets: shuffling means each worker eventually touches most of the
    dataset, and each worker ends up holding its own growing private copy
    of the metadata. Four workers, four copies, growing a little more each
    epoch -- exactly the "used+swap climbs every epoch, never plateaus"
    pattern from the crash logs.

    Fix: store per-item metadata as numpy arrays of PRIMITIVE dtype
    (int64, fixed-width unicode) instead of Python objects. A numpy array
    of primitive dtype has exactly one Python refcount for the whole
    buffer -- indexing into it touches no per-element refcount, so there
    is nothing for copy-on-write to duplicate as workers read it.
    """

    def __init__(self, caches, labels, max_frames=1500):
        self.caches = caches
        self.max_frames = max_frames

        ci_list, key_list = [], []
        seq_flat, seq_off = [], [0]
        end_flat, end_off = [], [0]
        has_end = []

        for ci, c in enumerate(caches):
            for k in c.index:
                if k not in labels:
                    continue
                lo, hi = c.index[k]
                if hi - lo < 8:
                    continue
                lab = labels[k]
                ci_list.append(ci)
                key_list.append(k)
                seq_flat.extend(lab["seq"])
                seq_off.append(len(seq_flat))
                if lab["ends"] is not None:
                    end_flat.extend(lab["ends"])
                    has_end.append(True)
                else:
                    has_end.append(False)
                end_off.append(len(end_flat))

        self.ci = np.asarray(ci_list, dtype=np.int32)
        width = max((len(k) for k in key_list), default=1)
        self.item_keys = np.asarray(key_list, dtype=f"<U{width}")
        self.seq_flat = np.asarray(seq_flat, dtype=np.int64)
        self.seq_off = np.asarray(seq_off, dtype=np.int64)
        self.end_flat = np.asarray(end_flat, dtype=np.int64)
        self.end_off = np.asarray(end_off, dtype=np.int64)
        self.has_end = np.asarray(has_end, dtype=bool)

    def __len__(self):
        return len(self.ci)

    def __getitem__(self, i):
        ci = int(self.ci[i])
        k = str(self.item_keys[i])
        x = self.caches[ci].get(k)                 # (T, L, D)
        T = x.shape[0]
        if T > self.max_frames:
            # Crop from the START. Cropping mid-utterance would leave the
            # CTC target describing audio that is no longer present, which
            # teaches the model that phones can vanish.
            x = x[:self.max_frames]
            T = self.max_frames

        s0, s1 = self.seq_off[i], self.seq_off[i + 1]
        seq = self.seq_flat[s0:s1]

        ends = None
        if self.has_end[i]:
            e0, e1 = self.end_off[i], self.end_off[i + 1]
            ends = [e for e in self.end_flat[e0:e1].tolist() if e < T]

        return {"x": torch.from_numpy(x), "seq": torch.from_numpy(seq),
                "ends": ends, "T": T, "key": k}


def make_collate(fixed_pad_to=None):
    """
    fixed_pad_to: if set, every batch is padded to this many frames
    instead of to that batch's own longest item. Costs some wasted
    compute on short batches, but every batch tensor then has the exact
    same shape.

    That matters with --workers>0 + pin_memory=True: PyTorch's host-side
    pinned-memory allocator caches one buffer per DISTINCT shape it has
    seen and does not release them back to the OS. Here T depends on
    which utterances shuffle into a batch, so shapes vary continuously --
    the cache can keep growing new buffers all run instead of reusing
    one, which looks identical to a leak in `free -h`. A fixed shape caps
    it at a single buffer after the first batch. Pass a.max_frames
    (--fixed-pad) to test this; leave None for the original behaviour.
    """
    def collate(batch):
        B = len(batch)
        T = fixed_pad_to if fixed_pad_to else max(b["T"] for b in batch)
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
    return collate


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


def _mem_snapshot():
    """Compact system memory snapshot for tracking growth across epochs
    objectively instead of eyeballing `free -h` between runs. Reads
    /proc/meminfo directly (Linux/WSL only) to avoid a psutil dependency.
    """
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, val = line.split(":")
                info[key.strip()] = int(val.strip().split()[0])  # kB
        avail = info.get("MemAvailable", 0) / 1024 / 1024
        swap_used = (info.get("SwapTotal", 0) - info.get("SwapFree", 0)) / 1024 / 1024
        return f"avail {avail:4.1f}G  swap {swap_used:4.1f}G"
    except Exception:
        return "mem n/a"


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
    M = _load("model")
    P = _load("phones")
    E = _load("extract")

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
    n_val = max(1, int(len(ds) * 0.1))
    g = torch.Generator().manual_seed(a.seed)
    tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val],
                                           generator=g)
    # persistent_workers: without it, DataLoader tears down and respawns
    # every worker each epoch, which means re-opening a fresh memmap
    # handle on cache/feats_iqra (44 GB) every time -- real overhead on a
    # file this size. pin_memory: makes the .to(device, non_blocking=True)
    # calls in the training loop actually overlap the H2D copy with
    # compute; without pinned memory that flag is silently a no-op and the
    # copy is synchronous regardless. Both are only useful with
    # num_workers > 0 / a CUDA device -- harmless if not, so left on
    # unconditionally rather than gated on a.workers/a.device.
    collate_fn = make_collate(a.max_frames if a.fixed_pad else None)
    dl_tr = torch.utils.data.DataLoader(
        tr, batch_size=a.batch, shuffle=True, collate_fn=collate_fn,
        num_workers=a.workers, drop_last=True,
        persistent_workers=(a.workers > 0 and not a.no_persistent_workers),
        pin_memory=True)
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
    dl_va = torch.utils.data.DataLoader(
        va, batch_size=a.batch, shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=True)

    model = M.PhoneHead(
        n_phones=n_phones, n_layers=len(caches[0].layers), dim=a.dim,
        blocks=a.blocks, heads=a.heads, kernel=a.kernel, ff_mult=a.ff_mult,
        fusion=a.fusion).to(device)

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
    step = 0
    start_epoch = 0

    if a.resume:
        # A real resume, not a warm-start: model AND optimizer momentum
        # AND scheduler position are restored, so training picks up as if
        # it never stopped. total_steps for OneCycleLR must match the
        # ORIGINAL run's --epochs for the schedule shape to line up --
        # pass the same --epochs you started with, not a shorter number.
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            sched.load_state_dict(ck["scheduler"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck.get("epoch", 0)
        best = ck.get("best", ck.get("val", float("inf")))
        print(f"resumed from {a.resume}  (epoch {start_epoch}, "
              f"best val so far {best:.4f})")
        if start_epoch >= a.epochs:
            print(f"--epochs ({a.epochs}) is not past the checkpoint's "
                  f"epoch ({start_epoch}) -- nothing to do; raise --epochs.")

    for ep in range(start_epoch, a.epochs):
        t0 = time.time()
        run_ctc, run_bnd, nb = 0.0, 0.0, 0
        for bi, b in enumerate(dl_tr):
            if a.max_batches and bi >= a.max_batches:
                break
            chunk, left = M.sample_chunk(True, a.dynamic, int(b["lens"].max()))
            x = b["x"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x, chunk=chunk, left_chunks=left)
                lc = ctc_loss(out["logits"], b["lens"], b["seq"], b["slens"])
                lb = torch.zeros((), device=device)
                if not a.no_boundary and out.get("boundary") is not None:
                    lb = boundary_loss(out["boundary"], b["bnd"].to(device),
                                       b["bnd_mask"].to(device), a.pos_weight)
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
        el = time.time() - t0
        per_b = el / max(nb, 1)
        print(f"epoch {ep+1:3d}/{a.epochs}  ctc {run_ctc/max(nb,1):7.4f}  "
              f"bnd {run_bnd/max(nb,1):7.4f}  val {vl:7.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {el:5.1f}s "
              f"({nb} batches, {per_b:.2f}s/batch)  {_mem_snapshot()}")
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

        if vl < best:
            best = vl
            torch.save({
                "model": model.state_dict(),
                "vocab": vocab, "config": config,
                "val": vl, "epoch": ep + 1,
            }, out_dir / "best.pt")

        # Saved every epoch (unlike best.pt, which only updates on
        # improvement) so a crash/OOM can pick up from wherever training
        # actually stopped, not from wherever validation last improved.
        # Carries optimizer/scheduler/scaler state too -- see --resume --
        # so continuing isn't a fresh warm-start on old weights.
        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "vocab": vocab, "config": config,
            "val": vl, "epoch": ep + 1, "best": best,
        }, out_dir / "last.pt")

    print(f"\nbest val CTC {best:.4f} -> {out_dir/'best.pt'}")
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
    E = _load("extract")
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
    P = _load("phones")
    E = _load("extract")
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
    r.add_argument("--clip", type=float, default=5.0)
    r.add_argument("--lam-bnd", type=float, default=0.3)
    r.add_argument("--pos-weight", type=float, default=20.0,
                   help="word ends are ~1 frame in 50; without this the "
                        "boundary head predicts 'no' everywhere")
    r.add_argument("--no-boundary", action="store_true")
    r.add_argument("--dynamic", action="store_true", default=True)
    r.add_argument("--no-dynamic", dest="dynamic", action="store_false")
    r.add_argument("--max-frames", type=int, default=1500)
    r.add_argument("--fixed-pad", action="store_true",
                   help="pad every batch to --max-frames instead of that "
                        "batch's own longest item; trades some wasted "
                        "compute for a constant tensor shape (see "
                        "make_collate's docstring)")
    r.add_argument("--no-persistent-workers", action="store_true",
                   help="tear down and re-fork DataLoader workers every "
                        "epoch instead of keeping them alive for the "
                        "whole run; useful to A/B test against the "
                        "fork/refcount growth issue described on Items")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--val-batches", type=int, default=40)
    r.add_argument("--max-batches", type=int, default=0,
                   help="stop each epoch after N batches. A TIMING PROBE, "
                        "not a training mode -- the LR schedule is still "
                        "sized for full epochs, so a capped run's loss "
                        "curve is not comparable to a real one.")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--resume", default=None,
                   help="path to a checkpoint saved by this script "
                        "(e.g. runs/head/last.pt) to continue training "
                        "from -- restores model, optimizer, scheduler, "
                        "AMP scaler, and epoch counter, unlike loading "
                        "best.pt's weights alone into a fresh run")
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
