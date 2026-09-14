#!/usr/bin/env python3
"""
Suwayhib v6 -- XLS-R feature extraction and caching.

    python -m suwayhib.pipeline.extract probe --out cache/probe --hours 2.5
    python -m suwayhib.pipeline.extract full  --out cache/feats --layers 14,15,16,17,18,19
    python -m suwayhib.pipeline.extract info  --cache cache/probe

WHY CACHE AT ALL
----------------
The encoder is FROZEN. Its output for a given waveform never changes, so
recomputing it every epoch is pure waste. v5's dataset preparation spawned
one ffmpeg subprocess per WORD and measured >40 minutes per epoch,
essentially all process spawn and I/O. Extract once, memmap, and every
subsequent training epoch becomes array slicing.

TWO MODES, AND WHY THEY ARE SEPARATE
-------------------------------------
probe: MANY layers over a LITTLE audio.
  The layer probe answers "which layers carry phonetic information", which
  is a comparison between layers, not an attempt to build a good
  recogniser. A linear probe on 1024-dim features has ~70K parameters;
  2.5 hours at 50Hz is ~450,000 frames, massively over-determined for
  that. The RANKING between layers stabilises long before the absolute
  accuracy does.

full: FEW layers over ALL the audio.
  Once the band is known, cache only that band.

Doing both at once would mean caching every candidate layer over the whole
corpus and discarding most of it: at 1024-dim/50Hz/fp16 each layer costs
~100KB per second of audio, so 6 layers x 79h is ~170GB against ~28GB for
one. The forward pass costs the same either way -- only storage differs.

WHICH LAYERS
------------
XLS-R-300M has 24 transformer layers, NOT 12. The phonetic peak in
24-layer wav2vec2-family models sits at roughly layers 15-19; the first
~10 and the last few are poor for phoneme recognition (Pasad et al.;
"Normalization through Fine-tuning" 2025). An earlier version of this plan
said "layers 6-16", which is the band for a 12-LAYER model and would have
missed the peak entirely. Default probe range here is 10-22 to bracket it
with margin on both sides.

Note the probe is NOT only for picking a single layer. The downstream head
uses a learned weighted sum over a band (SUPERB protocol: weighted sum
beats last-layer "either equally good or significantly better"). The probe
tells us WHERE the band is and confirms the expected single-peaked
profile. A FLAT probe curve is a red flag for the G2P or the frame
alignment, not evidence that layer choice does not matter.

MEMORY
------
Sized for a 4GB card. The encoder is inference-only (no gradients, no
optimizer state), so ~600MB of fp16 weights plus small activations.
Batching is by TOTAL SAMPLES, not utterance count, because a fixed
utterance batch size either wastes memory on short clips or OOMs on long
ones. --max-batch-seconds is the real knob.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from ..core import bootstrap as B

SR = 16000
MODEL = "facebook/wav2vec2-xls-r-300m"

# XLS-R stride: 20ms per frame => 50 frames/sec.
FRAME_MS = 20
FRAMES_PER_SEC = 1000 // FRAME_MS


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------

class Encoder:
    """Frozen XLS-R, returning selected hidden layers."""

    def __init__(self, model=MODEL, device="cuda", fp16=True):
        import torch
        from transformers import Wav2Vec2Model
        self.torch = torch
        self.device = torch.device(device)
        self.fp16 = fp16 and self.device.type == "cuda"
        self.model = Wav2Vec2Model.from_pretrained(
            model, torch_dtype=torch.float16 if self.fp16 else torch.float32)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_layers = self.model.config.num_hidden_layers
        self.dim = self.model.config.hidden_size
        n = sum(p.numel() for p in self.model.parameters())
        print(f"encoder: {model} ({n/1e6:.0f}M params, {self.n_layers} "
              f"layers, dim {self.dim}) on {self.device}"
              f"{' fp16' if self.fp16 else ''}")

    def __call__(self, waves, layers):
        """waves: list of 1-D float32 arrays. -> list of (T, L, D) fp16 arrays.

        Zero-padded into a batch, then trimmed back per utterance using the
        model's own frame-count formula. Trimming matters: padding frames
        are not silence, they are whatever the convolutional front end
        makes of a zero run, and leaving them in would put fabricated
        frames into the training cache.
        """
        torch = self.torch
        lengths = [len(w) for w in waves]
        maxlen = max(lengths)
        batch = np.zeros((len(waves), maxlen), dtype=np.float32)
        mask = np.zeros((len(waves), maxlen), dtype=np.int64)
        for i, w in enumerate(waves):
            batch[i, :len(w)] = w
            mask[i, :len(w)] = 1

        x = torch.from_numpy(batch).to(self.device)
        m = torch.from_numpy(mask).to(self.device)
        if self.fp16:
            x = x.half()

        with torch.no_grad():
            out = self.model(x, attention_mask=m, output_hidden_states=True)
        # hidden_states[0] is the conv feature projection; transformer
        # layer k is hidden_states[k]. We index by TRANSFORMER layer, so
        # layer 15 means hidden_states[15].
        hs = out.hidden_states

        feats = []
        for i in range(len(waves)):
            n_frames = int(self.model._get_feat_extract_output_lengths(
                torch.tensor(lengths[i])).item())
            stack = torch.stack([hs[k][i, :n_frames] for k in layers], dim=1)
            feats.append(stack.float().cpu().numpy().astype(np.float16))
        return feats


# --------------------------------------------------------------------------
# corpora
# --------------------------------------------------------------------------

def iter_reference_library(manifest, lib, cache_dir, limit_seconds=None):
    """Ayah-level items from our own reference library.

    Uses bootstrap.py's AudioCache when present -- that cache already
    round-trip verified, so reading it costs no subprocesses.
    """
    rows = B.load_manifest(manifest)
    paths = sorted({r["audio_path"] for r in rows})

    cache = None
    if cache_dir and (Path(cache_dir) / "index.json").exists():
        cache = B.AudioCache(cache_dir)

    total = 0.0
    for p in paths:
        wav = cache.ayah(p) if cache else B.decode(Path(lib) / p)
        dur = len(wav) / SR
        if limit_seconds and total + dur > limit_seconds:
            break
        total += dur
        # key must be filesystem-safe and stable across runs
        yield {"key": "ref/" + p.replace("/", "__").replace(".wav", ""),
               "audio": wav, "source": "reference_library", "path": p}


def _decode_audio_field(a):
    """-> (float32 mono array, sample_rate).

    `datasets` has changed this field's type across versions: older
    releases hand back a plain dict {"array", "sampling_rate"}, newer ones
    a torchcodec AudioDecoder object. Both shapes appear in the wild
    depending on the installed version, so handle both rather than pinning
    -- and fail loudly on anything else instead of guessing, since a
    silently mis-decoded waveform produces features that look entirely
    plausible and are entirely wrong.
    """
    if isinstance(a, dict):
        return np.asarray(a["array"], dtype=np.float32), int(
            a.get("sampling_rate", SR))

    # torchcodec AudioDecoder: get_all_samples() -> AudioSamples with
    # .data (channels, samples) and .sample_rate
    if hasattr(a, "get_all_samples"):
        s = a.get_all_samples()
        d = s.data
        if hasattr(d, "numpy"):
            d = d.numpy()
        d = np.asarray(d, dtype=np.float32)
        if d.ndim == 2:            # (channels, samples) -> mono
            d = d.mean(axis=0)
        return d, int(s.sample_rate)

    raise SystemExit(
        f"unrecognised audio field type {type(a)!r}. The `datasets` audio "
        f"representation has changed again; extend _decode_audio_field "
        f"rather than working around it at the call site.")


def iter_hf(dataset, split, limit_seconds=None, text_key="tashkeel_sentence"):
    """Items from an IqraEval HF dataset. Audio decoded by `datasets`."""
    from datasets import load_dataset
    # STREAMING. The non-streaming path materialises the split: on
    # Iqra_train's 71,391 examples that reached 14.3 GB resident (89% of
    # a 16 GB machine), which starved every other process and made the
    # extraction itself crawl. Streaming reads shard by shard and keeps
    # memory flat, at the cost of no random access -- which this loop
    # does not need, since it walks the split in order anyway.
    ds = load_dataset(dataset, split=split, streaming=True)
    total = 0.0
    for i, r in enumerate(ds):
        wav, sr = _decode_audio_field(r["audio"])
        if sr != SR:
            raise SystemExit(
                f"{dataset}: sampling rate {sr} != {SR}. Resample before "
                f"extraction -- silently feeding the wrong rate to the "
                f"encoder produces plausible-looking garbage.")
        dur = len(wav) / SR
        if limit_seconds and total + dur > limit_seconds:
            break
        total += dur
        yield {"key": f"{dataset.split('/')[-1]}/{r.get('id', i)}",
               "audio": wav, "source": dataset,
               "text": r.get(text_key) or r.get("sentence"),
               "phoneme_ref": r.get("phoneme_ref")}


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

class FeatureWriter:
    """Appends (T, L, D) fp16 blocks to one flat memmap plus an index.

    One big file rather than many small ones: 74k separate .npy files would
    make the index the bottleneck on a Windows-backed mount, and random
    access during training wants a single mapping anyway.
    """

    def __init__(self, out, layers, dim, est_frames):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.layers = list(layers)
        self.dim = dim
        self.capacity = int(est_frames * 1.15) + 1024      # slack
        shape = (self.capacity, len(self.layers), dim)
        self.data = np.lib.format.open_memmap(
            self.out / "feats.npy", mode="w+", dtype=np.float16, shape=shape)
        self.pos = 0
        self.index = {}
        self.meta_rows = []
        print(f"allocated {np.prod(shape) * 2 / 1e9:.1f} GB "
              f"({self.capacity} frames x {len(self.layers)} layers x {dim})")

    def add(self, key, feat, meta=None):
        t = feat.shape[0]
        if self.pos + t > self.capacity:
            raise SystemExit(
                f"cache capacity exceeded at {self.pos} frames. Re-run with "
                f"a larger --est-hours; the estimate was too low.")
        self.data[self.pos:self.pos + t] = feat
        self.index[key] = [self.pos, self.pos + t]
        self.pos += t
        if meta:
            self.meta_rows.append({"key": key, "frames": t, **meta})

    def close(self, trim=False):
        """Finalise. `trim` rewrites the file without its slack.

        TRIMMING IS OFF BY DEFAULT AND THAT IS DELIBERATE. The trim reads
        the whole array and writes a second copy: on a 50 GB cache that is
        100 GB of I/O on a /mnt/c mount, held open alongside a process
        already under memory pressure. A 20-hour Iqra_train extraction was
        killed by the OOM reaper at exactly this step, after ALL the GPU
        work had completed -- the most expensive possible place to fail.

        The slack is only wasted disk. Every consumer goes through
        FeatureCache, which slices by the byte ranges in index.json and
        never reads past `frames`, so unused tail frames are unreachable
        rather than wrong. index.json records the true count either way.
        """
        self.data.flush()
        if trim:
            used = self.data[:self.pos]
            np.save(self.out / "feats_trimmed.npy", used)
            del self.data
            os.replace(self.out / "feats_trimmed.npy",
                       self.out / "feats.npy")
        else:
            del self.data

        self._write_index()
        if self.meta_rows:
            with open(self.out / "meta.jsonl", "w", encoding="utf-8") as fh:
                for r in self.meta_rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {self.out}  {self.pos} frames "
              f"({self.pos / FRAMES_PER_SEC / 3600:.2f} h), "
              f"{self.pos * len(self.layers) * self.dim * 2 / 1e9:.1f} GB")

    def checkpoint(self):
        """Make everything written so far readable, mid-run."""
        self.data.flush()
        self._write_index()
        if self.meta_rows:
            with open(self.out / "meta.jsonl", "w", encoding="utf-8") as fh:
                for r in self.meta_rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _write_index(self):
        (self.out / "index.json").write_text(json.dumps({
            "layers": self.layers, "dim": self.dim,
            "frames": self.pos, "frame_ms": FRAME_MS,
            "dtype": "float16", "model": MODEL,
            "index": self.index,
        }))


class FeatureCache:
    """Read-only view. Slicing, no decode."""

    def __init__(self, path):
        d = Path(path)
        meta = json.loads((d / "index.json").read_text())
        self.layers = meta["layers"]
        self.dim = meta["dim"]
        self.frame_ms = meta["frame_ms"]
        self.index = meta["index"]
        self.data = np.load(d / "feats.npy", mmap_mode="r")

    def __len__(self):
        return len(self.index)

    def keys(self):
        return list(self.index)

    def get(self, key, layer=None):
        """-> (T, L, D) or (T, D) when `layer` (an absolute layer number)
        is given."""
        lo, hi = self.index[key]
        x = np.asarray(self.data[lo:hi], dtype=np.float32)
        if layer is None:
            return x
        return x[:, self.layers.index(layer), :]


# --------------------------------------------------------------------------
# extraction driver
# --------------------------------------------------------------------------

def run(items, layers, out, device, est_hours, max_batch_seconds, fp16):
    enc = Encoder(device=device, fp16=fp16)
    bad = [k for k in layers if not (1 <= k <= enc.n_layers)]
    if bad:
        raise SystemExit(f"layers {bad} outside 1..{enc.n_layers}")

    est_frames = int(est_hours * 3600 * FRAMES_PER_SEC)
    w = FeatureWriter(out, layers, enc.dim, est_frames)

    batch, keys, metas, batch_sec = [], [], [], 0.0
    n = 0
    t0 = time.time()
    last_ckpt = 0.0

    def flush():
        nonlocal batch, keys, metas, batch_sec, n
        if not batch:
            return
        feats = enc(batch, layers)
        for k, f, m in zip(keys, feats, metas):
            w.add(k, f, m)
        n += len(batch)
        done_sec = w.pos / FRAMES_PER_SEC
        el = time.time() - t0
        print(f"  {n} items  {done_sec/3600:.2f} h cached  "
              f"{done_sec/max(el,1e-9):.1f}x realtime", flush=True)
        batch, keys, metas, batch_sec = [], [], [], 0.0
        # Checkpoint the index periodically. The feature data is already
        # on disk (it is a memmap), so writing the index makes everything
        # extracted so far READABLE even if the process dies later. A run
        # that got 20 hours in and was OOM-killed during finalisation lost
        # all of it purely because the index had never been written.
        nonlocal last_ckpt
        if done_sec - last_ckpt > 1800:
            w.checkpoint()
            last_ckpt = done_sec

    for it in items:
        dur = len(it["audio"]) / SR
        if batch and batch_sec + dur > max_batch_seconds:
            flush()
        batch.append(it["audio"])
        keys.append(it["key"])
        metas.append({k: v for k, v in it.items()
                      if k not in ("audio", "key") and v is not None})
        batch_sec += dur
    flush()
    w.close()
    return 0


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_probe(a):
    """Many layers, little audio -- for the layer probe only."""
    layers = parse_layers(a.layers)
    print(f"PROBE: layers {layers} over ~{a.hours} h\n")
    items = iter_reference_library(a.manifest, a.lib, a.audio_cache,
                                   limit_seconds=a.hours * 3600)
    return run(items, layers, a.out, a.device, a.hours * 1.1,
               a.max_batch_seconds, not a.fp32)


def cmd_full(a):
    """Few layers, all the audio -- for training."""
    layers = parse_layers(a.layers)
    print(f"FULL: layers {layers}\n")

    def chain():
        if a.reference:
            yield from iter_reference_library(a.manifest, a.lib,
                                              a.audio_cache)
        for spec in a.hf or []:
            # "dataset:split:hours" -- hours optional. Per-source limits
            # rather than one global cap, because the sources are not
            # interchangeable: Iqra_Extra_IS26's 1.5h of REAL human
            # mispronunciations is the most valuable data in the project
            # per byte, and must never be the thing a global limit cuts
            # off partway through.
            parts = spec.split(":")
            name = parts[0]
            split = parts[1] if len(parts) > 1 and parts[1] else "train"
            hours = float(parts[2]) if len(parts) > 2 and parts[2] else None
            if hours:
                print(f"  {name}:{split} limited to {hours} h")
            yield from iter_hf(name, split,
                               limit_seconds=hours * 3600 if hours else None)

    return run(chain(), layers, a.out, a.device, a.est_hours,
               a.max_batch_seconds, not a.fp32)


def cmd_info(a):
    c = FeatureCache(a.cache)
    print(f"layers   : {c.layers}")
    print(f"dim      : {c.dim}")
    print(f"items    : {len(c)}")
    tot = sum(hi - lo for lo, hi in c.index.values())
    print(f"frames   : {tot} ({tot / FRAMES_PER_SEC / 3600:.2f} h)")
    print(f"size     : {tot * len(c.layers) * c.dim * 2 / 1e9:.1f} GB")
    k = c.keys()[0]
    x = c.get(k)
    print(f"\nsample   : {k}  shape {x.shape}")
    print(f"  finite : {np.isfinite(x).all()}")
    print(f"  range  : [{x.min():.3f}, {x.max():.3f}]  "
          f"mean {x.mean():.3f}  std {x.std():.3f}")
    # A layer that is constant across frames carries no information and
    # usually means an indexing mistake, so check rather than assume.
    for i, L in enumerate(c.layers):
        s = float(x[:, i, :].std())
        flag = "  <-- NEAR-CONSTANT, suspicious" if s < 1e-3 else ""
        print(f"  layer {L:2d}  std {s:.4f}{flag}")
    return 0


def parse_layers(s):
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description="XLS-R feature extraction")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--device", default="cuda")
        p.add_argument("--fp32", action="store_true",
                       help="disable fp16 (slower, more VRAM)")
        p.add_argument("--max-batch-seconds", type=float, default=60.0,
                       help="batch by total audio, not utterance count; "
                            "lower this first if you hit OOM")
        p.add_argument("--manifest", default="manifest/manifest_clean.csv")
        p.add_argument("--lib", default="reference_library")
        p.add_argument("--audio-cache", default="cache/audio")

    p = sub.add_parser("probe")
    common(p)
    p.add_argument("--layers", default="10-22",
                   help="XLS-R-300M has 24 layers; the phonetic peak is "
                        "around 15-19, so 10-22 brackets it")
    p.add_argument("--hours", type=float, default=2.5)
    p.add_argument("--out", default="cache/probe")
    p.set_defaults(func=cmd_probe)

    f = sub.add_parser("full")
    common(f)
    f.add_argument("--layers", required=True,
                   help="the band chosen by the probe, e.g. 14-19")
    f.add_argument("--out", default="cache/feats")
    f.add_argument("--est-hours", type=float, required=True,
                   help="total audio hours, for preallocation")
    f.add_argument("--reference", action="store_true",
                   help="include our own reference_library")
    f.add_argument("--hf", action="append",
                   help="HF dataset as name[:split[:hours]], repeatable. "
                        "e.g. IqraEval/Iqra_train:train:20 for a 20-hour "
                        "slice, or IqraEval/Iqra_Extra_IS26:train for all "
                        "of it. Without the hours field the WHOLE split is "
                        "extracted -- 79 h for Iqra_train.")
    f.set_defaults(func=cmd_full)

    i = sub.add_parser("info")
    i.add_argument("--cache", default="cache/probe")
    i.set_defaults(func=cmd_info)

    a = ap.parse_args()
    rc = a.func(a) or 0
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
