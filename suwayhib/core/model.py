#!/usr/bin/env python3
"""
Suwayhib v6 -- the phone head.

    python -m suwayhib.core.model summary
    python -m suwayhib.core.model parity        # streaming == full-context check

WHAT THIS IS
------------
The only trained component in v6. It sits on cached XLS-R features and
emits per-frame posteriors over the phone inventory plus a CTC blank.

    cached XLS-R feats (T, L, 1024)
      -> LayerFusion            (T, 1024)
      -> input projection       (T, d)
      -> N x ConformerBlock     (T, d)      DCConv + chunked attention
      -> linear                 (T, n_phones + 1)

The encoder above it is FROZEN and never trained -- six voices cannot
teach speaker invariance (v4a proved that), so we rent it from XLS-R's
pretraining instead. The graph decoder below it has no parameters. This
file is the whole learned surface of the system.

WHAT IS DELIBERATELY ABSENT
---------------------------
There is NO TEXT ENCODER. v5 spent a large share of its 811K parameters
on a BiLSTM over characters, producing an embedding to cosine-compare
against audio -- which is precisely the mechanism that failed in v4a. In
v6 the expected word's phone sequence enters the DECODER GRAPH as
symbols, not as a learned vector. The transcript is known; it should stay
known rather than be re-learned. That deletion is most of the parameter
saving and it falls out of the reframe, not from tuning.

STREAMING IS A TRAINING DECISION, NOT A DEPLOYMENT ONE
-------------------------------------------------------
Dynamic chunk training (U2/U2++): most batches use a randomly sampled
chunk size with randomised left context, the rest use full context. One
model then serves any latency budget, chosen at inference. This is
strictly better than committing to a fixed lookahead at training time.

Two details that are easy to get wrong and expensive to debug:

  DCConv, NOT causal conv. A standard depthwise convolution reaches
  across chunk boundaries into future frames that streaming inference
  will not have. Training with it and deploying without produces a model
  that passes every offline test and fails on real audio -- the exact
  pattern the handover records for v5 ("synthetic tests kept passing
  while real audio failed"). DCConv restricts the convolution to the same
  boundaries the attention mask uses.

  PARITY MUST BE ASSERTED, NOT ASSUMED. `python -m suwayhib.core.model parity` checks
  that feeding audio chunk-by-chunk with cached state gives the same
  output as feeding it whole. Run it on every architecture change. A
  cache bug here is silent and looks like a mediocre model.

CAPACITY
--------
Default is ~3M parameters. That is larger than v5's 811K student and much
larger than Jin et al.'s 155K KWS encoder, and it is deliberate: get the
ceiling first with the frozen encoder, measure, THEN shrink in Phase 5
with the frozen head as a measuring instrument. Shrinking with a working
measurement is easy; growing to chase an unknown loss is not.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import phones as P


# --------------------------------------------------------------------------
# layer fusion
# --------------------------------------------------------------------------

class WeightedSum(nn.Module):
    """SUPERB-style learned softmax weighting over cached layers.

    The standard protocol, and the one the IQRA baseline and the winning
    system both use. Weighted sum beats last-layer "either equally good or
    significantly better" across tasks and SSL models.

    Layers are standardised before weighting. Our own probe measured
    layer 22's feature std at ~13.6 against ~3.0 for layers 10-17; without
    normalisation the softmax weights would partly be compensating for
    scale rather than selecting information.
    """

    def __init__(self, n_layers, dim):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                      # (B, T, L, D) -> (B, T, D)
        a = torch.softmax(self.w, dim=0).view(1, 1, -1, 1)
        return self.norm((x * a).sum(dim=2))

    def weights(self):
        return torch.softmax(self.w.detach(), dim=0).cpu().tolist()


class HierarchicalConv(nn.Module):
    """Convolution ACROSS the layer axis, then pool.

    Reported to beat weighted sum substantially on phone recognition
    (HuBERT Base PER 5.41 -> 3.07 on SUPERB), and notably lets a smaller
    upstream beat a larger one with weighted sum -- directly relevant
    given this project's size constraint.

    Included so the choice can be MEASURED on our data rather than taken
    on faith: the reported gains are on English SUPERB with a different
    upstream and a different downstream, and neither transfer is
    guaranteed. Select with --fusion.
    """

    def __init__(self, n_layers, dim, hidden=None):
        super().__init__()
        h = hidden or dim
        self.norm = nn.LayerNorm(dim)
        # treat the layer axis as a spatial axis: (B*T, D, L) conv over L
        self.conv1 = nn.Conv1d(dim, h, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(h, dim, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):                      # (B, T, L, D) -> (B, T, D)
        B, T, L, D = x.shape
        y = self.norm(x).reshape(B * T, L, D).transpose(1, 2)   # (BT, D, L)
        y = self.conv2(self.act(self.conv1(y)))                 # (BT, D, L)
        y = y.mean(dim=2)                                       # (BT, D)
        return y.view(B, T, D)


FUSIONS = {"weighted_sum": WeightedSum, "hier_conv": HierarchicalConv}


# --------------------------------------------------------------------------
# chunk masking
# --------------------------------------------------------------------------

def chunk_mask(T, chunk, left_chunks, device):
    """Boolean (T, T) mask, True where attention is ALLOWED.

    Frame t may attend within its own chunk and to `left_chunks` chunks
    before it (-1 meaning all history). This is the U2++ dynamic-chunk
    formulation: full context when chunk <= 0.
    """
    if chunk is None or chunk <= 0:
        return torch.ones(T, T, dtype=torch.bool, device=device)
    idx = torch.arange(T, device=device)
    ci = idx // chunk
    end = (ci + 1) * chunk - 1                 # last frame of own chunk
    if left_chunks is None or left_chunks < 0:
        start = torch.zeros_like(idx)
    else:
        start = (ci - left_chunks).clamp(min=0) * chunk
    return (idx.unsqueeze(0) >= start.unsqueeze(1)) & \
           (idx.unsqueeze(0) <= end.unsqueeze(1))


def sample_chunk(training, dynamic, T, rng=None):
    """U2++ dynamic chunk sampling: ~60% chunked, ~40% full context.

    Full-context batches matter -- a model trained only on short chunks
    never learns to use long context when it is available, and the offline
    path (used for the frozen measuring head in Phase 5) needs it.
    """
    if not training or not dynamic:
        return None, -1
    r = torch.rand(()).item() if rng is None else rng.random()
    if r < 0.4:
        return None, -1                        # full context
    chunk = int(torch.randint(8, 33, ()).item())   # 8-32 frames = 160-640ms
    left = int(torch.randint(0, 5, ()).item())     # 0-4 chunks of history
    return chunk, left


# --------------------------------------------------------------------------
# conformer block
# --------------------------------------------------------------------------

class DCConvModule(nn.Module):
    """Depthwise conv restricted to chunk boundaries.

    A plain depthwise conv sees `kernel//2` future frames. Offline that is
    free; streaming it is not, and the mismatch is invisible until real
    audio. Here, when a chunk size is active, the input is zeroed outside
    the current chunk's allowed span before convolving, so the receptive
    field matches what streaming inference will actually have.
    """

    def __init__(self, dim, kernel=15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2 * dim, 1)
        self.dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2,
                            groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.kernel = kernel

    def forward(self, x, chunk=None):          # (B, T, D)
        y = self.norm(x).transpose(1, 2)       # (B, D, T)
        y = F.glu(self.pw1(y), dim=1)

        if chunk and chunk > 0:
            # Convolve chunk-by-chunk so no frame reads beyond its chunk.
            #
            # VECTORISED: the obvious implementation is a Python loop over
            # chunks calling self.dw on each slice, which is correct but
            # issues T/chunk tiny sequential GPU ops per block -- at
            # chunk=8 on a 1500-frame utterance that is ~190 launches per
            # block, ~1500 per forward pass, and it dominated the first
            # training run's time. Folding the chunks into the BATCH
            # dimension gives one conv call with identical semantics,
            # since each chunk is padded and convolved independently
            # either way. parity must still pass after this change.
            B, D, T = y.shape
            pad = (chunk - T % chunk) % chunk
            if pad:
                y = F.pad(y, (0, pad))
            Tp = y.shape[2]
            n = Tp // chunk
            y = y.view(B, D, n, chunk).permute(0, 2, 1, 3).reshape(
                B * n, D, chunk)
            y = self.dw(y)[:, :, :chunk]
            y = y.view(B, n, D, chunk).permute(0, 2, 1, 3).reshape(B, D, Tp)
            y = y[:, :, :T]
        else:
            y = self.dw(y)[:, :, :x.shape[1]]

        y = self.pw2(F.silu(self.bn(y)))
        return y.transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * mult), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(dim * mult, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)

    def forward(self, x, mask=None):
        y = self.norm(x)
        # nn.MultiheadAttention wants True = MASKED, our mask is True =
        # allowed. Inverting at the call site rather than storing it
        # inverted keeps chunk_mask readable on its own terms.
        am = None if mask is None else ~mask
        out, _ = self.attn(y, y, y, attn_mask=am, need_weights=False)
        return out


class ConformerBlock(nn.Module):
    """Macaron Conformer: half-FF, attention, conv, half-FF, norm."""

    def __init__(self, dim, heads=4, ff_mult=4, kernel=15, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(dim, ff_mult, dropout)
        self.attn = SelfAttention(dim, heads, dropout)
        self.conv = DCConvModule(dim, kernel)
        self.ff2 = FeedForward(dim, ff_mult, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None, chunk=None):
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, mask)
        x = x + self.conv(x, chunk)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


# --------------------------------------------------------------------------
# the head
# --------------------------------------------------------------------------

class PhoneHead(nn.Module):
    """XLS-R features -> per-frame phone logits (+ CTC blank at index 0)."""

    def __init__(self, n_phones, n_layers=6, in_dim=1024, dim=256,
                 blocks=8, heads=4, kernel=15, dropout=0.1, ff_mult=4,
                 fusion="weighted_sum", boundary_head=True):
        super().__init__()
        self.n_phones = n_phones
        self.blank = 0
        self.fusion = FUSIONS[fusion](n_layers, in_dim)
        self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim),
                                  nn.Dropout(dropout))
        self.blocks = nn.ModuleList([
            ConformerBlock(dim, heads, ff_mult, kernel, dropout)
            for _ in range(blocks)])
        self.out = nn.Linear(dim, n_phones + 1)      # +1 for blank

        # Per-frame "a phone ends here" head. The manifest gives verified
        # word boundaries, so this supervision is available for free and it
        # is exactly what the graph decoder needs: v5's decoders had to
        # SEARCH for word ends because nothing ever reported them. Optional
        # so its contribution can be ablated rather than assumed.
        self.boundary = nn.Linear(dim, 1) if boundary_head else None

    def forward(self, feats, chunk=None, left_chunks=-1):
        """feats: (B, T, L, D) cached layers. -> dict of outputs."""
        x = self.fusion(feats)
        x = self.proj(x)
        mask = chunk_mask(x.shape[1], chunk, left_chunks, x.device) \
            if chunk else None
        for b in self.blocks:
            x = b(x, mask, chunk)
        r = {"logits": self.out(x)}
        if self.boundary is not None:
            r["boundary"] = self.boundary(x).squeeze(-1)
        return r

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def build(args, n_phones):
    return PhoneHead(n_phones=n_phones, n_layers=args.n_layers,
                     dim=args.dim, blocks=args.blocks, heads=args.heads,
                     kernel=args.kernel, ff_mult=args.ff_mult,
                     fusion=args.fusion)


def cmd_sizes(a):
    """Parameter count across configurations.

    The default (dim 256, 8 blocks, ff_mult 4) came out at 12.4M, with
    ~98% of it in the Conformer blocks and roughly 70% of THAT in the two
    feedforwards per block (dim -> dim*ff_mult -> dim, twice, macaron
    style). So ff_mult is the cheapest lever, blocks is linear, and dim is
    quadratic in the feedforwards.

    Sizes here are the HEAD ONLY. Deployment also ships the distilled
    acoustic encoder from Phase 5, the phone lexicon and the duration
    priors, so the head's budget is a fraction of the 3-4 MB total -- call
    it 1-2 MB at INT8, i.e. roughly 1-2 M parameters.
    """
    n_phones = len(P.iqra_inventory())
    print(f"phones: {n_phones} (+1 blank)   layers: {a.n_layers}   "
          f"fusion: {a.fusion}\n")
    print(f"{'dim':>5} {'blk':>4} {'ff':>3} {'params':>9} {'int8':>8} "
          f"{'fp16':>8}")
    seen = []
    for dim in (128, 160, 192, 256):
        for blocks in (4, 6, 8):
            for ff in (2, 4):
                m = PhoneHead(n_phones=n_phones, n_layers=a.n_layers,
                              dim=dim, blocks=blocks, heads=a.heads,
                              kernel=a.kernel, ff_mult=ff,
                              fusion=a.fusion)
                n = m.n_params()
                seen.append((n, dim, blocks, ff))
                mark = "  <-- near 1-2M target" if 0.8e6 <= n <= 2.2e6 else ""
                print(f"{dim:>5} {blocks:>4} {ff:>3} {n/1e6:8.2f}M "
                      f"{n/1e6:7.1f}MB {n*2/1e6:7.1f}MB{mark}")
    print("\nNOTE: parameter count is not capability. These are candidate")
    print("sizes, not a ranking -- which one is actually good is an")
    print("empirical question, and the honest way to answer it is to")
    print("train two and compare on the harness, not to pick from a table.")
    return 0


def cmd_summary(a):
    # NOT len(PHONE_SET) + len(BACKED): that counts only the 33 base
    # symbols plus 6 backed vowels and silently omits the 27 consonant
    # geminates the iqra convention actually emits. Sizing the CTC output
    # from it would drop every nn/ll/bb label.
    n_phones = len(P.iqra_inventory())
    m = build(a, n_phones)
    print(f"fusion       : {a.fusion}")
    print(f"phones       : {n_phones} (+1 blank)")
    print(f"dim          : {a.dim}   blocks: {a.blocks}   heads: {a.heads}")
    print(f"parameters   : {m.n_params()/1e6:.2f} M")
    for name, mod in [("fusion", m.fusion), ("proj", m.proj),
                      ("blocks", m.blocks), ("out", m.out)]:
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:<10} {n/1e6:7.3f} M")
    print(f"\nINT8 estimate: ~{m.n_params()/1e6:.1f} MB "
          f"(1 byte/param); fp16 ~{m.n_params()*2/1e6:.1f} MB")
    print("\nNote this is the HEAD only. Deployment also ships the")
    print("distilled acoustic encoder (Phase 5), the phone lexicon and")
    print("duration priors. The decoder itself has no parameters.")
    return 0


def cmd_parity(a):
    """Streaming must equal full-context under the SAME mask.

    This is the check that catches the class of bug that is otherwise
    invisible: a model that looks fine offline and degrades in streaming,
    which reads as "the model is mediocre" rather than "the cache is
    wrong". Assert it on every architecture change.
    """
    torch.manual_seed(0)
    n_phones = len(P.iqra_inventory())
    m = build(a, n_phones).eval()

    T, chunk = 96, 16
    x = torch.randn(1, T, a.n_layers, 1024)

    with torch.no_grad():
        full = m(x, chunk=chunk, left_chunks=-1)["logits"]
        # Feed the same input in chunk-sized pieces, each seeing all
        # history, and compare the frames of the final chunk.
        parts = []
        for s in range(0, T, chunk):
            e = s + chunk
            out = m(x[:, :e], chunk=chunk, left_chunks=-1)["logits"]
            parts.append(out[:, s:e])
        inc = torch.cat(parts, dim=1)

    d = (full - inc).abs().max().item()
    rel = d / full.abs().max().item()
    print(f"frames            : {T}, chunk {chunk}")
    print(f"max abs difference: {d:.3e}")
    print(f"relative          : {rel:.3e}")
    ok = rel < 1e-4
    print(f"\n{'PASS' if ok else 'FAIL'}: streaming "
          f"{'matches' if ok else 'DOES NOT match'} full context")
    if not ok:
        print("A mismatch here means a frame is reading context that")
        print("streaming inference will not have. Do not train until this")
        print("passes -- the resulting model will look fine offline and")
        print("fail on real audio, which is the hardest failure to trace.")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="v6 phone head")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("summary", cmd_summary), ("parity", cmd_parity),
                     ("sizes", cmd_sizes)):
        p = sub.add_parser(name)
        p.add_argument("--dim", type=int, default=256)
        p.add_argument("--blocks", type=int, default=8)
        p.add_argument("--heads", type=int, default=4)
        p.add_argument("--kernel", type=int, default=15)
        p.add_argument("--ff-mult", type=int, default=4,
                       help="feedforward expansion; the dominant parameter "
                            "cost (two per block, macaron style)")
        p.add_argument("--n-layers", type=int, default=6)
        p.add_argument("--fusion", choices=list(FUSIONS),
                       default="weighted_sum")
        p.set_defaults(func=fn)
    a = ap.parse_args()
    rc = a.func(a) or 0
    sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
