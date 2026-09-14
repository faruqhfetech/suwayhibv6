#!/usr/bin/env python3
"""
Suwayhib v6 -- word-boundary head evaluation on held-out reciters.

    python -m suwayhib.tools.bnd_eval run --ckpt runs/head_v2/best.pt
    python -m suwayhib.tools.bnd_eval run --ckpt runs/stream/stage1_best.pt --n 80
    python -m suwayhib.tools.bnd_eval run --ckpt A.pt --ckpt B.pt   # compare

WHAT THIS ANSWERS
-----------------
Whether a checkpoint's boundary head actually places word ends, on audio
it has never seen -- and it is the only way to compare that number
ACROSS TRAINERS, because train.py and train_stream*.py disagree about
almost everything else they report.

It needs no testset/ and no recs/ (both gitignored, so neither exists on
a fresh clone or a Kaggle box). Reference audio is resolved exactly the
way the trainer resolves it: a local reference_library/ file if present,
otherwise fetched from everyayah.com and cached.

WHY R-VALUE AND NOT THE LOSS
----------------------------
The training-time `bnd` column is a mask-mean of pos_weight-weighted BCE,
which at logits ~0 evaluates to ln2 * (1 + 19p) in the positive-frame
density p. Over this corpus p spans 0.0057-0.042, so batch composition
alone moves it 0.77 -> 1.25 -- wider than any trend it could show. The
metrics here are tolerance-based and density-invariant, so they compare
across checkpoints, trainers and corpora. See core/boundary.py for the
derivation and the citations (Rasanen et al. 2009 for R-value;
arXiv:2411.10423 for the 40ms word-boundary tolerance and middle-frame
cluster collapsing; arXiv:2212.06387 for sweeping the threshold on held-
out data rather than fixing it at 0.5).

WHY HELD-OUT RECITERS SPECIFICALLY
----------------------------------
--holdout defaults to Husary,Minshawy: the reciters testset/meta.json
declares as the holdout split, and the ones train.py's reciter-held-out
split keeps out of training. Scoring a boundary head on reciters it
trained on measures memorisation. Note this is a RECITER hold-out, not
an ayah hold-out -- train_stream's own --val-items slice is a different
(and much smaller) thing, so numbers from the two are not interchangeable.

WHAT A GOOD RESULT LOOKS LIKE
-----------------------------
R-value clearly above ~0.3 -- randomly placed boundaries score about that
by chance at this tolerance, so 0.3 is the floor, not a baseline. OS near
0 means the head emits about as many boundaries as exist; large negative
OS is under-segmentation (it is missing ends), large positive is
over-segmentation (it is firing everywhere, which F1 would partly
forgive and R-value does not).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from ..core import boundary as BND
from ..pipeline import score as S
from ..train import train_stream_bndmetrics as T


def collect(scorer, manifest, lib, text, everyayah_cache, holdout, n,
             seed, max_frames):
    """-> (prob, target, mask) arrays over held-out reference items.

    Shuffled rather than walked in sorted key order: sorted keys are
    (reciter, surah, ayah), so a sorted walk would take the first N
    ayahs of one surah, whose word density is strongly autocorrelated
    (lag-1 +0.669 over this manifest). That biases the density of the
    evaluation slice, and the metrics below are density-invariant only
    in the sense that they do not depend on it MECHANICALLY -- a slice
    of unusually short or long ayahs is still a biased sample of the
    task. Shuffling costs nothing and removes the question.
    """
    probs, tgts, kept, skipped = [], [], 0, 0
    for it in T.stream_local(manifest, lib, text,
                             everyayah_cache=everyayah_cache,
                             shuffle_seed=seed):
        if holdout and not any(h.lower() in it["reciter"].lower()
                               for h in holdout):
            continue
        p = scorer.boundary_probs(it["audio"])
        if p is None:
            return None, None, None
        nf = min(len(p), max_frames)
        t = np.zeros(nf)
        ends = [e for e in it["ends"] if 0 <= e < nf]
        if not ends:
            # No word end lands inside the scored region -- contributes
            # only negatives, which would inflate precision denominators
            # without ever being matchable. Dropped, and counted.
            skipped += 1
            continue
        for e in ends:
            t[e] = 1.0
        probs.append(p[:nf])
        tgts.append(t)
        kept += 1
        if kept >= n:
            break
    if not probs:
        return None, None, None
    L = max(len(p) for p in probs)
    P_ = np.zeros((len(probs), L))
    Tg = np.zeros((len(probs), L))
    Mk = np.zeros((len(probs), L))
    for i, (p, t) in enumerate(zip(probs, tgts)):
        P_[i, :len(p)] = p
        Tg[i, :len(t)] = t
        Mk[i, :len(p)] = 1.0     # mask = the item's real frames only, so
                                 # right-padding is never scored
    if skipped:
        print(f"  ({skipped} items skipped: no word end inside "
              f"--max-frames {max_frames})")
    return P_, Tg, Mk


def cmd_run(a):
    holdout = [x.strip() for x in a.holdout.split(",") if x.strip()]
    tols = [int(x) for x in a.tol_frames.split(",") if x.strip()]
    rows = []
    for ckpt in a.ckpt:
        print(f"\n=== {ckpt} ===")
        scorer = S.LiveScorer(ckpt, a.device)
        P_, Tg, Mk = collect(scorer, a.manifest, a.lib, a.text,
                             a.everyayah_cache, holdout, a.n, a.seed,
                             a.max_frames)
        if P_ is None:
            print("  no boundary head in this checkpoint (or no items "
                  "matched --holdout); nothing to score")
            continue
        print(f"  {P_.shape[0]} items"
              f"{' from ' + '/'.join(holdout) if holdout else ''}, "
              f"{int(Tg.sum())} word boundaries")
        stored = scorer.bnd
        for tol in tols:
            r = BND.metrics(P_, Tg, Mk, tol_frames=tol)
            if r is None:
                continue
            print(f"  tol +/-{tol}fr ({BND.frames_to_ms(tol):3d}ms): "
                  f"P {r['precision']:.3f} R {r['recall']:.3f} "
                  f"F1 {r['f1']:.3f} OS {r['os']:+.3f} "
                  f"R-value {r['rvalue']:.4f}   "
                  f"(best th {r['threshold']:.2f}, "
                  f"n_hyp {r['n_hyp']}, n_ref {r['n_ref']})")
            rows.append((ckpt, tol, r))
            if stored and tol == stored.get("tol_frames"):
                # The checkpoint's own tuned threshold came from
                # train_stream's --val-items slice, which is a different
                # (smaller, ayah-level) hold-out than this reciter-level
                # one. Reporting both says whether that threshold
                # transfers to unseen voices, which is the thing that
                # actually matters at inference.
                m = BND.metrics(P_, Tg, Mk, tol_frames=tol,
                                thresholds=[stored["threshold"]])
                print(f"      at the checkpoint's STORED th "
                      f"{stored['threshold']:.2f}: "
                      f"P {m['precision']:.3f} R {m['recall']:.3f} "
                      f"R-value {m['rvalue']:.4f}   "
                      f"(tuned on its own val slice, where it scored "
                      f"{stored['rvalue']:.4f})")
    if len(a.ckpt) > 1 and rows:
        print("\n=== summary (R-value) ===")
        for tol in tols:
            # Last TWO path components: every run writes the same
            # basenames (best.pt / last.pt), so the basename alone makes
            # a comparison table unreadable.
            line = "   ".join(
                f"{'/'.join(Path(c).parts[-2:])} {r['rvalue']:.4f}"
                for c, t_, r in rows if t_ == tol)
            if line:
                print(f"  +/-{tol}fr: {line}")
    print("\nChance floor is ~0.30 at these tolerances -- randomly placed "
          "boundaries\nscore about that. Read R-value against 0.30, not "
          "against 0.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="boundary-head evaluation on held-out reciters")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ckpt", action="append", required=True,
                   help="checkpoint to score; repeat to compare several")
    r.add_argument("--n", type=int, default=40,
                   help="held-out items to score. 40 gives ~150 word "
                        "boundaries; recall then carries roughly +/-0.04 "
                        "standard error, so treat differences smaller "
                        "than that as noise")
    r.add_argument("--holdout", default="Husary,Minshawy",
                   help="substring match on reciter; empty string scores "
                        "every reciter (NOT held out -- memorisation)")
    r.add_argument("--tol-frames", default="1,2,3",
                   help="match tolerances in frames (20ms each). 2 = 40ms "
                        "is the word-boundary convention of "
                        "arXiv:2411.10423")
    r.add_argument("--max-frames", type=int, default=1200)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--device", default="cuda")
    r.add_argument("--manifest", default="manifest/manifest_clean.csv")
    r.add_argument("--lib", default="reference_library")
    r.add_argument("--text", default="texts/quran-simple-plain.txt")
    r.add_argument("--everyayah-cache", default="cache/everyayah")
    r.set_defaults(func=cmd_run)
    a = ap.parse_args()
    sys.exit(a.func(a) or 0)


if __name__ == "__main__":
    main()
