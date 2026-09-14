"""Word-boundary post-processing and evaluation metrics.

WHY THIS MODULE EXISTS, AND WHY IT LIVES IN core/.

The boundary head (core/model.py's PhoneHead.boundary, a single Linear
producing one logit per frame) predicts word-END positions. Turning those
logits into actual boundaries, and scoring them, is needed in two places
that must not disagree:

  * train/train_stream_bndmetrics.py -- to monitor the head during
    training and to pick stage{N}_best.pt.
  * pipeline/score.py -- to USE the head at inference, at the operating
    point training chose.

The import direction in this project is train/ -> pipeline/ -> core/
(train/phase1.py imports pipeline.score; pipeline/ imports only core/).
So pipeline/score.py cannot import the trainer, and the shared code has
to sit here or be duplicated. Duplicating a threshold-and-tolerance
convention is exactly how an inference path silently drifts from the
metric that selected the checkpoint, so: one copy, here.

WHY THESE METRICS AND NOT THE LOSS.

boundary_loss (in the trainer) is a mask-mean of pos_weight-weighted BCE.
Weighted BCE is the standard *training* objective for frame-level
boundary detection and is not in question. It is a bad *monitoring*
signal, for a reason that is arithmetic rather than taste: at logits ~ 0,

    (l * mask).sum() / mask.sum()  ==  ln2 * (1 - p + pos_weight * p)
                                   ==  ln2 * (1 + 19p)   [pos_weight=20]

where p is the density of positive (word-end) frames. torch's pos_weight
scales ONLY the positive term (at logit 0: target=1 -> 20*ln2 = 13.86,
target=0 -> ln2 = 0.693), so that identity is exact. Measured across all
4,424 items of manifest/manifest_clean.csv, p spans 0.0057 to 0.042 --
i.e. batch composition ALONE moves the printed loss 0.77 -> 1.25. Any
real trend in head quality smaller than that is invisible underneath it.

The speech-segmentation literature does not monitor the loss at all:

  * Rasanen, Laine & Altosaar, "An improved speech segmentation quality
    measure: the R-value", Interspeech 2009 -- introduces R-value
    precisely because "increases in phone boundary location detection
    rates are often due to increased over-segmentation levels and not to
    algorithmic improvements": a detector that fires constantly buys
    recall, and therefore F1, for free.
  * "Back to Supervision: Boosting Word Boundary Detection through Frame
    Classification" (arXiv:2411.10423) -- the closest published analogue
    to this setup (supervised frame classification for WORD boundaries).
    Reports P/R/F/OS/R-value at 40 ms tolerance on Buckeye, collapses
    boundary clusters to their middle frame, and selects checkpoints on
    validation R-value ("we saved the weights of models at the best
    validation R-value"). It reports no loss curves.
  * "Towards trustworthy phoneme boundary detection with autoregressive
    model and improved evaluation metric" (arXiv:2212.06387) -- the
    decision threshold is grid-searched on validation, not fixed at 0.5,
    and naive boundary counting lets insertions near a true boundary
    cancel deletions and report misleadingly high precision AND recall.

All of that is implemented below. Unlike the loss, these numbers are
invariant to positive-frame density, so they are comparable across
checkpoints without knowing what happened to be in the batch.
"""

import math

import numpy as np

# Frame rate of the XLS-R features these boundaries are indexed in.
# Matches pipeline/extract.py's FRAMES_PER_SEC (1000 // FRAME_MS) and the
# copies in decode.py / demo.py / the trainer; one frame is 20 ms.
FRAMES_PER_SEC = 50

# Threshold grid for the sweep. Deliberately NOT centred on 0.5: with
# pos_weight=20 the head is trained to be miscalibrated towards
# predicting boundaries, so its useful operating point sits well away
# from 0.5 and moves during training (arXiv:2212.06387 grid-searches it).
DEFAULT_THRESHOLDS = tuple(round(0.05 * k, 2) for k in range(1, 20))

# Default match tolerance. 2 frames = 40 ms, the word-boundary tolerance
# arXiv:2411.10423 uses on Buckeye. 1 frame = 20 ms is the stricter TIMIT
# phoneme convention.
DEFAULT_TOL_FRAMES = 2


def frames_to_ms(n_frames):
    return int(round(1000 * n_frames / FRAMES_PER_SEC))


def collapse_runs(prob, valid, threshold):
    """-> list of predicted boundary frame indices for ONE item.

    `prob` and `valid` are 1-D over frames. A contiguous run of
    above-threshold frames is collapsed to its MIDDLE frame rather than
    emitting one boundary per frame -- the post-processing of
    arXiv:2411.10423. This is not cosmetic: without it one soft
    5-frame-wide peak counts as 5 insertions, which crushes precision
    and (through OS = R/P - 1) the R-value, so the metric would end up
    measuring peak WIDTH instead of peak PLACEMENT.
    """
    hot = (np.asarray(prob) >= threshold) & np.asarray(valid, dtype=bool)
    out, i, n = [], 0, len(hot)
    while i < n:
        if hot[i]:
            j = i
            while j + 1 < n and hot[j + 1]:
                j += 1
            out.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    return out


def match_boundaries(ref, hyp, tol):
    """-> number of reference boundaries matched ONE-TO-ONE to a
    predicted boundary within +/- tol frames.

    The one-to-one constraint is the point. A many-to-one count lets
    insertions sitting near a true boundary cancel deletions elsewhere
    and report misleadingly high precision AND recall at the same time
    (the failure mode arXiv:2212.06387 flags). Greedy nearest-first
    matching; each hypothesis is consumable exactly once.
    """
    used = [False] * len(hyp)
    hits = 0
    for r in ref:
        best, best_d = -1, tol + 1
        for k, h in enumerate(hyp):
            if used[k]:
                continue
            d = abs(h - r)
            if d <= tol and d < best_d:
                best, best_d = k, d
        if best >= 0:
            used[best] = True
            hits += 1
    return hits


def r_value(recall, precision):
    """Rasanen et al. 2009, in the 0..1-normalised form the modern papers
    use. OS = R/P - 1 is the over-segmentation rate; r1 is the distance
    to the ideal (R=1, OS=0) operating point and r2 the distance to the
    OS = R - 1 line. The only way to score 1.0 is perfect recall with
    zero over-segmentation -- which is why this, and not F1, is the
    checkpoint-selection metric.

    -> (rvalue, os)
    """
    if precision <= 0.0:
        # No predictions, or none correct: OS divides by zero. Report the
        # worst score rather than raising -- an untrained head genuinely
        # sits here for the first few hundred steps, and a checkpoint
        # callback must not crash there.
        return 0.0, -1.0
    os_ = recall / precision - 1.0
    r1 = math.sqrt((1.0 - recall) ** 2 + os_ ** 2)
    r2 = (-os_ + recall - 1.0) / math.sqrt(2.0)
    return 1.0 - (abs(r1) + abs(r2)) / 2.0, os_


def metrics(prob, target, valid, tol_frames=DEFAULT_TOL_FRAMES,
            thresholds=DEFAULT_THRESHOLDS):
    """Tolerance-based boundary P/R/F1/OS/R-value over a batch, swept
    over `thresholds`.

    prob/target/valid are (B, T) array-likes: predicted PROBABILITIES
    (already sigmoided), the 0/1 boundary targets, and the valid-frame
    mask. `valid` -- not a length vector -- defines the scored region, so
    this counts frames on exactly the footing the training loss does and
    never scores a prediction made in right-padding.

    Counts are pooled over items BEFORE the metrics are computed
    (micro-averaging), so one short item with two word ends cannot swing
    the result the way it swings the loss.

    -> dict for the best-R-value threshold, with the full sweep under
    "sweep"; or None if there is nothing to score against.
    """
    prob = np.asarray(prob)
    target = np.asarray(target)
    valid = np.asarray(valid) > 0
    B = prob.shape[0]

    sweep = []
    for th in thresholds:
        n_ref = n_hyp = hits = 0
        for i in range(B):
            if not valid[i].any():
                continue
            ref = np.flatnonzero((target[i] > 0.5) & valid[i]).tolist()
            hyp = collapse_runs(prob[i], valid[i], th)
            n_ref += len(ref)
            n_hyp += len(hyp)
            hits += match_boundaries(ref, hyp, tol_frames)
        if n_ref == 0:
            return None
        rec = hits / n_ref
        prec = hits / n_hyp if n_hyp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rv, os_ = r_value(rec, prec)
        sweep.append({"threshold": float(th), "precision": prec,
                      "recall": rec, "f1": f1, "os": os_, "rvalue": rv,
                      "n_ref": n_ref, "n_hyp": n_hyp, "hits": hits})
    if not sweep:
        return None
    best = max(sweep, key=lambda d: d["rvalue"])
    return {**best, "sweep": sweep, "tol_frames": tol_frames,
            "tol_ms": frames_to_ms(tol_frames)}


def operating_point(ck):
    """Read a checkpoint's stored boundary operating point.

    train_stream_bndmetrics.py writes the tuned threshold into
    stage{N}_best.pt under "bnd_metrics" (see its _checkpoint). Without
    that, a `best` checkpoint is unusable at its own operating point --
    whoever loads it would have to re-derive the threshold, and 0.5 is
    NOT it.

    -> the stored dict, or None for checkpoints predating this (the
    original train_stream.py, and train.py's). Callers must handle None
    rather than assuming 0.5: an arbitrary threshold on a head trained
    with pos_weight=20 over-segments badly, which is the specific error
    r_value exists to make visible.
    """
    if not isinstance(ck, dict):
        return None
    m = ck.get("bnd_metrics")
    return m if isinstance(m, dict) and "threshold" in m else None
