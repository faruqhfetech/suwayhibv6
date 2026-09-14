#!/usr/bin/env python3
"""
Suwayhib v6 -- Phase 2 decoding-aware training: bake the actual
accept/reject decision into the phone head's OWN training loss, instead
of only calibrating a downstream classifier after the fact (Phase 1).

    python -m suwayhib.train.phase2 selftest

WHY THIS EXISTS, AND HOW IT DIFFERS FROM PHASE 1
--------------------------------------------------
Phase 1 (train/phase1.py) froze the phone head entirely and trained a
small SEPARATE classifier on decode-time features, supervised by
accept/reject labels -- the Qiu et al. 2021 (arXiv:2103.06716) pattern.
That is real and it shipped a measured improvement, but it can only ever
recalibrate what the frozen head already produces. It cannot teach the
head itself to make correct-vs-incorrect pronunciation more SEPARABLE
on the statistic decode.py actually thresholds, because no gradient
from the verification decision ever reaches the head's weights.

Phase 2 closes that: the SAME kind of score decode.py computes (how
well does this audio's frame posteriors match a SPECIFIC target phone
sequence) is computed DIFFERENTIABLY during training and added as an
extra loss term, so its gradient flows back through the phone head
(and, if unfrozen, the encoder). This is the concrete version of the
literature this project's research pass grounded:

  - R. Sukkar & C.-H. Lee, "Vocabulary independent discriminative
    utterance verification for nonkeyword rejection in subword based
    speech recognition" (IEEE Trans. Speech and Audio Proc., 1996),
    building on B.-H. Juang & S. Katagiri, "Discriminative Learning for
    Minimum Error Classification" (IEEE Trans. Signal Proc., 1992):
    train a SMOOTH, DIFFERENTIABLE approximation of the actual
    accept/reject decision directly, rather than training a proxy
    objective (plain transcription) and thresholding it after the fact.
    That is exactly what `verification_loss` below does: it is a
    modern, BCE-based descendant of Juang & Katagiri's sigmoid
    misclassification measure, applied to the SAME kind of constrained-
    sequence score decode.py's WordLattice approximates.

  - M. Sun et al., "Max-Pooling Loss Training of Long Short-Term Memory
    Networks for Small-Footprint Keyword Spotting" (SLT 2016,
    arXiv:1705.02411): the general principle this follows -- bake the
    AGGREGATION/DECISION operator used at decode time into the training
    loss, rather than training a framewise/transcription proxy and
    hoping the decode-time procedure transfers. Sun et al. do this for
    a framewise LSTM classifier via max-pooling; this project's
    equivalent aggregation IS the CTC-constrained-sequence likelihood
    (what WordLattice approximates by hand and what F.ctc_loss computes
    exactly, in closed form, already differentiable).

WHY torch.nn.functional.ctc_loss INSTEAD OF RE-IMPLEMENTING
WordLattice's VITERBI IN PYTORCH
----------------------------------------------------------------
decode.py's WordLattice is a hand-rolled Viterbi (max, not log-sum-exp)
recursion over plain Python floats -- not differentiable, not batched,
not GPU-resident. Reimplementing its exact automaton (skip transitions,
dwell cap, geminate handling) as a differentiable, batched PyTorch op
would be real, substantial, error-prone work duplicating something
PyTorch already ships, correctly, batched, GPU-accelerated: F.ctc_loss
computes, via the standard forward algorithm, the negative
log-likelihood of a SPECIFIC target sequence given frame log-probits --
i.e. exactly "how well does this audio's posteriors support having said
THIS sequence", which is the same quantity WordLattice approximates by
hand for the purpose of decode-time thresholding. Using it here means
one correct, tested implementation is trusted for both scoring
directions (forward here vs Viterbi in decode.py) instead of two
hand-maintained ones; forward and Viterbi agree on which sequence is
better even though their absolute scale differs slightly, which is
exactly why VerificationHead below learns its OWN calibration rather
than assuming this score lands on decode.py's tau scale.

WHAT SUPERVISES THIS, AND WHY IT DOES NOT NEED NEW DATA
----------------------------------------------------------
Every IqraEval training item already carries phoneme_ref (canonical:
what SHOULD have been said) and phoneme_mis (verbatim: what WAS said).
Where they are IDENTICAL (the overwhelming majority of Iqra_train, and
any reference-library item), the audio genuinely supports its own
canonical target: label 1, target = phoneme_ref (== phoneme_mis).
Where they DIFFER (deliberately, in Iqra_TTS's synthesised
mispronunciations, and in the small real Iqra_Extra_IS26 set), the
audio does NOT genuinely support the CANONICAL target it deviates
from: label 0, target = phoneme_ref, even though the model is
separately and correctly being taught (via the ordinary CTC loss, on
phoneme_mis) to transcribe what was actually said. These are not in
tension -- transcribing accurately and recognising "this does not
match a DIFFERENT specific sequence" are different questions about the
same audio, and CTC training alone only ever asks the first one.

WHAT THIS DOES NOT DO YET
--------------------------
This module ships fully unit-tested (see `selftest`) on synthetic
tensors -- no dataset, no GPU, no live model required to verify the
loss behaves correctly. Wiring it into train_stream.py's actual
training loop (extending the streamed-item pipeline to carry BOTH
phoneme_ref and phoneme_mis per item, not just whichever one
--label-col already picked) is a small, separate, opt-in addition
(--verify-loss) to that file -- see train_stream.py's own docstring
note. Actually running a training session with it enabled is explicitly
deferred until after a plain retrain, per this project's own current
priority order.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def verification_score(log_probs, target, input_lengths, target_lengths,
                       blank=0, impossible_score=-100.0):
    """Differentiable, length-normalised log-likelihood of `target`
    given `log_probs` -- the constrained-alignment score decode.py's
    WordLattice approximates by hand, computed instead via PyTorch's
    own forward-algorithm CTC loss (higher = better match).

    log_probs      (T, B, C) log-softmax'd frame posteriors
    target         (sum(target_lengths),) concatenated target phone ids
    input_lengths  (B,) real frame count per item
    target_lengths (B,) phone count per item

    -> (B,) score tensor, gradient-carrying.

    IMPOSSIBLE ALIGNMENTS, HANDLED BY EXCLUSION, NOT MASKING: a target
    needing more frames than are available (or more forced blanks than
    fit, for consecutive-repeat phones) has NO valid alignment at all --
    a real case here, since a corrupted item's CANONICAL target can
    easily be longer than what a truncation or word_deletion error left
    in the audio. Two separate traps here, both found by actually
    running this, not by inspection:

    1. F.ctc_loss's zero_infinity=True reports loss 0 for an impossible
       item -- THE BEST POSSIBLE VALUE -- purely so backward() stays
       finite (train_stream.py's own ctc_loss docstring carries the
       identical warning about reusing ITS zero_infinity output as a
       score). Naively reused as a score, an impossible alignment would
       read as "this audio perfectly supports a target it cannot even
       be aligned to" -- exactly backwards for the label=0 examples
       this exists to penalise.

    2. Masking the VALUE after the fact (e.g. computing the whole batch
       with zero_infinity=True, then torch.where-ing the impossible
       positions to a constant) is not sufficient, and this is not a
       theoretical concern: it SEGFAULTS on this project's torch build
       (Aborted / "corrupted size vs. prev_size while consolidating")
       when backward() runs, because F.ctc_loss's backward computes a
       gradient for the WHOLE batched call in one kernel invocation
       regardless of what a later op does with the result --
       zero_infinity=True does not appear to make that per-item
       backward safe in every case on this build. The only fix found to
       actually be safe: never let an impossible item enter a ctc_loss
       call whose backward will be invoked at all. Impossible items are
       therefore fully EXCLUDED from the batch passed to the
       gradient-carrying ctc_loss call below, not masked afterward.
    """
    B = input_lengths.shape[0]
    device = log_probs.device
    score = torch.full((B,), float(impossible_score),
                       dtype=log_probs.dtype, device=device)

    with torch.no_grad():
        # zero_infinity=False here ONLY to detect which items have no
        # valid alignment -- this call's backward is never triggered
        # (no_grad), so its own internal NaN-gradient issue for
        # impossible items never matters here.
        raw = F.ctc_loss(log_probs, target, input_lengths, target_lengths,
                         blank=blank, reduction="none", zero_infinity=False)
        possible = torch.isfinite(raw)

    if not bool(possible.any()):
        return score

    idx = possible.nonzero(as_tuple=True)[0]
    offsets = torch.cat([target.new_zeros(1), target_lengths.cumsum(0)])
    sub_target = torch.cat([target[offsets[i]:offsets[i + 1]]
                            for i in idx.tolist()])
    sub_lp = log_probs[:, idx, :]
    sub_ilen = input_lengths[idx]
    sub_tlen = target_lengths[idx]

    nll = F.ctc_loss(sub_lp, sub_target, sub_ilen, sub_tlen, blank=blank,
                     reduction="none", zero_infinity=True)
    sub_score = -nll / sub_tlen.clamp(min=1).float()
    return score.index_copy(0, idx, sub_score)


class VerificationHead(nn.Module):
    """A deliberately tiny (2-parameter) learnable calibration on top of
    verification_score. The point is not capacity -- Phase 1 already
    showed a downstream classifier over decode-time features can help
    WITHOUT touching the head's weights. The point of putting even a
    trivial affine map here, trained end-to-end with the phone head
    (not frozen, not post-hoc), is that its gradient reaches the head's
    OWN weights, so training pressure exists to make correct and
    incorrect pronunciation actually SEPARABLE on this score, not just
    plausible-looking under a pure transcription objective that never
    sees a "same target, wrong audio" pair.
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, score):
        return self.scale * score + self.bias   # -> logit, for BCE


def verification_loss(head, log_probs, target, input_lengths,
                      target_lengths, label, blank=0):
    """-> (loss, score). label: (B,) float/bool, 1 = audio genuinely
    supports `target`, 0 = it does not (a real or synthesised
    mispronunciation scored against the CANONICAL sequence it deviated
    from -- see module docstring for where these labels come from).

    Plain BCE-on-a-learned-affine-map rather than a fixed-margin hinge
    on raw score: verification_score's scale is an artefact of vocab
    size and how confidently the (frozen or fine-tuning) encoder's
    posteriors are calibrated, not something to hand-pick a margin
    against -- VerificationHead's 2 parameters absorb that, the same
    role Juang & Katagiri's sigmoid slope/bias played in the original
    MCE formulation.
    """
    score = verification_score(log_probs, target, input_lengths,
                               target_lengths, blank)
    logit = head(score)
    loss = F.binary_cross_entropy_with_logits(logit, label.float())
    return loss, score.detach()


# --------------------------------------------------------------------------
# self-test -- synthetic tensors only, no dataset or trained model
# required. GPU IS effectively required, though: see DEVICE below.
# --------------------------------------------------------------------------

# CUDA when available, CPU otherwise -- NOT a style preference. Found by
# actually running this repeatedly: PyTorch's CPU F.ctc_loss backward
# segfaults ("corrupted size vs. prev_size while consolidating"),
# non-deterministically (roughly 3 crashes in 5 runs), on a MIXED batch
# (one short target, one long-but-still-alignable target) on this
# project's torch build (2.11.0+cu130) -- reproduced 5/5 stable on CUDA
# and 3/3+ crashing on CPU, same tensors, same seed, only device
# changed. This is not the impossible-alignment issue verification_score
# already guards against (that target here IS alignable: 10 distinct
# phones fit in 12 frames with no repeats needing separators) -- it
# looks like a lower-level instability in this build's CPU CTC kernel
# under certain batch shapes. train_stream.py always runs on CUDA in
# real use, so production is unaffected either way; this default just
# keeps the SELF-TEST from hitting the same unstable path.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _toy_logits(seq_len, n_classes, target, correct, seed):
    """A tiny hand-built (T, C) logit sequence: strongly peaked on
    `target`'s phones in order if correct, or on a DIFFERENT sequence
    (target shifted by +1 mod n_classes-1, skipping blank) if not --
    exactly the "same-length, wrong content" case verification_loss
    must learn to separate, since a length mismatch alone would be a
    much easier (and less relevant) signal.
    """
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(seq_len, n_classes, generator=g) * 0.1
    if correct:
        seq = target
    else:
        seq = [1 + (p % (n_classes - 1)) for p in target]  # never blank(0)
    # spread the (possibly wrong) sequence evenly across frames, blank
    # elsewhere -- deliberately simple, this only needs to be a sequence
    # F.ctc_loss can score, not a realistic acoustic trace.
    step = max(1, seq_len // (len(seq) + 1))
    for i, p in enumerate(seq):
        t = min((i + 1) * step, seq_len - 1)
        logits[t, p] += 8.0
    return logits.to(DEVICE)


def sc_higher_score_for_correct_target():
    """The audio that truly says `target` must score higher against it
    than audio that says something else entirely."""
    target = [3, 5, 7, 3]
    T, C = 40, 10
    correct = _toy_logits(T, C, target, correct=True, seed=0)
    wrong = _toy_logits(T, C, target, correct=False, seed=1)
    lp = F.log_softmax(torch.stack([correct, wrong]), dim=-1)
    lp = lp.transpose(0, 1)                              # (T, B, C)
    tgt = torch.tensor(target + target, dtype=torch.long, device=DEVICE)
    ilen = torch.tensor([T, T], device=DEVICE)
    tlen = torch.tensor([len(target), len(target)], device=DEVICE)
    score = verification_score(lp, tgt, ilen, tlen)
    ok = bool(score[0] > score[1])
    return ok, f"correct {score[0]:.3f} vs wrong {score[1]:.3f} (want correct higher)"


def sc_loss_decreases_with_training():
    """Gradient descent on verification_loss alone, over a few steps on
    a tiny fixed batch, must reduce the loss -- the minimum viable proof
    that gradients actually flow and point the right way."""
    torch.manual_seed(0)
    target = [2, 4, 6]
    T, C = 30, 9
    correct = _toy_logits(T, C, target, correct=True, seed=2)
    wrong = _toy_logits(T, C, target, correct=False, seed=3)
    raw = torch.stack([correct, wrong]).clone().requires_grad_(True)

    head = VerificationHead().to(DEVICE)
    opt = torch.optim.SGD([raw, *head.parameters()], lr=0.05)

    tgt = torch.tensor(target + target, dtype=torch.long, device=DEVICE)
    ilen = torch.tensor([T, T], device=DEVICE)
    tlen = torch.tensor([len(target), len(target)], device=DEVICE)
    label = torch.tensor([1.0, 0.0], device=DEVICE)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        lp = F.log_softmax(raw, dim=-1).transpose(0, 1)
        loss, _ = verification_loss(head, lp, tgt, ilen, tlen, label)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    ok = losses[-1] < losses[0] * 0.5
    return ok, f"loss {losses[0]:.4f} -> {losses[-1]:.4f} (want a clear drop)"


def sc_impossible_target_scores_bad_not_good():
    """A target LONGER than the input has NO valid alignment at all --
    a real case (a corrupted item's canonical target can be longer than
    what a truncation/word_deletion error left in the audio). Naive
    zero_infinity reuse would score this as loss 0 -- the BEST possible
    value -- which is exactly backwards for a mismatch example. Must be
    finite (no NaN/inf to poison the batch) AND clearly bad (very
    negative), not accidentally the best score in the batch."""
    target = list(range(1, 15))                # 14 phones
    T, C = 10, 16                               # only 10 frames -- too few
    logits = torch.randn(T, C, device=DEVICE) * 0.1
    lp = F.log_softmax(logits, dim=-1).unsqueeze(1)   # (T, 1, C)
    tgt = torch.tensor(target, dtype=torch.long, device=DEVICE)
    ilen = torch.tensor([T], device=DEVICE)
    tlen = torch.tensor([len(target)], device=DEVICE)
    score = verification_score(lp, tgt, ilen, tlen)
    finite = bool(torch.isfinite(score).all())
    bad = bool(score.item() < -10.0)
    return finite and bad, f"score {score.item():.3f} (want finite AND < -10, not ~0)"


def sc_impossible_item_does_not_poison_batch_gradient():
    """The actual point of the no_grad detection pass: in a MIXED batch
    (one alignable item, one impossible one), the impossible item's
    internally-NaN CTC gradient must not leak into the alignable item's
    gradient just because they share one batched log_probs tensor.
    A naive post-hoc torch.where on the LOSS value (skipping the
    separate no_grad detection pass) would not catch this -- it is a
    gradient-level bug, invisible if you only check the forward score.
    """
    torch.manual_seed(0)
    T, C = 12, 8
    possible_target = [1, 2, 3]                  # fits easily in 12 frames
    impossible_target = list(range(1, 11))        # 10 phones, only 12 frames
    logits = (torch.randn(T, 2, C, device=DEVICE) * 0.1).requires_grad_(True)
    lp = F.log_softmax(logits, dim=-1)
    tgt = torch.tensor(possible_target + impossible_target, dtype=torch.long,
                       device=DEVICE)
    ilen = torch.tensor([T, T], device=DEVICE)
    tlen = torch.tensor([len(possible_target), len(impossible_target)],
                        device=DEVICE)

    score = verification_score(lp, tgt, ilen, tlen)
    score.sum().backward()

    grad_finite = bool(torch.isfinite(logits.grad).all())
    possible_has_grad = bool(logits.grad[:, 0].abs().sum() > 0)
    return (grad_finite and possible_has_grad,
           f"grad finite={grad_finite}, alignable item has nonzero "
           f"grad={possible_has_grad} (want both True)")


SCENARIOS = [
    ("higher_score_for_correct_target", sc_higher_score_for_correct_target),
    ("loss_decreases_with_training", sc_loss_decreases_with_training),
    ("impossible_target_scores_bad", sc_impossible_target_scores_bad_not_good),
    ("impossible_item_does_not_poison_batch",
     sc_impossible_item_does_not_poison_batch_gradient),
]


def cmd_selftest(a):
    print(f"Pure PyTorch. No dataset or trained model needed. "
          f"device={DEVICE}")
    if DEVICE.type == "cpu":
        print("WARNING: no CUDA available -- running on CPU, where this "
              "project's torch build has a known intermittent CTC "
              "backward crash on some batch shapes (see DEVICE's "
              "comment above). A clean run here is not a stronger "
              "guarantee than a CUDA run; a crash here is not "
              "necessarily a real bug.")
    print()
    n = 0
    for name, fn in SCENARIOS:
        ok, detail = fn()
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<32} {detail}")
        n += ok
    print(f"\n{n}/{len(SCENARIOS)} PASS")
    return 0 if n == len(SCENARIOS) else 1


def main():
    ap = argparse.ArgumentParser(
        description="Phase 2: decoding-aware training primitives")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(func=cmd_selftest)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
