#!/usr/bin/env python3
"""
Suwayhib v6 -- Phase 3: the streaming decoder.

    python -m suwayhib.pipeline.decode selftest          # pure numpy, no model needed
    python -m suwayhib.pipeline.decode run --ckpt runs/head/best.pt
    python -m suwayhib.pipeline.decode sweep --ckpt runs/head/best.pt   # pick tau

WHAT THIS DOES
--------------
Given the phone sequence of the word the learner is EXPECTED to say next,
and a stream of per-frame log-probabilities from the trained head, decide
-- frame by frame -- whether that word has now been said. Confirmation is
strictly sequential and never retracted: the green edge only ever moves
forward.

WHY THERE IS NO FILLER STATE (short version: the literature says CTC
does not need one)
-----------------------------------------------------------------------
Three separate filler formulations were built and discarded before the
literature was checked properly. Recording the failures, because each
looked correct in isolation and each was caught by a different test:

  v1  filler = max(logp) per frame.
      max(logp) is by definition the best score ANY path can achieve on
      a frame, so a PERFECT constrained match could only ever tie it,
      never beat it. Every clean-confirm scenario failed with the two
      scores identical to four decimals. Caught by the self-test, before
      any real audio.

  v2  max(logp) minus a fixed per-frame penalty.
      Fixed v1, but the penalty was charged on silent frames too. During
      trailing silence the constrained path pays ~0 (recognising blank
      is free) while filler bleeds the penalty forever -- so after ANY
      word, correct or not, enough trailing silence made filler lose by
      ATTRITION. Reproduced: a deliberately wrong phone followed by 300
      frames of silence confirmed anyway, with the constrained score
      sitting perfectly FLAT at -60.00 the whole time. Zero new
      evidence; filler simply ran out of patience. Caught by tracing
      real recs/ audio.

  v3  filler = logp[blank], i.e. a silence/background model.
      Fixed v2's attrition, but real CTC output is SPIKY -- a trained
      model is blank-dominant on most frames even inside a correctly
      recognised phone. So filler became nearly free almost everywhere
      and essentially NOTHING confirmed, including plainly correct
      recordings. Caught immediately on real audio.

The pattern across all three: each was an attempt to bolt a competing
hypothesis onto a framework that already has one. Hwang et al., "Online
Keyword Spotting with a Character-Level Recurrent Neural Network"
(arXiv:1512.08903) -- CTC, character-level, streaming, i.e. our exact
setting -- built both variants and reported that removing the filler
network entirely, and simply thresholding the keyword path's negative
log-posterior, gives "virtually no performance difference". Their
conclusion, verbatim: the filler network is not needed for the CTC
keyword spotter. CTC's blank symbol already absorbs the non-keyword
modelling role that filler plays in classical keyword/filler HMMs.

Two further points from the same literature, both of which say the
discarded design was the weakest possible version of the idea:
  - where filler IS used with CTC successfully (Rosenberg et al.,
    arXiv:1710.09617), it is a SEPARATE DECODER ON A SEPARATE GRAPH
    whose best-path score is compared against the keyword graph's --
    not one extra scalar state inside the keyword lattice.
  - properly done, filler is a TRAINING-TIME construct: an extra unit
    added to the CTC label set, with non-keyword spans relabelled to it
    during training (arXiv:1808.00639). Our head has no such unit, so
    any decode-time filler was always approximating something the model
    was never taught.

WHAT REPLACES IT
----------------
Confirm on the LENGTH-NORMALISED log-posterior of the constrained path:

    score = (Viterbi log-prob of the best alignment of this word's
             phone sequence to the frames consumed so far) / n_frames

Dividing by frame count matters and is not cosmetic. An unnormalised
log-probability grows more negative the longer the audio, so a fixed
threshold on it would silently demand higher per-frame confidence from
long words than short ones -- and Qur'anic words vary from 1 to 19
phones (measured, phones.py). Normalising makes tau mean "average
per-frame log-probability", comparable across words.

ONE threshold, one meaning, instead of three interacting knobs.

DURATION CAP
------------
Kept, and it is doing real work. Each lattice state tracks how many
consecutive frames its own self-loop won; past a cap, further
self-looping is penalised. This is what makes the v5 failure -- word 0
of Recording_5 aligning over 3.75 SECONDS, swallowing the next two
words -- structurally impossible rather than merely unlikely. Without
it, parking in one state is free.

COMMIT LAG
----------
The score must stay above tau for several consecutive frames before the
word is actually confirmed, absorbing single-frame spikes. Handover
section 6: "commit lag ~150ms" -- 7-8 frames at 20ms/frame.

THE ASYMMETRY THAT SETS tau
---------------------------
A false stall costs the learner one repetition. A false confirm advances
the cursor past audio that was never verified, and because confirmation
is monotonic and never retracted, everything after it is misaligned --
unrecoverable within the session. So tau is chosen for a low
false-confirm rate, NOT for balanced accuracy. `sweep` reports both
directions so the operating point is picked on evidence.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..core import phones as P
from ..core import reftext as RT
from . import harness as H
from . import score as S

FRAMES_PER_SEC = 50
NEG_INF = float("-inf")


# --------------------------------------------------------------------------
# lattice
# --------------------------------------------------------------------------

def build_states(phone_ids):
    """[p1, p2, ...] -> [blank, p1, blank, p2, ..., pN, blank].

    Blank is written as -1 here rather than its real vocab id, so that
    the "may I skip the blank between two phones" test can compare
    states[i] against states[i-2] without a phone that happens to share
    blank's index being mistaken for a blank state.
    """
    states = [-1]
    for p in phone_ids:
        states.append(p)
        states.append(-1)
    return states


def skip_allowed(states, i):
    """May state i be entered directly from i-2, skipping the blank?

    Standard CTC rule: only between two PHONE states, and only when the
    phones differ. Allowing it between identical phones would collapse a
    genuine geminate -- the doubled symbols (ll, nn, bb, ...) that
    merge_gemination produces and that IqraEval's inventory encodes --
    into a single occurrence.
    """
    return (i >= 2 and states[i] != -1 and states[i - 2] != -1
            and states[i] != states[i - 2])


class WordLattice:
    """Viterbi alignment of one word's phone sequence against a growing
    frame stream.

    Scores are log-probabilities; the recursion takes max (Viterbi), not
    log-sum-exp (forward). Viterbi is what "best alignment" means, needs
    no numerical-stability tricks in log space, and is the standard
    choice in the streaming keyword-spotting literature this follows.
    """

    def __init__(self, phone_ids, blank_id, dwell_cap, dwell_penalty=3.0):
        self.states = build_states(phone_ids)
        self.L = len(self.states)
        self.blank_id = blank_id
        self.dwell_cap = dwell_cap
        self.dwell_penalty = dwell_penalty
        # Only state 0 starts alive. An earlier version also seeded
        # state 1, following the usual CTC convention that the first
        # label is a valid t=0 start -- but combined with the skip
        # transition that let state 3 inherit state 1's PRE-EVIDENCE
        # baseline, so a word's exit score went finite at t=1 with its
        # second phone never having appeared at all. State 1 gets an
        # equally valid score one frame later through the ordinary
        # 0 -> 1 transition; that one frame of latency is correct, not
        # a regression.
        self.score = [NEG_INF] * self.L
        # Frames the winning path into each state has spent OCCUPYING A
        # PHONE (not blank). Doubles as BOTH the normalisation
        # denominator AND the duration cap's own counter -- see below
        # for why a separate self-loop-only counter was not enough.
        self.score[0] = 0.0
        self.phone_frames = [0] * self.L
        self.n_frames = 0
        self.max_dwell = 0

    def _emit(self, i, logp):
        s = self.states[i]
        return logp[self.blank_id] if s == -1 else logp[s]

    def step(self, logp):
        prev, prev_pf = self.score, self.phone_frames
        new = [NEG_INF] * self.L
        new_pf = [0] * self.L

        for i in range(self.L):
            # Each candidate is penalised independently based on ITS OWN
            # inherited phone-frame count, not just on whether THIS
            # state's self-loop has been chosen repeatedly.
            #
            # BUG FOUND BY THE SELF-TEST: a self-loop-only dwell counter
            # missed an exploit entirely. Feed one phone confidently
            # forever with the next phone never arriving: the SKIP
            # transition (phone -> phone, bypassing the blank) kept
            # re-deriving fresh from the FIRST phone's continuously
            # good, continuously GROWING score every single frame --
            # never once choosing self-loop for the target state, so its
            # self-loop-only dwell counter never advanced past 1 and the
            # cap never fired, even though the inherited phone-frame
            # count was growing right along with the first phone's
            # legitimate accumulation. That growth was exactly what was
            # diluting the normalised score. Keying the cap on inherited
            # phone-frames instead, for every candidate regardless of
            # transition type, closes this: the moment the SOURCE
            # state's own phone-frame count passes the cap, whatever it
            # feeds forward is penalised too.
            cand_self = prev[i]
            if cand_self > NEG_INF and prev_pf[i] >= self.dwell_cap:
                cand_self -= self.dwell_penalty
            best, best_pf = cand_self, prev_pf[i]

            if i >= 1:
                c = prev[i - 1]
                if c > NEG_INF and prev_pf[i - 1] >= self.dwell_cap:
                    c -= self.dwell_penalty
                if c > best:
                    best, best_pf = c, prev_pf[i - 1]
            if skip_allowed(self.states, i):
                c = prev[i - 2]
                if c > NEG_INF and prev_pf[i - 2] >= self.dwell_cap:
                    c -= self.dwell_penalty
                if c > best:
                    best, best_pf = c, prev_pf[i - 2]

            if best == NEG_INF:
                continue
            new[i] = best + self._emit(i, logp)
            new_pf[i] = best_pf + (1 if self.states[i] != -1 else 0)

        self.score, self.phone_frames = new, new_pf
        self.n_frames += 1
        self.max_dwell = max(self.max_dwell,
                             max(new_pf) if new_pf else 0)

    @property
    def exit_raw_and_span(self):
        """(raw log-prob, phone-frames consumed) for whichever of
        {trailing blank, final phone} scores higher.

        The span counts only frames spent occupying a PHONE state --
        see __init__ for why silence must not lengthen it.
        """
        idx = self.L - 1
        if self.L >= 2 and self.score[self.L - 2] > self.score[idx]:
            idx = self.L - 2
        raw = self.score[idx]
        if raw == NEG_INF:
            return NEG_INF, 0
        return raw, max(1, self.phone_frames[idx])

    @property
    def exit_score(self):
        """Log-prob normalised by phone-frames actually consumed, not by
        wall-clock time since the lattice began or since the word's
        span started. See __init__ for the two attrition variants this
        replaced."""
        raw, span = self.exit_raw_and_span
        if raw == NEG_INF:
            return NEG_INF
        return raw / span


# --------------------------------------------------------------------------
# streaming decoder
# --------------------------------------------------------------------------

class StreamingDecoder:
    """Strictly sequential, monotonic word confirmation over one ayah.

        dec = StreamingDecoder(words, blank_id, tau=...)
        for logp in frames:
            ev = dec.step(logp)

    TWO FAILURE MODES FOUND BY INSTRUMENTING A STALLED WORD'S SCORE
    OVER TIME, NOT VISIBLE FROM THE SELF-TEST OR ANY SINGLE-FRAME VIEW
    --------------------------------------------------------------------
    1. A full word alignment can complete in as few frames as the
       automaton's structural minimum (~1 frame per phone via skip
       transitions), regardless of whether that many frames of real
       audio have actually elapsed. Measured on a real recording: a
       9-phone word "fully aligned" (reached the exit state) in 8
       frames (160ms) -- physically impossible for real speech, but
       perfectly legal in the DP, and it is the ONLY completed
       candidate available for the exit score to report until an
       honestly-timed alignment also finishes. At a loose enough tau
       this lets an utterance confirm on evidence that is not evidence
       at all -- and per the testset tau sweep, that is not
       theoretical: "correct confirmed" is NON-MONOTONIC in tau,
       peaking around -1.5/-1.75 and dropping on the LOOSE side too,
       which only makes sense if looser tau is causing premature
       confirms that misalign every later word and fail the utterance.
       `min_frames_per_phone` below rejects the exit score as usable
       until enough wall-clock frames have elapsed for the alignment to
       be physically plausible, independent of tau.

    2. Once a word's lattice completes, it keeps listening to every
       subsequent frame FOREVER while stalled -- there is no bound.
       Because the exit score's raw log-prob keeps accumulating whatever
       comes next (normalised only by the now-frozen phone-frame span),
       a stalled word's displayed score is not stable: it drifts based
       on unrelated future audio. Measured on a real "clean" word that
       never confirmed: its normalised score peaked near tau (-2.13,
       tau -1.5) around the word's true acoustic completion, then
       degraded to -11.1 as more frames arrived -- because once this
       word failed to confirm, the NEXT word's speech onset started
       bleeding into what this lattice can only interpret as "this
       word's trailing silence", and it is not silence. `max_frames`
       below bounds how long a single attempt keeps accumulating this
       kind of contamination: past that point the lattice resets to a
       fresh attempt (cursor unchanged) instead of compounding a failed
       alignment with an increasingly irrelevant tail. `_word_frames`
       is deliberately NOT reset by this -- it is the total time spent
       on this word across however many internal resets, and is what
       the learner-facing `stalled` signal is built from; resetting it
       would make the UI misreport recovery that never happened.
    """

    def __init__(self, words, blank_id, tau=-1.5, dwell_cap=40,
                 commit_lag=8, stall_lag=60, min_frames_per_phone=2,
                 max_frames_per_phone=25, max_frames_floor=100):
        self.words = words
        self.blank_id = blank_id
        self.tau = tau
        self.dwell_cap = dwell_cap
        self.commit_lag = commit_lag
        self.stall_lag = stall_lag
        self.min_frames_per_phone = min_frames_per_phone
        self.max_frames_per_phone = max_frames_per_phone
        self.max_frames_floor = max_frames_floor

        self.cursor = 0
        self.confirmed = []
        self._word_frames = 0
        self._new_lattice()

    def _new_lattice(self):
        self.lattice = (WordLattice(self.words[self.cursor], self.blank_id,
                                    self.dwell_cap)
                        if self.cursor < len(self.words) else None)
        self._above = 0
        self._below = 0

    @property
    def done(self):
        return self.cursor >= len(self.words)

    def step(self, logp):
        if self.done:
            return {"confirmed": False, "word_idx": None, "stalled": False,
                    "done": True, "score": None}

        self.lattice.step(logp)
        self._word_frames += 1
        s = self.lattice.exit_score

        n_phones = (self.lattice.L - 1) // 2
        min_frames = max(2, n_phones * self.min_frames_per_phone)
        above = s > self.tau and self.lattice.n_frames >= min_frames

        self._above = self._above + 1 if above else 0
        self._below = 0 if above else self._below + 1

        confirmed, idx = False, None
        if self._above >= self.commit_lag:
            idx = self.cursor
            self.confirmed.append(idx)
            self.cursor += 1
            self._word_frames = 0
            self._new_lattice()
            confirmed = True
        else:
            max_frames = max(self.max_frames_floor,
                             n_phones * self.max_frames_per_phone)
            if self.lattice.n_frames >= max_frames:
                self._new_lattice()

        return {"confirmed": confirmed, "word_idx": idx,
                "stalled": self._word_frames >= self.stall_lag,
                "done": self.done, "score": s}


def decode_utterance(words, logp, blank_id=0, tau=-1.5, dwell_cap=40,
                     commit_lag=8, stall_lag=60, min_frames_per_phone=2,
                     max_frames_per_phone=25, max_frames_floor=100):
    """-> (confirmed [(frame, word_idx)], final cursor, decoder)."""
    dec = StreamingDecoder(words, blank_id, tau, dwell_cap, commit_lag,
                           stall_lag, min_frames_per_phone,
                           max_frames_per_phone, max_frames_floor)
    trace = []
    for t in range(logp.shape[0]):
        ev = dec.step(logp[t])
        if ev["confirmed"]:
            trace.append((t, ev["word_idx"]))
        if dec.done:
            break
    return trace, dec.cursor, dec


# --------------------------------------------------------------------------
# synthetic frames for the self-test
# --------------------------------------------------------------------------

VOCAB = {"<blank>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "x": 5}
N_CLASSES = len(VOCAB)
BLANK = 0


def _norm(v):
    return v - np.log(np.sum(np.exp(v)))


def confident(sym, peak=6.0, floor=-6.0):
    v = np.full(N_CLASSES, float(floor))
    v[sym] = peak
    return _norm(v)


def leaning(sym, blank_lead=0.5, peak=2.5, floor=-4.0):
    """Blank-dominant but leaning toward `sym` -- the between-spike
    resting state of real CTC output."""
    v = np.full(N_CLASSES, float(floor))
    v[BLANK] = peak
    v[sym] = peak - blank_lead
    return _norm(v)


def spiky_word(phone_syms, rest=10, spike=4, lead=5, trail=15):
    """A realistically SPIKY frame sequence: mostly blank-dominant, with
    short confident spikes per phone.

    This shape exists because a filler formulation that passed every
    clean-block scenario in this file failed on all six real recordings
    -- real CTC output is blank-dominant even inside a correctly
    recognised phone, and clean one-hot blocks never exercise that. Any
    change to the lattice or the confirmation rule must be checked
    against this shape, not only the tidy ones.
    """
    seq = [confident(BLANK) for _ in range(lead)]
    for s in phone_syms:
        seq += [leaning(s) for _ in range(rest)]
        seq += [confident(s) for _ in range(spike)]
    seq += [confident(BLANK) for _ in range(trail)]
    return np.array(seq)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def sc_clean_confirm():
    """Textbook input, textbook result: confirms exactly once."""
    seq = ([BLANK] * 3 + [VOCAB["a"]] * 5 + [BLANK] * 2 +
           [VOCAB["b"]] * 5 + [BLANK] * 10)
    logp = np.array([confident(s) for s in seq])
    trace, cur, _ = decode_utterance([[VOCAB["a"], VOCAB["b"]]], logp,
                                     tau=-1.5, commit_lag=5)
    return len(trace) == 1 and cur == 1, f"{len(trace)} confirms (want 1)"


def sc_wrong_word():
    """Audio says 'x x x ...'; expected word is 'a b'. Must never
    confirm, and must eventually report a stall."""
    logp = np.array([confident(VOCAB["x"]) for _ in range(60)])
    dec = StreamingDecoder([[VOCAB["a"], VOCAB["b"]]], BLANK, tau=-1.5,
                           commit_lag=5, stall_lag=15)
    evs = [dec.step(logp[t]) for t in range(len(logp))]
    n = sum(e["confirmed"] for e in evs)
    st = any(e["stalled"] for e in evs)
    return n == 0 and st, f"{n} confirms (want 0), stalled={st}"


def sc_wrong_middle_phone():
    """'a', then the WRONG phone where 'b' belongs, then 'c'. Must not
    confirm on the strength of a correct first and last phone -- the
    mean-vs-worst-token trap the project has hit before (handover
    section 6: "one wrong diacritic in an eight-token word barely moves
    the mean but IS the error")."""
    seq = ([BLANK] * 3 + [VOCAB["a"]] * 5 + [BLANK] * 2 +
           [VOCAB["x"]] * 5 + [BLANK] * 2 + [VOCAB["c"]] * 5 + [BLANK] * 10)
    logp = np.array([confident(s) for s in seq])
    trace, _, _ = decode_utterance(
        [[VOCAB["a"], VOCAB["b"], VOCAB["c"]]], logp, tau=-1.5, commit_lag=5)
    return len(trace) == 0, f"{len(trace)} confirms (want 0)"


def sc_wrong_survives_silence():
    """The attrition trap. A wrong word followed by arbitrarily long
    silence must NEVER eventually confirm. An earlier design let exactly
    this happen: the competing score drained on every silent frame until
    it lost, with the constrained score sitting perfectly flat and no
    new evidence arriving. Checked at 300 AND 1000 frames, because a fix
    that merely delays attrition still fails at some length."""
    out = []
    for n_sil in (300, 1000):
        seq = ([BLANK] * 3 + [VOCAB["a"]] * 5 + [BLANK] * 2 +
               [VOCAB["x"]] * 5 + [BLANK] * n_sil)
        logp = np.array([confident(s) for s in seq])
        trace, _, _ = decode_utterance(
            [[VOCAB["a"], VOCAB["b"], VOCAB["c"]]], logp, tau=-1.5,
            commit_lag=5)
        out.append((n_sil, len(trace)))
    return all(n == 0 for _, n in out), f"{out} (want 0 at every length)"


def sc_duration_cap():
    """The 3.75-second parking failure, reproduced deliberately. Audio
    holds 'a' forever and 'b' never arrives. Must not confirm."""
    logp = np.array([confident(VOCAB["a"]) for _ in range(200)])
    dec = StreamingDecoder([[VOCAB["a"], VOCAB["b"]]], BLANK, tau=-1.5,
                           dwell_cap=10, commit_lag=5, stall_lag=15)
    evs = [dec.step(logp[t]) for t in range(len(logp))]
    n = sum(e["confirmed"] for e in evs)
    st = any(e["stalled"] for e in evs)
    return n == 0 and st, f"{n} confirms (want 0), stalled={st}"


def sc_sequential():
    """Two words confirm in order, cursor advancing monotonically."""
    seq = ([BLANK] * 2 + [VOCAB["a"]] * 5 + [BLANK] * 5 +
           [VOCAB["b"]] * 5 + [BLANK] * 10)
    logp = np.array([confident(s) for s in seq])
    trace, _, _ = decode_utterance([[VOCAB["a"]], [VOCAB["b"]]], logp,
                                   tau=-1.5, commit_lag=5)
    order = [w for _, w in trace]
    return order == [0, 1], f"order {order} (want [0, 1])"


def sc_geminate():
    """A doubled phone keeps its intervening blank: the skip transition
    must be refused between identical phones, or geminates collapse."""
    lat = WordLattice([VOCAB["a"], VOCAB["a"]], BLANK, dwell_cap=25)
    rule_ok = not skip_allowed(lat.states, 3)
    seq = ([BLANK] * 2 + [VOCAB["a"]] * 4 + [BLANK] * 4 +
           [VOCAB["a"]] * 4 + [BLANK] * 10)
    logp = np.array([confident(s) for s in seq])
    trace, _, _ = decode_utterance([[VOCAB["a"], VOCAB["a"]]], logp,
                                   tau=-1.5, commit_lag=5)
    return (rule_ok and len(trace) == 1,
            f"skip refused={rule_ok}, {len(trace)} confirms (want 1)")


def sc_spiky_confirm():
    """Realistic blank-dominant spiky input, correct word: must confirm.
    This is the shape whose absence let a broken design pass everything
    else while failing on every real recording."""
    logp = spiky_word([VOCAB["a"], VOCAB["b"]])
    trace, _, _ = decode_utterance([[VOCAB["a"], VOCAB["b"]]], logp,
                                   tau=-1.5, dwell_cap=60, commit_lag=5)
    return len(trace) == 1, f"{len(trace)} confirms on spiky input (want 1)"


def sc_spiky_wrong():
    """Realistic spiky input, wrong phones: must not confirm."""
    logp = spiky_word([VOCAB["x"], VOCAB["x"]])
    trace, _, _ = decode_utterance([[VOCAB["a"], VOCAB["b"]]], logp,
                                   tau=-1.5, dwell_cap=60, commit_lag=5)
    return len(trace) == 0, f"{len(trace)} confirms on wrong spiky (want 0)"


def sc_length_invariance():
    """A short word and a long word, both spoken correctly and equally
    confidently, must land at COMPARABLE scores -- that is what length
    normalisation buys, and without it one fixed tau cannot serve words
    ranging from 1 to 19 phones (the measured range in Juz Amma)."""
    short = spiky_word([VOCAB["a"]])
    long_ = spiky_word([VOCAB["a"], VOCAB["b"], VOCAB["c"], VOCAB["d"]])
    _, _, d1 = decode_utterance([[VOCAB["a"]]], short, tau=-99,
                                dwell_cap=60, commit_lag=10 ** 9)
    _, _, d2 = decode_utterance(
        [[VOCAB["a"], VOCAB["b"], VOCAB["c"], VOCAB["d"]]], long_, tau=-99,
        dwell_cap=60, commit_lag=10 ** 9)
    s1, s2 = d1.lattice.exit_score, d2.lattice.exit_score
    gap = abs(s1 - s2)
    return gap < 0.5, (f"short {s1:.3f} vs long {s2:.3f}, gap {gap:.3f} "
                       f"(want <0.5)")


SCENARIOS = [
    ("clean_confirm", sc_clean_confirm),
    ("wrong_word", sc_wrong_word),
    ("wrong_middle_phone", sc_wrong_middle_phone),
    ("wrong_survives_silence", sc_wrong_survives_silence),
    ("duration_cap", sc_duration_cap),
    ("sequential_order", sc_sequential),
    ("geminate_kept", sc_geminate),
    ("spiky_confirm", sc_spiky_confirm),
    ("spiky_wrong", sc_spiky_wrong),
    ("length_invariance", sc_length_invariance),
]


def cmd_selftest(a):
    print("Pure numpy. No model, no audio.\n")
    n = 0
    for name, fn in SCENARIOS:
        ok, detail = fn()
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<24} {detail}")
        n += ok
    print(f"\n{n}/{len(SCENARIOS)} PASS")
    if n < len(SCENARIOS):
        print("\nReal audio is not worth running until these pass. Every")
        print("decoder failure in this project's history is represented")
        print("by one of these scenarios.")
        return 1
    return 0


# --------------------------------------------------------------------------
# real audio
# --------------------------------------------------------------------------

def frame_logp(scorer, audio):
    import torch
    feats = scorer.enc([audio], scorer.layers)[0]
    x = torch.from_numpy(feats.astype("float32")).unsqueeze(0).to(
        scorer.device)
    with torch.no_grad():
        logits = scorer.model(x)["logits"][0]
        return torch.log_softmax(logits.float(), -1).cpu().numpy()


def prepare(a):
    scorer = S.LiveScorer(a.ckpt, a.device)
    rt = RT.RefText(a.text)
    items = []
    for d in H.load_descriptions(a.descriptions):
        path = Path(a.recs) / d["file"]
        if not path.exists():
            continue
        words = [[scorer.vocab[p] for p in P.word_to_phones_iqra(w)
                  if p in scorer.vocab]
                 for w in rt.words(d["surah"], d["ayah"])]
        items.append((d, words, frame_logp(scorer, S.load_audio(path))))
    return items


def prepare_testset(a):
    """Same (d, words, logp) shape as prepare(), sourced from testset/
    instead of recs/ -- 2,000 labelled synthetic items with real ground
    truth, versus six recordings. Reuses score.py's ALREADY-TESTED
    load_testset/load_audio rather than re-parsing the schema a fourth
    time in this project; that schema (error_type, is_error, word_idx,
    tier, split, ...) was confirmed by direct inspection, not assumed,
    the first time score.py needed it.

    `d["kind"]` is mapped to "correct" (is_error=0, includes clean AND
    control_seam -- the seam itself is not a real error) or "error"
    (is_error=1), so the exact same three-column report in cmd_sweep
    works unchanged for either source. There is no "artifact" case here
    -- that column will simply read 0/0 for a testset run.

    word_idx and error_type are carried through on `d` for anyone who
    wants a finer per-error-type breakdown later; cmd_sweep's existing
    columns do not need them.
    """
    scorer = S.LiveScorer(a.ckpt, a.device)
    rt = RT.RefText(a.text)

    rows = S.load_testset(a.testset)
    if a.split:
        rows = [r for r in rows if r["split"] == a.split]
    if a.n:
        rows = rows[:a.n]

    items, skipped = [], 0
    for row in rows:
        if not row["_path"].exists():
            skipped += 1
            continue
        words_text = rt.words(row["surah"], row["ayah"])
        if len(words_text) != row["n_words"]:
            # Same guard score.py's cmd_testset uses: a word-count
            # mismatch between the label and reftext's own split means
            # something is inconsistent for this item, and silently
            # using the wrong word count would misattribute every
            # downstream stall/confirm position.
            skipped += 1
            continue
        words = [[scorer.vocab[p] for p in P.word_to_phones_iqra(w)
                  if p in scorer.vocab] for w in words_text]
        d = {"file": row["file"],
             "kind": "error" if row["is_error"] else "correct",
             "word_idx": row.get("word_idx"),
             "error_type": row.get("error_type")}
        items.append((d, words, frame_logp(scorer, S.load_audio(row["_path"]))))

    if skipped:
        print(f"  skipped {skipped} testset items "
              f"(missing audio or word-count mismatch)")
    print(f"  prepared {len(items)} testset items"
          f"{f' (split={a.split})' if a.split else ''}")
    return items


def _prepare_any(a):
    return prepare_testset(a) if a.source == "testset" else prepare(a)


def cmd_run(a):
    items = _prepare_any(a)
    print(f"\ntau {a.tau}   dwell cap {a.dwell_cap} frames "
          f"({a.dwell_cap * 20}ms)   commit lag {a.commit_lag} frames "
          f"({a.commit_lag * 20}ms)   min {a.min_frames_per_phone}/max "
          f"{a.max_frames_per_phone} frames per phone (floor "
          f"{a.max_frames_floor})\n")
    for d, words, logp in items:
        trace, cursor, dec = decode_utterance(
            words, logp, tau=a.tau, dwell_cap=a.dwell_cap,
            commit_lag=a.commit_lag,
            min_frames_per_phone=a.min_frames_per_phone,
            max_frames_per_phone=a.max_frames_per_phone,
            max_frames_floor=a.max_frames_floor)
        print(f"{d['file']:<16} [{d['kind']:<9}] words={len(words)}  "
              f"confirmed={len(trace)}  cursor={cursor}")
        for t, wi in trace:
            print(f"    t={t:4d} ({t / FRAMES_PER_SEC:5.2f}s)  word {wi}")
        if cursor < len(words) and dec.lattice is not None:
            print(f"    stuck at word {cursor}  "
                  f"score {dec.lattice.exit_score:.3f} vs tau {a.tau}  "
                  f"max dwell {dec.lattice.max_dwell} frames")
    return 0


def cmd_sweep(a):
    """Sweep tau and report, for each value, what happens on every
    recording -- so the operating point is chosen from evidence rather
    than guessed.

    The column to read first is whether ERRORS still stall, not overall
    accuracy. A false stall costs one repetition; a false confirm
    silently misaligns the rest of the ayah and cannot be undone.
    Prefer the strictest tau that still confirms the correct
    recordings.

    --source testset runs this same sweep over the 2000-item labelled
    set instead of the six recordings. Six items can only show the
    SHAPE of the tradeoff; at that resolution a 1-in-6 disagreement
    cannot be told apart from one unlucky recording. testset/ is where
    the operating point should actually be picked from.
    """
    items = _prepare_any(a)
    lo, hi, step = a.tau_lo, a.tau_hi, a.tau_step
    taus = [round(lo + i * step, 3)
            for i in range(int(round((hi - lo) / step)) + 1)]

    print(f"\n{'tau':>7} | {'correct confirmed':>18} | "
          f"{'error/weak stalled':>18} | {'artifact confirmed':>18}")
    print("-" * 72)
    for tau in taus:
        ok_c = tot_c = ok_e = tot_e = ok_a = tot_a = 0
        for d, words, logp in items:
            _, cursor, _ = decode_utterance(
                words, logp, tau=tau, dwell_cap=a.dwell_cap,
                commit_lag=a.commit_lag,
                min_frames_per_phone=a.min_frames_per_phone,
                max_frames_per_phone=a.max_frames_per_phone,
                max_frames_floor=a.max_frames_floor)
            full = cursor >= len(words)
            if d["kind"] == "correct":
                tot_c += 1
                ok_c += full
            elif d["kind"] == "artifact":
                tot_a += 1
                ok_a += full
            else:
                tot_e += 1
                ok_e += not full
        print(f"{tau:7.2f} | {ok_c:8d}/{tot_c:<9d} | "
              f"{ok_e:8d}/{tot_e:<9d} | {ok_a:8d}/{tot_a:<9d}")

    if a.source == "recs":
        print("\nWith six recordings this is a smoke test, not a")
        print("calibration. Re-run with --source testset for a real")
        print("operating point.")
    else:
        print("\nEven 2000 synthetic items is still one voice (plus")
        print("TTS/reciters already in the corpus) -- see the project's")
        print("own standing note that real speaker diversity is still")
        print("unmeasured.")

    if a.by_error_type and a.source == "testset":
        print(f"\nby error_type, at tau={a.tau}:")
        from collections import Counter
        seen, ok = Counter(), Counter()
        for d, words, logp in items:
            et = d.get("error_type") or ("clean" if d["kind"] == "correct"
                                         else "unknown")
            _, cursor, _ = decode_utterance(
                words, logp, tau=a.tau, dwell_cap=a.dwell_cap,
                commit_lag=a.commit_lag,
                min_frames_per_phone=a.min_frames_per_phone,
                max_frames_per_phone=a.max_frames_per_phone,
                max_frames_floor=a.max_frames_floor)
            full = cursor >= len(words)
            want_full = (d["kind"] == "correct")
            seen[et] += 1
            ok[et] += (full == want_full)
        for k in sorted(seen):
            print(f"  {k:<24} {ok[k]:4d}/{seen[k]:<4d} ({ok[k]/seen[k]:.1%})")
        print("\nA category sitting well below the others at this tau is")
        print("where the PHONE HEAD needs work, not where tau needs")
        print("adjusting -- no threshold fixes a category the model")
        print("cannot discriminate at all.")
    return 0


def main():

    ap = argparse.ArgumentParser(description="v6 streaming decoder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest").set_defaults(func=cmd_selftest)

    def common(p):
        p.add_argument("--ckpt", default="runs/head/best.pt")
        p.add_argument("--device", default="cuda")
        p.add_argument("--recs", default="recs")
        p.add_argument("--descriptions", default="recs/descriptions.txt")
        p.add_argument("--text", default="texts/quran-simple-plain.txt")
        p.add_argument("--dwell-cap", type=int, default=40)
        p.add_argument("--commit-lag", type=int, default=8)
        p.add_argument("--min-frames-per-phone", type=int, default=2,
                       help="an exit score does not count as 'above tau' "
                            "until this many frames per phone have "
                            "elapsed -- rejects structurally-fastest-"
                            "possible alignments that complete before "
                            "real speech could have produced them")
        p.add_argument("--max-frames-per-phone", type=int, default=25,
                       help="a word attempt that has run this many "
                            "frames per phone without confirming resets "
                            "to a fresh attempt, so a stalled word's "
                            "score cannot keep drifting on whatever "
                            "audio (silence, or the next word bleeding "
                            "in) arrives after it has already failed")
        p.add_argument("--max-frames-floor", type=int, default=100,
                       help="floor on the above, so short words still "
                            "get a fair minimum attempt window")
        # New, opt-in only: default "recs" reproduces exactly the
        # existing behaviour with zero args changed, so nothing that
        # already worked against recs/ is affected by adding this.
        p.add_argument("--source", choices=["recs", "testset"],
                       default="recs")
        p.add_argument("--testset", default="testset")
        p.add_argument("--split", choices=["dev", "holdout"], default=None,
                       help="testset/ only. holdout = Husary/Minshawy, "
                            "the genuine generalisation check")
        p.add_argument("--n", type=int, default=0,
                       help="testset/ only. 0 = all 2000; try a smaller "
                            "number first, this re-runs live encoding "
                            "for every item")

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--tau", type=float, default=-1.5)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep")
    common(s)
    s.add_argument("--tau-lo", type=float, default=-4.0)
    s.add_argument("--tau-hi", type=float, default=-0.2)
    s.add_argument("--tau-step", type=float, default=0.2)
    s.add_argument("--by-error-type", action="store_true",
                   help="testset/ only: after the sweep, break down "
                        "pass rate by error_type at a single --tau")
    s.add_argument("--tau", type=float, default=-1.4,
                   help="the single tau used for --by-error-type only; "
                        "the main sweep table ignores this")
    s.set_defaults(func=cmd_sweep)

    a = ap.parse_args()
    sys.exit(a.func(a) or 0)


if __name__ == "__main__":
    main()
