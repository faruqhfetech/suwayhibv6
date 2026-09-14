#!/usr/bin/env python3
"""
Suwayhib v6 -- streaming evaluation harness on recs/.

    python -m suwayhib.pipeline.harness selftest              # harness must FAIL a random stub
    python -m suwayhib.pipeline.harness run --model teacher_worst_token --device cpu
    python -m suwayhib.pipeline.harness run --model random --device cpu     # same as selftest

WHY THIS EXISTS, AND WHY IT COMES BEFORE ANY MODEL
-----------------------------------------------------
v5 wrote five streaming decoders across several rewrites, and every one of
them passed its synthetic unit tests before failing on real recordings.
recs/ -- seven takes from one uncontaminated voice -- was, per the
handover, "the only test that has been consistently informative," and it
was not run early enough. This harness exists so it is run FIRST, against
literally nothing (a stub that returns random numbers), specifically to
prove the harness itself can fail. A test that cannot fail is not a test.

WHAT IT MEASURES
-----------------
For each recording, feed the model frame-by-frame (or word-by-word, for a
non-streaming scorer -- see `TeacherWorstToken` below) against the known
verse, and record:

  - cursor trajectory: which word index is confirmed, in order
  - time-to-confirm per word
  - FALSE CONFIRMATIONS: a word confirmed that is not the true next word,
    or confirmed out of order. Target: ZERO, always, on every recording.
    A false confirm advances both cursor and audio position and is
    UNRECOVERABLE under the monotonic-green design (handover section 1,
    corrected: strict sequentiality prevents skipping ahead, but does NOT
    prevent a false confirm from cascading into consuming the next word's
    audio -- that asymmetry is why false confirms are checked separately
    and weighted infinitely more heavily than stalls here).
  - STALL POSITION: the word index the cursor is stuck at when the
    recording ends, for recordings known to contain an error.

descriptions.txt gives the (surah, ayah) and a note for each recording,
classified by classify_note() into four kinds -- correct, error, weak,
artifact. See that function for what each means and why the distinction
matters; the short version is that "weak" (right word, under-articulated)
SHOULD stall, while "artifact" (right word, defective capture) should
NOT, and conflating them would hide a real false-positive class.

EXPECTED_STALL_WORD carries the verified stall position for the items
where one is known by evidence. It is deliberately not guessed: a wrong
"known-correct" stall position would silently pass a broken decoder.

MODEL INTERFACE
-----------------
A model for this harness is anything exposing:

    .word_score(audio, word_idx, surah, ayah) -> float in [0, 1]

  i.e. "how much does this audio support having just heard word_idx of
  this verse, at this point in the recording." This is deliberately the
  loosest possible interface: it fits the random stub, a whole-verse
  scorer like textscore.py used word-by-word (SIMULATED streaming, not
  real streaming -- see caveat below), and, later, the real frame-
  synchronous graph decoder from Phase 3 by having its adapter walk the
  audio incrementally and report the same per-word score at each check.

  CAVEAT: textscore.py's actual decoder sees the WHOLE recording at once
  and scores each word non-causally. Wrapping it here checks the METRICS
  and the STALL/CONFIRM LOGIC end-to-end, which is real value now, but it
  is not a test of streaming behaviour itself -- only Phase 3's frame-
  synchronous decoder, run through this same harness, tests that.
"""

import argparse
import random
import re
import subprocess
import sys
from pathlib import Path

from ..core import reftext as RT
from ..core import textscore as TS

SR = 16000

# ---------------------------------------------------------------------
# ANNOTATE BY EAR. (surah, ayah, recording_filename) -> word_idx (0-based)
# of the word the person got wrong. Leave unset (absent from this dict)
# until verified by listening -- a guess here is worse than no claim,
# since a wrong "known-correct" stall position would silently pass a
# broken decoder.
# ---------------------------------------------------------------------
EXPECTED_STALL_WORD = {
    # Verified against textscore.py's free transcription (the `heard:`
    # line, which is Whisper's own reading, independent of forced scoring)
    # plus the user's own listening. Only fill in entries where BOTH agree
    # -- a wrong "known-correct" stall position would silently pass a
    # broken decoder, which is worse than no claim at all.
    #
    # 78:6 = وَجَعَلْنَا ... (4 words); 78:9 = وَجَعَلْنَا نَوْمَكُمْ سُبَاتًا (3 words)
    "Recording_4.m4a": 2,   # "wrong word in the second position" -- the
                            # user's own description. w0/w1 confirm, w2 is
                            # the substituted word.
    "Recording_5.m4a": 1,   # deliberate s/S swap is in سُبَاتًا (w2), BUT
                            # this take also under-articulates نَوْمَكُمْ
                            # (w1, scored 0.000), so the green edge
                            # correctly stops at w1 and never reaches the
                            # deliberate error. Recorded here as w1 because
                            # that is the CORRECT behaviour for this audio,
                            # not because the labelled error is at w1.
                            # NOTE: this means Recording_5 does not
                            # actually exercise s/S detection end-to-end --
                            # a cleaner re-take of 78:9 with ONLY the s/S
                            # swap would be genuinely useful evaluation
                            # data.
    "Recording_3.m4a": 1,   # weak articulation of نَوْمَكُمْ (w1)
    # Recording_1.m4a is kind="artifact" and must COMPLETE, so it has no
    # expected stall -- a stall there is the false positive being tracked.
}


# --------------------------------------------------------------------------
# descriptions.txt parsing
# --------------------------------------------------------------------------

def classify_note(note):
    """-> one of "correct", "error", "weak", "artifact".

    FOUR categories, because the recordings genuinely contain four
    different things and collapsing them hides what is being measured.

      correct   clean recitation; the device should confirm every word.

      error     a genuinely wrong phoneme or word (Recording_4's wrong
                word, Recording_5's deliberate s/S swap). The device
                should stall.

      weak      the RIGHT word, under-articulated (Recording_3: Whisper
                free-transcribes lawmakum for nawmakum). The device
                should ALSO stall -- that is the whole design: the green
                edge stops, the learner repeats the word more clearly,
                and it confirms. This is the case the handover's worked
                example documents on the same word ("the model had been
                right every time; the articulation was weak"), and the
                case no synthetic corruption in injector.py generates.

      artifact  the right word, said correctly, but the CAPTURE is
                defective (Recording_1: the recorder started late and ate
                the leading wa-al-, so Whisper hears jibaala for
                waaljibaala and the weak token is exactly the initial
                waw). Handover section 6 records this as a confirmed
                cause of one false flag.

                CRITICALLY, the expected behaviour here is the OPPOSITE
                of weak: the learner said it perfectly and can do nothing
                about it, so a stall is a FALSE POSITIVE. This category
                is expected to FAIL today, and that failure is the honest
                report of a known open limitation, not something to be
                relabelled away. Phase 6's AudioWorklet capture (raw PCM,
                ring buffer, no MediaRecorder container framing) exists
                partly to stop generating these; this item is the
                regression test for that.
    """
    n = note.lower()
    if "artifact" in n or "clipped" in n:
        return "artifact"
    if "weak" in n:
        return "weak"
    if n.startswith("correct"):
        return "correct"
    return "error"


def load_descriptions(path):
    """-> [{"file": ..., "surah": int, "ayah": int, "kind": str,
            "correct": bool, "note": str}, ...]

    Format (as shipped): "Recording_4.m4a | 78 6 | error - wrong word ..."
    Pipe-delimited, middle field is "SURAH AYAH", third field is free text
    classified by classify_note().
    """
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            raise SystemExit(f"unexpected descriptions.txt line: {line!r}")
        fname, ref, note = parts
        m = re.match(r"(\d+)\s+(\d+)", ref)
        if not m:
            raise SystemExit(f"could not parse surah/ayah from: {ref!r}")
        kind = classify_note(note)
        out.append({
            "file": fname,
            "surah": int(m.group(1)),
            "ayah": int(m.group(2)),
            "kind": kind,
            "correct": kind == "correct",
            "note": note,
        })
    return out


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def load_audio(path):
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"could not decode {path}: {r.stderr.decode()[:300]}")
    import numpy as np
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

class RandomModel:
    """Returns uniform random scores, independent of audio or word.

    This MUST fail the harness. If it passes, the harness measures nothing
    -- per handover section 6, "a random model must fail your metric,"
    applied here to the harness itself rather than to a trained checkpoint.
    """
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def word_score(self, audio, word_idx, surah, ayah):
        return self.rng.random()


class TeacherWorstToken:
    """Wraps textscore.py's forced-decoding scorer.

    NON-STREAMING: scores the whole recording once, then answers
    per-word queries from that single pass. This tests the harness's
    metrics and stall/confirm logic end-to-end using a scorer already
    known to work (v4's shipped device), but it is NOT a test of
    streaming behaviour -- see module docstring.
    """
    name = "teacher_worst_token"

    def __init__(self, model="whisper-base-quran",
                 text="texts/quran-simple-plain.txt", device="cpu",
                 threshold=0.10):
        self.scorer = TS.TextScorer(model=model, text=text, device=device)
        self.threshold = threshold
        self._cache = {}

    def _scored(self, audio, surah, ayah):
        key = (id(audio), surah, ayah)
        if key not in self._cache:
            self._cache[key] = self.scorer.score(audio, surah, ayah)
        return self._cache[key]

    def word_score(self, audio, word_idx, surah, ayah):
        r = self._scored(audio, surah, ayah)
        if not r or word_idx >= len(r["words"]):
            return 0.0
        return r["words"][word_idx]["worst"]


MODELS = {"random": RandomModel, "teacher_worst_token": TeacherWorstToken}


# --------------------------------------------------------------------------
# the walk: strictly sequential, monotonic, per handover section 1
# --------------------------------------------------------------------------

def walk(model, audio, words, surah, ayah, threshold=0.5, max_stall_checks=1):
    """Strictly sequential cursor. Returns a trajectory dict.

    Since every current model in MODELS answers word_score() from a single
    whole-recording pass rather than a true incremental stream, "time" here
    is CHECK COUNT, not audio position -- a placeholder axis Phase 3's real
    frame-synchronous decoder will replace with actual elapsed frames.
    """
    cursor = 0
    confirmed = []          # [(word_idx, score)]
    false_confirms = []     # anything confirmed out of order -- must be []
    trace = []

    # HONEST LIMITATION, found by testing this against constructed fake
    # models before handing it over: with the current per-word-batch-scorer
    # interface, word_idx in this loop is ALWAYS equal to cursor at the
    # point of the check -- the loop enforces strict order and breaks on
    # the first non-confirm, so it can never query out of sequence. That
    # means `false_confirms` below is CORRECT but currently VACUOUS: it
    # cannot fire against any model built on this interface, including the
    # real Phase-3 graph decoder's word_score() if wrapped the same way.
    #
    # Real false-confirm detection needs a model that maintains its OWN
    # cursor across a genuine incremental audio stream (chunk by chunk),
    # so it can independently desync -- e.g. confirm word k+1 before k, or
    # jump the cursor by more than one word in a single step. That needs a
    # different, richer interface (reset() + step(chunk) -> {cursor,
    # stalled}) and a second walk function to match. Deferred to Phase 3,
    # when there is an actual streaming decoder to wrap; building that
    # interface speculatively now, against a decoder that does not exist
    # yet, risks guessing its shape wrong. Do not treat a clean
    # `false_confirms: []` from THIS walk() as evidence of anything beyond
    # "the per-word scorer didn't contradict itself when queried in order,"
    # which is a real but much weaker property.
    for word_idx in range(len(words)):
        score = model.word_score(audio, word_idx, surah, ayah)
        trace.append({"word_idx": word_idx, "score": score})
        if score >= threshold:
            if word_idx != cursor:
                false_confirms.append({
                    "expected_cursor": cursor, "confirmed_instead": word_idx,
                    "score": score})
            confirmed.append((word_idx, score))
            cursor = word_idx + 1
        else:
            break     # STRICTLY SEQUENTIAL: stop at the first non-confirm

    return {
        "surah": surah, "ayah": ayah, "n_words": len(words),
        "cursor_final": cursor, "confirmed": confirmed,
        "false_confirms": false_confirms, "trace": trace,
        "completed": cursor >= len(words),
    }


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(model, recs_dir, descriptions, ref_text_path, threshold, verbose):
    rt = RT.RefText(ref_text_path)

    results = []
    for d in descriptions:
        path = Path(recs_dir) / d["file"]
        if not path.exists():
            print(f"  [SKIP] {d['file']} not found at {path}")
            continue
        words = rt.words(d["surah"], d["ayah"])
        audio = load_audio(path)
        r = walk(model, audio, words, d["surah"], d["ayah"], threshold)
        r.update({"file": d["file"], "correct": d["correct"],
                  "kind": d["kind"], "note": d["note"],
                  "expected_stall": EXPECTED_STALL_WORD.get(d["file"])})
        results.append(r)

        status = _verdict(r)
        print(f"  [{status}] {d['file']:<16s} {d['surah']}:{d['ayah']:<3d} "
              f"[{d['kind']}] ({d['note']})")
        print(f"          words={r['n_words']}  "
              f"confirmed={len(r['confirmed'])}  "
              f"cursor_final={r['cursor_final']}  "
              f"false_confirms={len(r['false_confirms'])}")
        if r["false_confirms"] and verbose:
            for fc in r["false_confirms"]:
                print(f"          FALSE CONFIRM: {fc}")
        if verbose:
            for t in r["trace"]:
                print(f"          w{t['word_idx']}  score={t['score']:.3f}")
    return results


def _verdict(r):
    """PASS/FAIL against the properties the harness actually checks.

    - ANY recording: zero false confirms, always. Non-negotiable per the
      asymmetry in module docstring / handover section 1 correction.
    - kind="correct": must complete (cursor reaches the end).
    - kind="artifact": must ALSO complete. The word was said correctly;
      only the capture is defective, so a stall is a false positive the
      learner cannot act on. EXPECTED TO FAIL under the current
      record-then-score path -- that is the point of the category.
    - kind="error" or "weak": must NOT complete (must stall somewhere).
      Both expect a stall -- a weak articulation SHOULD stop the green
      edge, since the device's whole design is that the learner repeats
      the word and it then confirms. The two are separated for reporting,
      not for expected behaviour.
      If EXPECTED_STALL_WORD is set for this file, cursor_final must
      equal it exactly; otherwise this weaker check is all that runs.
    """
    if r["false_confirms"]:
        return "FAIL: false confirm"
    if r["kind"] in ("correct", "artifact"):
        if r["completed"]:
            return "PASS"
        if r["kind"] == "artifact":
            return ("KNOWN-FAIL: capture artifact stalled "
                    f"at w{r['cursor_final']} (false positive)")
        return "FAIL: correct recording stalled"
    # error or weak: both must stall
    label = r["kind"]
    if r["expected_stall"] is not None:
        return ("PASS" if r["cursor_final"] == r["expected_stall"]
                else f"FAIL: expected stall at w{r['expected_stall']}, "
                     f"got w{r['cursor_final']}")
    return ("PASS" if not r["completed"]
            else f"FAIL: {label} recording completed without stalling")


def summarise(results):
    n = len(results)
    # KNOWN-FAIL is reported separately from FAIL: it is a documented open
    # limitation with a named owner (Phase 6 capture), not a regression.
    # Counting it as a plain failure would create pressure to relabel it
    # away; counting it as a pass would hide it.
    known = [r for r in results if _verdict(r).startswith("KNOWN-FAIL")]
    fails = [r for r in results if _verdict(r).startswith("FAIL")]
    n_pass = n - len(fails) - len(known)

    print(f"\n{'=' * 70}")
    print(f"{n_pass}/{n} PASS"
          + (f"   ({len(known)} known-fail)" if known else ""))

    # break down by kind, so a future model's stalls can be attributed --
    # stalling on a substitution and stalling on weak articulation are both
    # correct, but they measure different capabilities
    by_kind = {}
    for r in results:
        k = r.get("kind", "?")
        ok = _verdict(r).startswith("PASS")
        p, t = by_kind.get(k, (0, 0))
        by_kind[k] = (p + (1 if ok else 0), t + 1)
    print("  by kind: " + "  ".join(
        f"{k} {p}/{t}" for k, (p, t) in sorted(by_kind.items())))

    for r in known:
        print(f"  KNOWN  {r['file']}: {_verdict(r)}")
    for r in fails:
        print(f"  FAIL   {r['file']}: {_verdict(r)}")
    return len(fails)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_selftest(a):
    """The harness must FAIL the random stub. If this prints PASS, the
    harness is not testing anything and must not be trusted."""
    descriptions = load_descriptions(a.descriptions)
    model = RandomModel(seed=a.seed)
    print(f"model: {model.name}  (expected: mostly FAIL, especially on "
          f"'correct' recordings -- random scores rarely confirm 3+ words "
          f"in a row against a fixed threshold)\n")
    results = evaluate(model, a.recs, descriptions, a.text, a.threshold,
                       a.verbose)
    n_fail = summarise(results)
    if n_fail == 0:
        print("\n*** HARNESS SELFTEST FAILED ***")
        print("A random model passed. The harness measures nothing until")
        print("this is fixed -- do not trust any model result until a")
        print("random stub reliably fails here.")
        return 1
    print(f"\nharness selftest PASSED (the random model correctly failed "
          f"{n_fail}/{len(results)})")
    return 0


def cmd_run(a):
    descriptions = load_descriptions(a.descriptions)
    cls = MODELS[a.model]
    kwargs = {}
    if a.model == "random":
        kwargs["seed"] = a.seed
    else:
        kwargs.update(model=a.whisper_model, text=a.text, device=a.device)
    model = cls(**kwargs)
    print(f"model: {model.name}\n")
    results = evaluate(model, a.recs, descriptions, a.text, a.threshold,
                       a.verbose)
    n_fail = summarise(results)
    return 1 if n_fail else 0


def main():
    ap = argparse.ArgumentParser(description="v6 streaming eval harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common_args = [
        ("--recs", "recs"), ("--descriptions", "recs/descriptions.txt"),
        ("--text", "texts/quran-simple-plain.txt"),
        ("--threshold", 0.5),
    ]

    s = sub.add_parser("selftest")
    for name, default in common_args:
        s.add_argument(name, default=default,
                       type=type(default) if not isinstance(default, str)
                       else str)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_selftest)

    r = sub.add_parser("run")
    for name, default in common_args:
        r.add_argument(name, default=default,
                       type=type(default) if not isinstance(default, str)
                       else str)
    r.add_argument("--model", choices=list(MODELS), default="teacher_worst_token")
    r.add_argument("--whisper-model", default="whisper-base-quran")
    r.add_argument("--device", default="cpu")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    a = ap.parse_args()
    sys.exit(a.func(a) or 0)


if __name__ == "__main__":
    main()
