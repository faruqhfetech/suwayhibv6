#!/usr/bin/env python3
"""
Suwayhib v6 -- score the trained phone head.

    python -m suwayhib.pipeline.score recs --ckpt runs/head/best.pt
    python -m suwayhib.pipeline.score testset --ckpt runs/head/best.pt   # inspects schema first

WHY THIS EXISTS, AND WHY IT IS NOT A QuranMB.v2 SCORER
---------------------------------------------------------
QuranMB.v2's ground truth (Reference_phn, Annotated_phn, keyed by ID) is
NOT publicly distributed -- confirmed by tracing the actual scoring repo
(github.com/Iqra-Eval/interspeech_IqraEval), which ships align_data.py and
metric.py but no truth CSV, plus the existence of
huggingface.co/spaces/IqraEval/Leaderboard, a submission portal that
scores predictions server-side. That is a deliberately blind test set,
by design, so the benchmark stays a fair comparison point. F1 ~= 0.4414
cannot be computed locally by any amount of correct code; it requires a
submission.

So Gate 1's local validation instead uses data this project already
controls: recs/ (six recordings, ground truth from listening, corrected
today into four categories) and testset/ (2,000 labelled synthetic items
with the artifact control). This is the ORIGINAL v3/v4/v5 evaluation
method -- the QuranMB detour was worth attempting since it would have
given an external number, but it was never the only path.

THE METRIC MATCHES THE OFFICIAL ONE, EVEN THOUGH THE DATA DOES NOT
---------------------------------------------------------------------
Correct_Rate and Accuracy below are deliberately the same formulas as
Iqra-Eval/interspeech_IqraEval/mdd_eval/metric.py: Levenshtein-align
predicted phones against ground truth, then

    Accuracy     = 1 - (D + S + I) / len(ground_truth)
    Correct_Rate = 1 - (D + S)     / len(ground_truth)

(HTK-style %Corr vs %Acc; PER = 1 - Accuracy.) Reusing their exact
formula means a future submission's official F1 and this script's local
numbers are at least measuring the same underlying quantity, even though
the SET they are measured on differs.

LIVE FEATURES, NOT THE CACHE
------------------------------
train.py reads pre-extracted features from extract.py's cache. recs/ and
testset/ were never extracted -- there is no cache for them, and building
one for six files would be pure overhead. So this script runs the frozen
XLS-R encoder LIVE, in memory, on each item, using the SAME layer band
the checkpoint was trained on (read from the checkpoint's own config,
not re-specified here, so a mismatch is impossible by construction).
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..core import model as M
from ..core import phones as P
from ..core import reftext as RT
from . import align_phonemes as AP
from . import extract as E
from . import harness as H

SR = 16000


# --------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------

def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = M.PhoneHead(
        n_phones=cfg["n_phones"], n_layers=cfg["n_layers"], dim=cfg["dim"],
        blocks=cfg["blocks"], heads=cfg["heads"], kernel=cfg["kernel"],
        ff_mult=cfg["ff_mult"], fusion=cfg["fusion"])
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    vocab = ck["vocab"]
    inv = {v: k for k, v in vocab.items()}
    print(f"loaded {path}  epoch {ck['epoch']}  val {ck['val']:.4f}")
    print(f"  layers {cfg['layers']}  dim {cfg['dim']}  blocks {cfg['blocks']}"
          f"  params {model.n_params()/1e6:.2f}M")
    return model, vocab, inv, cfg


# --------------------------------------------------------------------------
# live feature extraction -- reuses extract.py's Encoder, on raw audio
# --------------------------------------------------------------------------

class LiveScorer:
    def __init__(self, ckpt_path, device="cuda"):
        self.device = torch.device(device)
        self.model, self.vocab, self.inv, self.cfg = load_checkpoint(
            ckpt_path, self.device)
        self.layers = self.cfg["layers"]
        self.enc = E.Encoder(device=device, fp16=(self.device.type == "cuda"),
                             max_layer=max(self.layers))

    def phones(self, audio):
        """Raw float32 audio -> decoded phone symbol list (greedy CTC)."""
        feats = self.enc([audio], self.layers)[0]          # (T, L, D) fp16
        x = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0).to(
            self.device)
        with torch.no_grad():
            logits = self.model(x)["logits"][0]             # (T, C)
            ids = logits.argmax(dim=-1).cpu().tolist()
        return greedy_ctc_decode(ids, self.inv)


def greedy_ctc_decode(ids, inv):
    """Argmax per frame -> collapse repeats -> drop blank (index 0)."""
    out = []
    prev = None
    for i in ids:
        if i != prev:
            if i != 0:
                out.append(inv[i])
            prev = i
    return out


# --------------------------------------------------------------------------
# alignment / metric -- matches Iqra-Eval/interspeech_IqraEval/mdd_eval
# exactly (see module docstring), reusing align_phonemes.py's verified
# Levenshtein implementation rather than a second copy.
# --------------------------------------------------------------------------

def correct_rate_and_accuracy(truth, pred):
    """-> (correct_rate, accuracy, D, S, I, C, len(truth))

    truth is SEQ1 in their convention: deletions/substitutions are counted
    against IT (ground truth phones the prediction failed to produce or
    got wrong). Insertions (extra predicted phones with no ground-truth
    counterpart) count against accuracy but not correct_rate, matching
    their metric.py precisely.
    """
    pairs = AP.align(truth, pred)
    D = sum(1 for t, p in pairs if t is not None and p is None)
    ins = sum(1 for t, p in pairs if t is None and p is not None)
    S = sum(1 for t, p in pairs if t is not None and p is not None and t != p)
    C = len(truth) - D - S
    acc = 1 - (D + S + ins) / max(len(truth), 1)
    corr = 1 - (D + S) / max(len(truth), 1)
    return corr, acc, D, S, ins, C, len(truth)


# --------------------------------------------------------------------------
# recs/ -- known schema, known ground truth (today's four-category work)
# --------------------------------------------------------------------------

def load_audio(path):
    import subprocess
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"decode failed: {path}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def cmd_recs(a):

    scorer = LiveScorer(a.ckpt, a.device)
    rt = RT.RefText(a.text)
    descriptions = H.load_descriptions(a.descriptions)

    print(f"\n{'file':<16} {'kind':<9} {'corr':>6} {'acc':>6} "
          f"{'D':>3} {'S':>3} {'I':>3}   canonical vs predicted")
    for d in descriptions:
        path = Path(a.recs) / d["file"]
        if not path.exists():
            continue
        words = rt.words(d["surah"], d["ayah"])
        canon = []
        for w in words:
            canon.extend(P.word_to_phones_iqra(w))

        audio = load_audio(path)
        pred = scorer.phones(audio)
        corr, acc, D, S, I, C, n = correct_rate_and_accuracy(canon, pred)

        print(f"{d['file']:<16} {d['kind']:<9} {corr:6.3f} {acc:6.3f} "
              f"{D:3d} {S:3d} {I:3d}")
        if a.verbose:
            print(f"    canon: {' '.join(canon)}")
            print(f"    pred : {' '.join(pred)}")

    print("\nHow to read this, per kind:")
    print("  correct  -> corr/acc should be HIGH (little D/S)")
    print("  error    -> corr/acc should be LOWER, with S concentrated at")
    print("              the known bad word (see EXPECTED_STALL_WORD in")
    print("              harness.py for the word index)")
    print("  weak     -> similar to error: the model SHOULD show a")
    print("              substitution near the under-articulated word,")
    print("              since the acoustic evidence genuinely is weak")
    print("  artifact -> the word itself was said correctly; errors here,")
    print("              if any, should cluster at the START of the")
    print("              sequence (the clipped onset), not distributed")
    print("              through it")
    print("\nThis is a WHOLE-UTTERANCE greedy decode with no duration")
    print("prior and no filler path -- i.e. it exercises the trained head")
    print("alone, not the Phase 3 graph decoder. Treat it as 'does the")
    print("head's output make phonetic sense', not as a streaming test;")
    print("harness.py remains the tool for streaming behaviour.")
    return 0


# --------------------------------------------------------------------------
# testset/ -- SCHEMA NOT YET CONFIRMED. This inspects rather than assumes.
# --------------------------------------------------------------------------

def load_testset(testset_dir):
    """-> list of label rows, each augmented with a resolved audio path.

    SCHEMA (confirmed by inspection, not assumed):
      error_type   e.g. "madd_shortening", "word_repetition", "control_seam"
      is_error     0 or 1 -- ground truth for detection
      word_idx     which word the injected error targets (meaningless for
                   is_error=0 items)
      tier         "structural" | "prosodic" | "control" | others
      reciter, surah, ayah, n_words
      split        "dev" | "holdout" -- holdout_reciters in meta.json are
                   Husary and Minshawy, so split follows FROM the reciter,
                   not an independent random draw
      orig_duration_ms, new_duration_ms  -- corruption can change length
    """
    labels = []
    with open(Path(testset_dir) / "labels.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            row["_path"] = Path(testset_dir) / row["file"]
            labels.append(row)
    return labels


def cmd_testset(a):
    """Score the trained head against testset/'s 2000 labelled synthetic
    items, reporting the ARTIFACT CONTROL FIRST -- per the project's own
    standing rule (handover section 3 / v6 plan section 5): a model that
    flags control_seam items (spliced audio, no real error, 10% of the
    set) is detecting edit artifacts, not mispronunciation, and that
    number must be reported before anything else is trusted.

    This scores DETECTION, not phone accuracy: for each item, decode the
    predicted phones, align against the canonical sequence for that verse,
    and call it "flagged" if any substitution or deletion lands at
    word_idx (mapped from phones to words via the same word boundaries
    phones.py would produce). is_error=0 items should not be flagged;
    is_error=1 items should be, ideally at the right word.

    holdout split (Husary, Minshawy) is reported SEPARATELY from dev,
    since train.py's current train/val split does NOT respect this
    reciter-level hold-out (a known gap -- see train.py's docstring note
    added after this was discovered). Numbers on holdout are the only
    ones in this run that test genuine generalisation; dev numbers may be
    inflated by the reciter having been seen, even if this exact ayah/
    corruption was not.
    """

    scorer = LiveScorer(a.ckpt, a.device)
    rt = RT.RefText(a.text)
    labels = load_testset(a.testset)
    if a.n:
        labels = labels[:a.n]

    by_split = {"dev": Counter(), "holdout": Counter()}
    rows = []

    for row in labels:
        if not row["_path"].exists():
            continue
        words = rt.words(row["surah"], row["ayah"])
        canon = []
        word_spans = []          # (start_idx_in_canon, end_idx_in_canon)
        for w in words:
            ph = P.word_to_phones_iqra(w)
            word_spans.append((len(canon), len(canon) + len(ph)))
            canon.extend(ph)

        audio = load_audio(row["_path"])
        pred = scorer.phones(audio)
        _, _, D, S, I, C, n = correct_rate_and_accuracy(canon, pred)

        pairs = AP.align(canon, pred)
        # map each canon-side error position back to a word index, so
        # "flagged at word_idx" can be checked against ground truth
        flagged_words = set()
        pos = 0
        for t, p in pairs:
            if t is not None:
                if p is None or p != t:
                    for wi, (lo, hi) in enumerate(word_spans):
                        if lo <= pos < hi:
                            flagged_words.add(wi)
                            break
                pos += 1

        split = row["split"]
        flagged_any = len(flagged_words) > 0
        # BUG FOUND ON FIRST RUN: the artifact-control check originally
        # used flagged_any (ANY mismatch anywhere in the utterance) as
        # "did the model react to the splice". That is the wrong test --
        # per-utterance phone accuracy on recs/ ran 55-87%, so almost any
        # multi-word utterance has SOME scattered error somewhere whether
        # or not it was spliced. flagged_any measures "is this
        # transcription imperfect", which it always partly is; it does
        # not measure sensitivity to the seam specifically. The count
        # also silently disagreed with the by-error-type breakdown below
        # (28 vs 15 control items) because that loop used a DIFFERENT,
        # inconsistent criterion -- both are unified here into one
        # definition so they cannot diverge again: flagged_target, an
        # error landing AT word_idx, which for a control item IS the
        # splice location (injector.py places it there), and for a real
        # error IS the corrupted word. Same test, same code path, for
        # both control and real items.
        flagged_target = row.get("word_idx") in flagged_words
        # BUG FOUND ON FULL RUN: tier=="control" turned out to group BOTH
        # control_seam (200 items) AND clean (400 items) together -- the
        # 600 = 461+139 top-line total only made sense once this was
        # traced. clean scored a perfect 0% false-positive rate, which
        # diluted the reported artifact-control number from the true
        # 17% (34/200 control_seam items flagged at the splice point)
        # down to a blended 5.7%. Match on error_type specifically, which
        # is the field the report's own label claims to measure.
        is_ctrl = row["error_type"] == "control_seam"

        key = ("TP" if row["is_error"] and flagged_target else
              "FN" if row["is_error"] and not flagged_target else
              "FP" if not row["is_error"] and flagged_target else "TN")
        by_split[split][key] += 1
        rows.append({**row, "flagged_any": flagged_any,
                    "flagged_target": flagged_target, "is_ctrl": is_ctrl,
                    "D": D, "S": S, "I": I})

    # Derived from `rows` AFTER the loop, from the SAME flagged_target
    # field every other number in this function uses -- one source of
    # truth, so this count cannot silently disagree with anything below
    # it the way the first version did.
    print("=" * 70)
    print("ARTIFACT CONTROL (report first, always):")
    for split in ("dev", "holdout"):
        ctrl = [r for r in rows if r["split"] == split and r["is_ctrl"]]
        if ctrl:
            f = sum(r["flagged_target"] for r in ctrl)
            print(f"  {split:8s}  {f}/{len(ctrl)} control_seam items "
                  f"flagged AT THE SPLICE POINT ({f/len(ctrl):.1%}) "
                  f"-- should be near 0%")
    print(f"  (total items scored: {len(rows)}, cross-check against "
          f"n below)")
    print()

    for split in ("dev", "holdout"):
        c = by_split[split]
        tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
        n = tp + fn + fp + tn
        if n == 0:
            continue
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"{split.upper()}  (n={n}"
              f"{'  -- Husary/Minshawy, genuine hold-out' if split=='holdout' else ''})")
        print(f"  TP {tp}  FN {fn}  FP {fp}  TN {tn}")
        print(f"  precision {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")
        print()

    print("Note: this F1 is word-level flag detection on OUR synthetic")
    print("corruptions, not IqraEval's phoneme-level TA/FR/FA/TR F1 on")
    print("QuranMB.v2 -- the two are not comparable numbers, only")
    print("analogous ones. See module docstring.")

    if a.by_error_type:
        print("\nby error_type:")
        # SAME criterion as the artifact-control section and the TP/FN/
        # FP/TN tally above: flagged_target, uniformly. For control
        # items "success" is NOT flagged (is_error=0); for real error
        # types "success" is flagged (is_error=1). No item is double
        # counted or dropped -- sum of et[] must equal len(rows).
        et = Counter()
        et_success = Counter()
        for r in rows:
            et[r["error_type"]] += 1
            want = r["flagged_target"] if r["is_error"] else \
                not r["flagged_target"]
            et_success[r["error_type"]] += int(want)
        assert sum(et.values()) == len(rows), \
            "accounting bug: error_type counts do not sum to total rows"
        for k in sorted(et):
            print(f"  {k:<24} {et_success[k]:4d}/{et[k]:<4d} "
                  f"({et_success[k]/et[k]:.1%})")
    return 0


# --------------------------------------------------------------------------
# leaderboard submission
# --------------------------------------------------------------------------

def cmd_submit(a):
    """Build a QuranMB.v2 leaderboard submission CSV: columns ID, Labels.

    Runs ONLY the trained head over ALREADY-CACHED features -- extract.py
    ran XLS-R over every QuranMB.v2 item already; redoing that live would
    be pure wasted GPU time.

    THE ID PROBLEM, FOUND WRITING THIS: extract.py's iter_hf keys cache
    items by r.get('id', i) -- lowercase. QuranMB.v2's real column is
    'ID' (uppercase; confirmed via d.column_names == ['ID', 'audio']).
    The lookup missed it and silently fell back to the positional index,
    so cache/feats_quranmb is keyed "QuranMB.v2/0".."QuranMB.v2/1641",
    not the real "00000_00001"-style IDs the leaderboard needs.

    Recovered here by reloading the dataset (cheap: the ID column is
    plain strings, does not trigger audio decode) and zipping it
    POSITIONALLY against the cache. That is only valid if row order is
    stable between the extraction run and this reload -- NOT assumed:
    verified below by comparing recorded frame counts against a fresh
    duration check on a sample of items, since a silent mismatch here
    would submit WRONG predictions under RIGHT-looking IDs, the one
    failure mode that would not look broken.
    """
    import csv
    import json as _json
    cache = E.FeatureCache(a.cache)
    model, vocab, inv, cfg = load_checkpoint(a.ckpt, torch.device(a.device))
    device = torch.device(a.device)

    from datasets import Audio, load_dataset
    ds = load_dataset("IqraEval/QuranMB.v2", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    real_ids = ds["ID"]
    print(f"cache items: {len(cache)}   dataset rows: {len(real_ids)}")
    if len(cache) != len(real_ids):
        raise SystemExit("COUNT MISMATCH -- positional mapping is not "
                         "trustworthy. Investigate before submitting.")

    meta = {}
    with open(Path(a.cache) / "meta.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = _json.loads(line)
            meta[r["key"]] = r["frames"]

    n_check = min(a.check_n, len(real_ids))
    mism = 0
    for i in range(n_check):
        key = f"QuranMB.v2/{i}"
        wav, sr = E._decode_audio_field(ds[i]["audio"])
        approx_frames = int(len(wav) / sr * 50)
        if abs(approx_frames - meta.get(key, -999)) > 3:
            mism += 1
    if mism:
        raise SystemExit(f"{mism}/{n_check} items show a duration "
                         f"mismatch between cache and a fresh reload -- "
                         f"the ordering assumption is NOT safe. Do not "
                         f"submit; the ID-to-audio mapping would be wrong.")
    print(f"order check passed on {n_check} items -- positional mapping "
          f"is trustworthy")

    rows = []
    for i, real_id in enumerate(real_ids):
        feats = cache.get(f"QuranMB.v2/{i}")
        x = torch.from_numpy(feats.astype("float32")).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)["logits"][0]
            ids = logits.argmax(dim=-1).cpu().tolist()
        rows.append((real_id, " ".join(greedy_ctc_decode(ids, inv))))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(real_ids)}")

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "Labels"])
        w.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} rows)")
    print(f"sample: {rows[0]}")
    print("\nASSUMPTION WORTH CHECKING: 'Labels' is filled with space-")
    print("separated phones in IqraEval's own spelling (matches")
    print("align_data.py's 'Prediction' column format from their local")
    print("scoring script). If the leaderboard rejects the format or")
    print("scores near zero, check the Metrics tab for the exact")
    print("expected format before assuming the model is bad.")
    return 0

def main():
    ap = argparse.ArgumentParser(description="score the v6 phone head")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("recs")
    r.add_argument("--ckpt", default="runs/head/best.pt")
    r.add_argument("--device", default="cuda")
    r.add_argument("--recs", default="recs")
    r.add_argument("--descriptions", default="recs/descriptions.txt")
    r.add_argument("--text", default="texts/quran-simple-plain.txt")
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_recs)

    t = sub.add_parser("testset")
    t.add_argument("--ckpt", default="runs/head/best.pt")
    t.add_argument("--device", default="cuda")
    t.add_argument("--testset", default="testset")
    t.add_argument("--text", default="texts/quran-simple-plain.txt")
    t.add_argument("--n", type=int, default=0, help="0 = all 2000")
    t.add_argument("--by-error-type", action="store_true")
    t.set_defaults(func=cmd_testset)

    sb = sub.add_parser("submit")
    sb.add_argument("--ckpt", default="runs/head/best.pt")
    sb.add_argument("--device", default="cuda")
    sb.add_argument("--cache", default="cache/feats_quranmb")
    sb.add_argument("--out", default="submission.csv")
    sb.add_argument("--check-n", type=int, default=30)
    sb.set_defaults(func=cmd_submit)

    a = ap.parse_args()
    rc = a.func(a) or 0
    sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
