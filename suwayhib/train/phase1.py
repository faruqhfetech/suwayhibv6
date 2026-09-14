#!/usr/bin/env python3
"""
Suwayhib v6 -- Phase 1 decoding-aware calibration: a LEARNED rejection
model, replacing decode.py's hand-set global tau.

    python -m suwayhib.train.phase1 extract --ckpt runs/head/best.pt \
        --out cache/phase1/features.jsonl
    python -m suwayhib.train.phase1 train --features cache/phase1/features.jsonl \
        --out runs/phase1/model.pkl

WHY THIS EXISTS
---------------
decode.py's StreamingDecoder accepts or rejects a word with ONE hand-set
scalar (tau) compared against ONE aggregate number (the lattice's
length-normalised exit score). The testset tau sweep showed this is not
a small tuning gap: "correct confirmed" is non-monotonic in tau, and no
single global threshold serves every phone/word equally well (phones
differ hugely in natural duration and confusability -- a 1-phone word
and a 9-phone madd-lengthened word do not deserve the same margin).

The literature on this is direct and specific (see suwayhib's own
research notes / conversation history for full citations; summarised
here because this file's design follows them):

  - D. Qiu et al., "Learning Word-Level Confidence for Subword
    End-to-End ASR" (Google, ICASSP 2021, arXiv:2103.06716) and
    A. Laptev & B. Ginsburg, "Fast Entropy-Based Methods of Word-Level
    Confidence Estimation..." (NVIDIA, ASRU 2023, arXiv:2212.08703):
    freeze the ASR model, train a SEPARATE lightweight classifier on
    decode-time features (posterior, entropy, duration/alignment
    stats), supervised by correct/incorrect labels obtained by decoding
    against known ground truth. That is exactly this file's design --
    it is the lowest-risk, no-retraining phase of closing the gap
    between "the model is a good phone recogniser" (what CTC training
    optimises) and "the system makes good accept/reject decisions"
    (what decode.py actually needs).

  - R. Sukkar & C.-H. Lee, "Vocabulary independent discriminative
    utterance verification..." (IEEE Trans. Speech and Audio Proc.,
    1996), building on Juang & Katagiri's Minimum Classification Error
    framework (1992): the historical root of "train/calibrate the
    accept/reject statistic directly" rather than thresholding a proxy
    objective after the fact. This file's classifier is the modern,
    small-data-friendly version of that idea; a deeper version (folding
    this into the phone head's own training loss, per M. Sun et al.'s
    max-pooling loss training, SLT 2016, arXiv:1705.02411) is Phase 2,
    not attempted here.

WHAT THIS DOES NOT DO
----------------------
This does not retrain the phone head. It is a calibration layer bolted
onto whichever checkpoint you point it at (today's runs/head/best.pt,
or a future train_stream.py checkpoint) -- re-run `extract` + `train`
against a new checkpoint any time one ships, the same way decode.py's
tau sweep already gets re-run per checkpoint.

DATA
----
testset/'s 2,000 labelled synthetic items already carry exactly the
supervision this needs (is_error, word_idx) -- no new data collection.
`extract` runs the REAL StreamingDecoder (unmodified, same class
decode.py uses) frame-by-frame over each item and, at the moment each
word's attempt CONCLUDES (confirmed, or the utterance ran out of frames
while stuck on it), snapshots a feature vector for that word. This
matches production timing exactly: a word later in the utterance that
the decoder never reaches (because an earlier word never confirmed)
simply does not appear in the training data, the same way it would
never be evaluated in real deployment.

Label per word: 0 (should have been rejected) if this is the specific
word_idx an is_error=1 item corrupted; 1 (should have been accepted)
otherwise -- including every word of an is_error=0 item, and every
OTHER word of an is_error=1 item (only one word is corrupted per
item). control_seam items are is_error=0 by the project's own
established convention (a splice with no real pronunciation error), so
every one of their words is label 1, same as decode.py/score.py already
treat them.

FEATURES
--------
The first version of this file used only 8 aggregate-over-the-whole-
word features (norm_score, raw_score, n_phones, n_frames, max_dwell,
frames_per_phone, span_ratio, reached_exit). Measured result: neither a
logistic regression nor a small MLP over those 8 could clearly beat
just thresholding norm_score directly -- decode.py's existing aggregate
score was already carrying nearly all the signal available at THAT
granularity. That is why the 4 features below exist: they come from
WordLattice.traceback()/phone_segment_stats(), i.e. from knowing which
FRAMES the winning alignment assigned to which INDIVIDUAL phone, not
just the whole word's average.

  norm_score          the SAME statistic decode.py's tau thresholds today
  raw_score           un-normalised log-prob (how MUCH evidence, not
                      just its average)
  n_phones            word length
  n_frames            wall-clock frames the attempt actually ran for
  max_dwell           longest any single state self-looped
  frames_per_phone    pace -- n_frames / n_phones
  span_ratio          phone-frames actually consumed / n_phones
  reached_exit        0/1 -- did the lattice ever reach a scorable
                      alignment at all
  worst_phone_score   the WORST individual phone's mean log-prob along
                      the winning path -- directly targets the "mean-
                      vs-worst-token trap" (handover section 6: one
                      badly-matched phone barely moves an AVERAGED
                      score but IS the error) that norm_score, being an
                      average, structurally cannot see
  worst_phone_entropy peak per-frame entropy of the full output
                      distribution anywhere on the winning path -- high
                      entropy means the model itself was uncertain,
                      independent of which symbol it leaned toward
  n_rushed_phones     count of phone segments occupying fewer than 2
                      frames -- a per-phone version of the same
                      physical-implausibility concern min_frames_per_phone
                      addresses at the whole-word level in decode.py
  blank_frac          fraction of the attempt's frames NOT assigned to
                      any phone by the winning path
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from ..core import phones as P
from ..core import reftext as RT
from ..pipeline import decode as D
from ..pipeline import score as S

FEATURE_KEYS = ["norm_score", "raw_score", "n_phones", "n_frames",
                "max_dwell", "frames_per_phone", "span_ratio",
                "reached_exit", "worst_phone_score", "worst_phone_entropy",
                "n_rushed_phones", "blank_frac"]

# Sentinels for the "never reached a scorable alignment" case -- keeps
# every feature a finite float so a plain sklearn classifier can use it,
# while `reached_exit` tells the classifier explicitly when these two
# are sentinels rather than real measurements.
_NORM_SENTINEL = -50.0
_RAW_SENTINEL = -500.0


def extract_features(lat):
    """The original 8 aggregate features PLUS 4 per-phone ones from
    WordLattice.phone_segment_stats() -- worst_phone_score in particular
    targets the "mean-vs-worst-token trap" (handover section 6: one
    badly-matched phone barely moves an averaged score but IS the
    error) that norm_score alone cannot see, since norm_score is an
    AVERAGE over the whole word."""
    n_phones = (lat.L - 1) // 2
    raw, span = lat.exit_raw_and_span
    reached = raw > D.NEG_INF
    stats = lat.phone_segment_stats() if reached else None
    worst = (stats["worst_phone_score"] if stats and
             stats["worst_phone_score"] is not None else None)
    return {
        "norm_score": float(raw / span) if reached else _NORM_SENTINEL,
        "raw_score": float(max(raw, _RAW_SENTINEL)) if reached
                    else _RAW_SENTINEL,
        "n_phones": int(n_phones),
        "n_frames": int(lat.n_frames),
        "max_dwell": int(lat.max_dwell),
        "frames_per_phone": float(lat.n_frames / max(n_phones, 1)),
        "span_ratio": float(span / max(n_phones, 1)) if reached else 0.0,
        "reached_exit": int(reached),
        "worst_phone_score": float(max(worst, _RAW_SENTINEL))
                             if worst is not None else _RAW_SENTINEL,
        "worst_phone_entropy": float(stats["worst_phone_entropy"])
                               if stats else 0.0,
        "n_rushed_phones": int(stats["n_rushed_phones"]) if stats else 0,
        "blank_frac": float(stats["blank_frac"]) if stats else 0.0,
    }


def cmd_extract(a):
    scorer = S.LiveScorer(a.ckpt, a.device)
    rt = RT.RefText(a.text)
    rows = S.load_testset(a.testset)
    if a.n:
        rows = rows[:a.n]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = skipped = 0

    with open(out, "w", encoding="utf-8") as fh:
        for row in rows:
            if not row["_path"].exists():
                skipped += 1
                continue
            words_text = rt.words(row["surah"], row["ayah"])
            if len(words_text) != row["n_words"]:
                skipped += 1
                continue
            words = [[scorer.vocab[p] for p in P.word_to_phones_iqra(w)
                      if p in scorer.vocab] for w in words_text]

            audio = S.load_audio(row["_path"])
            logp = D.frame_logp(scorer, audio)

            dec = D.StreamingDecoder(words, blank_id=0, tau=a.tau,
                                     dwell_cap=a.dwell_cap,
                                     commit_lag=a.commit_lag)

            def _write(lattice, word_idx):
                nonlocal n_written
                bad_word = row["is_error"] and word_idx == row.get("word_idx")
                feat = extract_features(lattice)
                feat.update({
                    "label": 0 if bad_word else 1,
                    "file": row["file"], "word_idx": word_idx,
                    "split": row["split"],
                    "error_type": row.get("error_type"),
                    "tau": a.tau,
                })
                fh.write(json.dumps(feat) + "\n")
                n_written += 1

            for t in range(logp.shape[0]):
                prev_lattice, prev_cursor = dec.lattice, dec.cursor
                ev = dec.step(logp[t])
                if ev["confirmed"]:
                    _write(prev_lattice, prev_cursor)
                if dec.done:
                    break
            if not dec.done and dec.lattice is not None:
                _write(dec.lattice, dec.cursor)

    print(f"wrote {n_written} word-level examples ({skipped} items "
          f"skipped) -> {out}")
    return 0


def _load_xy(rows, split):
    sub = [r for r in rows if r["split"] == split]
    X = np.array([[r[k] for k in FEATURE_KEYS] for r in sub], dtype=float)
    y = np.array([r["label"] for r in sub], dtype=int)
    return X, y, sub


def cmd_train(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    rows = [json.loads(l) for l in open(a.features, encoding="utf-8")]
    Xtr, ytr, _ = _load_xy(rows, "dev")
    Xva, yva, va_rows = _load_xy(rows, "holdout")
    if len(Xtr) == 0 or len(Xva) == 0:
        raise SystemExit("empty train or val split -- re-run extract "
                         "and check testset/labels.jsonl's split field")

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)

    if a.model == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=3000,
                            random_state=0)
    clf.fit(Xtr_s, ytr)
    proba = clf.predict_proba(Xva_s)[:, 1]

    print(f"\ntrain n={len(ytr)} (accept-labelled {int(ytr.sum())})   "
          f"val n={len(yva)} (accept-labelled {int(yva.sum())})")

    # Reference row: what the CURRENT hand-set-tau decision (the thing
    # this file exists to replace) does on this EXACT same held-out set
    # of decision points -- not a different metric on a different split.
    tau_used = va_rows[0]["tau"] if va_rows else a.tau
    manual_accept = (Xva[:, FEATURE_KEYS.index("reached_exit")] > 0) & \
                    (Xva[:, FEATURE_KEYS.index("norm_score")] > tau_used)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        yva, manual_accept.astype(int), average="binary", pos_label=1,
        zero_division=0)
    print(f"\nmanual tau={tau_used:<6} precision {mp:.4f}  recall {mr:.4f}"
          f"  f1 {mf1:.4f}   <- what decode.py does today, for comparison")

    print(f"\n{'thresh':>7} | {'precision':>9} | {'recall':>7} | "
          f"{'f1':>6}   (all w.r.t. label=1 'should accept')")
    best_f1, best_th = -1.0, None
    for th in np.arange(0.05, 1.0, 0.05):
        pred = (proba >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            yva, pred, average="binary", pos_label=1, zero_division=0)
        flag = ""
        if f1 > best_f1:
            best_f1, best_th, flag = f1, th, "  <- best f1"
        print(f"{th:7.2f} | {p:9.4f} | {r:7.4f} | {f1:6.4f}{flag}")

    print(f"\nA FALSE ACCEPT (predicting 1 for a truly bad word) costs far")
    print(f"more than a false reject (handover: false confirm is")
    print(f"unrecoverable, false stall costs one repetition). Pick the")
    print(f"deployed threshold from the HIGH-PRECISION end of this table,")
    print(f"not the max-F1 row -- max-F1 optimises a symmetric cost this")
    print(f"project does not have.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"scaler": sc, "model": clf, "features": FEATURE_KEYS},
                   fh)
    print(f"\nsaved -> {out}")
    return 0


def load_model(path):
    """For decode.py (or anything else) to consume later: load the
    pickled {scaler, model, features} bundle and return a function
    lattice -> P(accept)."""
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    sc, clf, keys = bundle["scaler"], bundle["model"], bundle["features"]

    def predict(lattice):
        feat = extract_features(lattice)
        x = np.array([[feat[k] for k in keys]], dtype=float)
        return float(clf.predict_proba(sc.transform(x))[0, 1])

    return predict


def main():
    ap = argparse.ArgumentParser(
        description="Phase 1: learned rejection model for decode.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--ckpt", default="runs/head/best.pt")
    e.add_argument("--device", default="cuda")
    e.add_argument("--testset", default="testset")
    e.add_argument("--text", default="texts/quran-simple-plain.txt")
    e.add_argument("--tau", type=float, default=-1.5)
    e.add_argument("--dwell-cap", type=int, default=40)
    e.add_argument("--commit-lag", type=int, default=8)
    e.add_argument("--n", type=int, default=0, help="0 = all items")
    e.add_argument("--out", default="cache/phase1/features.jsonl")
    e.set_defaults(func=cmd_extract)

    t = sub.add_parser("train")
    t.add_argument("--features", default="cache/phase1/features.jsonl")
    t.add_argument("--model", choices=["logreg", "mlp"], default="logreg",
                   help="logreg is the honest default for this little "
                        "data; try mlp once there is enough of it to not "
                        "just memorise dev")
    t.add_argument("--out", default="runs/phase1/model.pkl")
    t.set_defaults(func=cmd_train)

    a = ap.parse_args()
    return a.func(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
