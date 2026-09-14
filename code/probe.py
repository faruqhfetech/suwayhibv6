#!/usr/bin/env python3
"""
Suwayhib v6 -- XLS-R layer probe.

    python code/probe.py run --cache cache/probe
    python code/probe.py labels --cache cache/probe --n 5

WHAT THIS ANSWERS
-----------------
Which XLS-R layers carry phonetic information for Qur'anic recitation, so
the downstream head can be fed a narrow band instead of all 24 layers.

This is a COMPARISON BETWEEN LAYERS, not an attempt to build a good
phone recogniser. A linear probe on 1024-dim features has ~70K parameters
and 2.49 h at 50Hz gives ~448K frames, so the ranking stabilises long
before the absolute accuracy does. Do not read the accuracy numbers as
anything but relative.

WHAT A GOOD RESULT LOOKS LIKE
------------------------------
A SINGLE-PEAKED curve, peaking around layers 15-19. In 24-layer
wav2vec2-family models the early layers are acoustic (and carry speaker
identity, which we do not want), the middle-to-late layers are most
phonetic, and the last few drift toward the pretraining objective.

A FLAT CURVE IS A RED FLAG, and it points upstream, not at the model: it
would mean the frame labels are wrong, which means either phones.py or
the word-boundary-to-frame mapping below is broken. Check that before
concluding anything about layer choice.

Note also from extract.py info: layer 22's feature std is ~13.6 against
~3.0 for layers 10-17. That scale inflation in the top layers is a known
wav2vec2 property and is NOT evidence of information content -- which is
exactly why features are standardised per layer before fitting here. Left
unstandardised, the high-variance layers would win on scale alone.

THE FRAME-LABEL PROBLEM, AND THE ASSUMPTION IT FORCES
-------------------------------------------------------
The manifest gives WORD spans, verified. We need PHONE labels per frame.
Within a word we do not know where each phone begins.

This probe splits each word span UNIFORMLY across its phones. That is
demonstrably wrong at the single-frame level -- a long vowel occupies far
more time than a stop consonant -- but it is UNBIASED with respect to
layer choice: every layer is scored against the same imperfect labels, so
the comparison remains valid even though the absolute accuracy is
depressed. Since the probe's only job is ranking, that trade is
acceptable, and it avoids bootstrapping a forced aligner (an aligner on
top of quran-align's already-machine-estimated boundaries) purely to
answer a model-selection question.

--mode word offers a check on that assumption: it classifies which WORD a
frame belongs to, using only verified boundaries and no within-word
guessing. If the two modes disagree about where the peak is, the uniform
split is doing real damage and deserves a second look.

BOUNDARY FRAMES ARE DROPPED. --trim discards frames near word edges,
where the uniform split is least trustworthy and where coarticulation
genuinely blurs identity. This is a deliberate choice to measure the
layers rather than the labelling noise.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FRAMES_PER_SEC = 50


def _load(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------
# frame labelling
# --------------------------------------------------------------------------

def build_frame_labels(manifest, text_path, mode="phone", trim=1):
    """-> {cache_key: (frame_idx array, label array, label_names)}

    cache_key matches extract.py's iter_reference_library convention:
        "ref/" + audio_path with "/" -> "__" and ".wav" stripped
    """
    B = _load("bootstrap")
    P = _load("phones")

    text, _ = P.load_words_for_g2p(text_path)
    rows = B.load_manifest(manifest)
    ayahs = B.group_ayahs(rows)

    per_key = {}
    vocab = {}
    skipped = Counter()

    for (rec, surah, ayah), words in ayahs.items():
        key = "ref/" + words[0]["audio_path"].replace("/", "__").replace(
            ".wav", "")
        t = text.get((surah, ayah))
        if not t:
            skipped["no_text"] += 1
            continue
        toks = t.split()
        if len(toks) != len(words):
            # word-count disagreement between the manifest and the text is
            # exactly the class of bug bootstrap.py's checks exist to
            # catch; if it appears here, do not paper over it.
            skipped["word_count_mismatch"] += 1
            continue

        idx, lab = [], []
        for w, tok in zip(words, toks):
            f0 = w["start_ms"] * FRAMES_PER_SEC // 1000
            f1 = w["end_ms"] * FRAMES_PER_SEC // 1000
            if f1 - f0 < 2 * trim + 1:
                skipped["word_too_short"] += 1
                continue

            if mode == "word":
                units = [tok]
            else:
                units = P.word_to_phones(tok)
            if not units:
                skipped["no_units"] += 1
                continue

            # UNIFORM SPLIT. See module docstring for why this is
            # acceptable for ranking and wrong for absolute accuracy.
            edges = np.linspace(f0, f1, len(units) + 1).astype(int)
            for u, (a, b) in zip(units, zip(edges[:-1], edges[1:])):
                a, b = a + trim, b - trim
                if b <= a:
                    continue
                if u not in vocab:
                    vocab[u] = len(vocab)
                idx.extend(range(a, b))
                lab.extend([vocab[u]] * (b - a))

        if idx:
            per_key[key] = (np.array(idx, dtype=np.int64),
                            np.array(lab, dtype=np.int64))

    names = [None] * len(vocab)
    for u, i in vocab.items():
        names[i] = u
    return per_key, names, skipped


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------

def gather(cache, per_key, layer, max_frames, seed=0):
    """-> (X, y) for one layer, subsampled to max_frames."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    n = 0
    keys = [k for k in per_key if k in cache.index]
    rng.shuffle(keys)
    for k in keys:
        idx, lab = per_key[k]
        feats = cache.get(k, layer=layer)
        ok = idx < feats.shape[0]
        if not ok.all():
            # frames past the end mean the manifest span exceeds the audio
            # for this item; bootstrap.py verify checks this, so it should
            # be rare. Clip rather than crash, but it is worth noticing.
            idx, lab = idx[ok], lab[ok]
        if len(idx) == 0:
            continue
        Xs.append(feats[idx])
        ys.append(lab)
        n += len(idx)
        if n >= max_frames:
            break
    X = np.concatenate(Xs)[:max_frames]
    y = np.concatenate(ys)[:max_frames]
    return X, y


def fit_probe(X, y, n_classes, seed=0, iters=200, val_frac=0.2):
    """Multinomial logistic regression, standardised inputs.

    Standardisation is not optional here: layer 22's raw std is ~4x layer
    10's, and without it the probe would partly rank layers by scale.
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n_val = int(len(X) * val_frac)
    Xtr, ytr, Xva, yva = X[n_val:], y[n_val:], X[:n_val], y[:n_val]

    sc = StandardScaler().fit(Xtr)
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)

    # SGD with log loss rather than LogisticRegression: same linear
    # multinomial model, but LBFGS on ~96K x 1024 x 34 classes takes
    # minutes PER LAYER, and 13 layers of that is an hour of waiting to
    # answer a ranking question. SGD converges to a comparable solution
    # in seconds. Since only the RELATIVE ordering is being read (see
    # module docstring), the small optimisation gap is irrelevant --
    # every layer is fit by the identical procedure.
    #
    # Note: sklearn >=1.8 removed LogisticRegression's `multi_class`
    # argument (multinomial is now the default), which is what the
    # earlier version tripped over.
    clf = SGDClassifier(loss="log_loss", max_iter=iters, tol=1e-4,
                        n_jobs=-1, random_state=seed, early_stopping=False)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xva, yva)), float(clf.score(Xtr, ytr))


def cmd_run(a):
    E = _load("extract")
    cache = E.FeatureCache(a.cache)
    per_key, names, skipped = build_frame_labels(
        a.manifest, a.text, mode=a.mode, trim=a.trim)

    n_frames = sum(len(v[0]) for v in per_key.values())
    print(f"mode          : {a.mode}")
    print(f"items labelled: {len(per_key)}  (cache has {len(cache)})")
    print(f"classes       : {len(names)}")
    print(f"labelled frames: {n_frames}")
    if skipped:
        print(f"skipped       : {dict(skipped)}")

    overlap = sum(1 for k in per_key if k in cache.index)
    print(f"in cache      : {overlap}")
    if overlap == 0:
        raise SystemExit(
            "No labelled item is present in the feature cache. The cache "
            "key convention in extract.py and build_frame_labels have "
            "diverged -- compare a key from each before going further.")

    # Majority-class baseline. Any layer that fails to beat this is
    # carrying no usable phonetic information, and if EVERY layer fails to
    # beat it the labels are wrong, not the model.
    _, y0 = gather(cache, per_key, cache.layers[0], a.max_frames)
    counts = np.bincount(y0, minlength=len(names))
    majority = counts.max() / counts.sum()
    print(f"majority class: {majority:.4f} "
          f"({names[int(counts.argmax())]!r})\n")

    print(f"{'layer':>6} {'val acc':>9} {'train acc':>10}   {'':<28}")
    results = {}
    best = (None, -1.0)
    for L in cache.layers:
        X, y = gather(cache, per_key, L, a.max_frames)
        va, tr = fit_probe(X, y, len(names), iters=a.iters)
        results[L] = {"val": va, "train": tr}
        bar = "#" * int(max(0.0, va - majority) * 200)
        print(f"{L:>6} {va:9.4f} {tr:10.4f}   {bar}")
        if va > best[1]:
            best = (L, va)

    print(f"\npeak: layer {best[0]} at {best[1]:.4f} "
          f"(majority {majority:.4f})")

    vals = [results[L]["val"] for L in cache.layers]
    spread = max(vals) - min(vals)
    print(f"spread across layers: {spread:.4f}")
    if spread < 0.02:
        print("\n*** FLAT CURVE -- DO NOT PROCEED ON THIS RESULT ***")
        print("Layer choice should matter, and here it does not. That")
        print("points upstream at the labels, not at the encoder: check")
        print("phones.py output and the word-span-to-frame mapping in")
        print("build_frame_labels before reading anything into the peak.")

    # Suggest a band around the peak, which is what actually gets cached
    # for training -- the head uses a learned weighted sum over a band,
    # not a single layer.
    order = sorted(cache.layers, key=lambda L: -results[L]["val"])
    band = sorted(order[:a.band])
    print(f"\nsuggested band (top {a.band}): {band}")
    print(f"  extract.py full --layers {','.join(str(x) for x in band)}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({
            "mode": a.mode, "majority": majority, "results": results,
            "peak_layer": best[0], "band": band, "n_classes": len(names),
            "frames_used": int(min(a.max_frames, n_frames)),
        }, indent=2))
        print(f"\nwrote {a.out}")
    return 0


def cmd_labels(a):
    """Print a few labelled items, to eyeball the frame mapping before
    trusting any probe number."""
    P = _load("phones")
    per_key, names, skipped = build_frame_labels(
        a.manifest, a.text, mode=a.mode, trim=a.trim)
    print(f"classes: {len(names)}   skipped: {dict(skipped)}\n")
    for k in list(per_key)[:a.n]:
        idx, lab = per_key[k]
        runs, cur, start = [], lab[0], 0
        for i in range(1, len(lab)):
            if lab[i] != cur:
                runs.append((names[cur], idx[start], idx[i - 1]))
                cur, start = lab[i], i
        runs.append((names[cur], idx[start], idx[-1]))
        print(f"{k}  {len(idx)} frames, {len(runs)} runs")
        for nm, a0, a1 in runs[:14]:
            print(f"    {nm:<6} frames {a0:4d}-{a1:4d} "
                  f"({(a1-a0+1)*20:4d} ms)")
        print()
    return 0


def main():
    ap = argparse.ArgumentParser(description="XLS-R layer probe")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cache", default="cache/probe")
        p.add_argument("--manifest", default="manifest/manifest_clean.csv")
        p.add_argument("--text", default="texts/quran-simple-plain.txt")
        p.add_argument("--mode", choices=["phone", "word"], default="phone")
        p.add_argument("--trim", type=int, default=1,
                       help="frames dropped at each unit edge")

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--max-frames", type=int, default=120000)
    r.add_argument("--iters", type=int, default=30)
    r.add_argument("--band", type=int, default=6)
    r.add_argument("--out", default="manifest/probe_results.json")
    r.set_defaults(func=cmd_run)

    l = sub.add_parser("labels")
    common(l)
    l.add_argument("--n", type=int, default=3)
    l.set_defaults(func=cmd_labels)

    a = ap.parse_args()
    rc = a.func(a) or 0
    sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
