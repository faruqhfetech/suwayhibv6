#!/usr/bin/env python3
"""
Suwayhib -- Step 3: synthetic error injection with artifact controls.

Builds a labelled test set from verified reference recitations.

    python injector.py build --manifest manifest_full.csv --lib reference_library \
        --out testset --n 2000 --seed 1

WHY THE CONTROLS MATTER
-----------------------
Naive splicing leaves a seam: a click, a phase discontinuity, an amplitude
step. A model can score well by detecting EDITS rather than mispronunciation,
and that skill transfers to exactly zero real learners.

Three defences, all on by default:
  1. equal-power crossfades at every join
  2. RMS matching of inserted material to what it replaced
  3. a CONTROL class -- audio cut and rejoined AT THE SAME POINT, no error.
     Labelled negative. If your model flags these, it is reading seams, not
     recitation. Report control false-positive rate ALONGSIDE your headline
     number; a system that fails here has not learned the task.

ERROR TIERS
-----------
  structural : word_deletion, word_repetition, truncation
               -> what beginners actually do most. Should be near-solved.
  prosodic   : madd_shortening, madd_lengthening
               -> duration errors. Detectable from alignment ratios.
  segmental  : word_substitution, near_miss_substitution
               -> the hard tier. Expect weak numbers; report separately.

Report metrics PER TIER. An aggregate F1 over a mixture you chose is a
number about your sampling, not about your system.
"""

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SR = 16000                 # everything resampled here; plenty for speech
XFADE_MS = 20              # equal-power crossfade at every join
CONTROL_FRACTION = 0.20    # share of clean items that get seam-only controls


# --------------------------------------------------------------------------
# audio io
# --------------------------------------------------------------------------

def load_audio(path, sr=SR):
    """Decode any format to mono float32 at sr via ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"decode failed {path}: {r.stderr.decode()[:200]}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def save_audio(path, x, sr=SR):
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr),
           "-ac", "1", "-i", "-", str(path)]
    subprocess.run(cmd, input=x.tobytes(), capture_output=True, check=True)


def ms2s(ms):
    return int(ms * SR / 1000)


# --------------------------------------------------------------------------
# seam-safe splicing  -- the core of the artifact defence
# --------------------------------------------------------------------------

def crossfade_concat(a, b, xfade=ms2s(XFADE_MS)):
    """Equal-power crossfade join. Preserves perceived loudness through the
    seam, unlike a linear fade which dips."""
    n = min(xfade, len(a), len(b))
    if n <= 0:
        return np.concatenate([a, b])
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    fade_out, fade_in = np.cos(t), np.sin(t)
    mid = a[-n:] * fade_out + b[:n] * fade_in
    return np.concatenate([a[:-n], mid, b[n:]])


def splice(segments):
    out = segments[0]
    for s in segments[1:]:
        out = crossfade_concat(out, s)
    return out


def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def match_rms(x, target_rms, max_gain=4.0):
    """Scale x toward a target level so inserted material doesn't stand out
    on loudness alone -- another cue a model could cheat on."""
    cur = rms(x)
    if cur <= 1e-8 or target_rms <= 1e-8:
        return x
    g = float(np.clip(target_rms / cur, 1.0 / max_gain, max_gain))
    return x * g


def time_stretch(x, rate):
    """Pitch-preserving stretch. rate<1 = longer. librosa if present, else
    ffmpeg atempo (which chains for extreme factors)."""
    if len(x) < 512:
        return x
    try:
        import librosa
        return librosa.effects.time_stretch(y=x.astype(np.float32), rate=rate)
    except Exception:
        pass
    tempo = rate
    filters = []
    while tempo < 0.5:
        filters.append("atempo=0.5"); tempo /= 0.5
    while tempo > 2.0:
        filters.append("atempo=2.0"); tempo /= 2.0
    filters.append(f"atempo={tempo:.6f}")
    cmd = ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
           "-i", "-", "-af", ",".join(filters), "-f", "f32le", "-"]
    r = subprocess.run(cmd, input=x.tobytes(), capture_output=True)
    if r.returncode != 0:
        return x
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

def load_manifest(path):
    """-> {(reciter, surah, ayah): [ {word_idx,start_ms,end_ms}, ... ] sorted}"""
    ayahs = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["multiword_segment"]):
                continue                       # cannot isolate these words
            ayahs[(r["reciter"], int(r["surah"]), int(r["ayah"]))].append({
                "word_idx": int(r["word_idx"]),
                "start_ms": int(r["start_ms"]),
                "end_ms": int(r["end_ms"]),
                "audio_path": r["audio_path"],
            })
    for k in ayahs:
        ayahs[k].sort(key=lambda w: w["word_idx"])
    return dict(ayahs)


def load_suspects(path, top_n=50):
    """(surah, ayah, word_idx) pairs too dubious to use as ground truth."""
    bad = set()
    p = Path(path)
    if not p.exists():
        return bad
    with open(p, encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if i >= top_n:
                break
            bad.add((int(r["surah"]), int(r["ayah"]), int(r["word_idx"])))
    return bad


def word_slices(audio, words):
    """[(start_sample, end_sample)] per word, plus leading/trailing context."""
    return [(ms2s(w["start_ms"]), ms2s(w["end_ms"])) for w in words]


# --------------------------------------------------------------------------
# error generators
# each returns (audio, label_dict) or None if not applicable
# --------------------------------------------------------------------------

def _parts(audio, sl, k):
    """audio before word k, word k itself, audio after word k."""
    return audio[:sl[k][0]], audio[sl[k][0]:sl[k][1]], audio[sl[k][1]:]


def err_control(audio, sl, k, rng, pool):
    """NEGATIVE. Cut and rejoin at the same points. Seam present, no error."""
    pre, mid, post = _parts(audio, sl, k)
    return splice([pre, mid, post]), {"error_type": "control_seam",
                                      "is_error": 0, "word_idx": k,
                                      "tier": "control"}


def err_clean(audio, sl, k, rng, pool):
    """NEGATIVE. Untouched audio."""
    return audio, {"error_type": "clean", "is_error": 0,
                   "word_idx": -1, "tier": "control"}


def err_word_deletion(audio, sl, k, rng, pool):
    pre, mid, post = _parts(audio, sl, k)
    if len(mid) < ms2s(80):
        return None
    return splice([pre, post]), {"error_type": "word_deletion", "is_error": 1,
                                 "word_idx": k, "tier": "structural"}


def err_word_repetition(audio, sl, k, rng, pool):
    pre, mid, post = _parts(audio, sl, k)
    if len(mid) < ms2s(80):
        return None
    return splice([pre, mid, mid, post]), {"error_type": "word_repetition",
                                           "is_error": 1, "word_idx": k,
                                           "tier": "structural"}


def err_truncation(audio, sl, k, rng, pool):
    """Learner stops mid-ayah. Only meaningful past the first word."""
    if k < 1 or k >= len(sl) - 1:
        return None
    return audio[:sl[k][1]], {"error_type": "truncation", "is_error": 1,
                              "word_idx": k + 1, "tier": "structural"}


def err_madd_shortening(audio, sl, k, rng, pool):
    pre, mid, post = _parts(audio, sl, k)
    if len(mid) < ms2s(250):
        return None                        # need a long word to shorten
    rate = rng.uniform(1.8, 2.6)           # >1 = shorter
    return splice([pre, match_rms(time_stretch(mid, rate), rms(mid)), post]), \
        {"error_type": "madd_shortening", "is_error": 1, "word_idx": k,
         "tier": "prosodic", "stretch_rate": round(rate, 3)}


def err_madd_lengthening(audio, sl, k, rng, pool):
    pre, mid, post = _parts(audio, sl, k)
    if len(mid) < ms2s(120):
        return None
    rate = rng.uniform(0.40, 0.60)
    return splice([pre, match_rms(time_stretch(mid, rate), rms(mid)), post]), \
        {"error_type": "madd_lengthening", "is_error": 1, "word_idx": k,
         "tier": "prosodic", "stretch_rate": round(rate, 3)}


def err_word_substitution(audio, sl, k, rng, pool):
    """Swap in a different word FROM THE SAME RECITER -- same voice, same
    recording chain, so the model cannot win on speaker or channel cues."""
    pre, mid, post = _parts(audio, sl, k)
    cand = pool.get("same_reciter", [])
    if not cand or len(mid) < ms2s(80):
        return None
    donor = rng.choice(cand)
    if len(donor) < ms2s(80):
        return None
    donor = match_rms(donor, rms(mid))
    return splice([pre, donor, post]), {"error_type": "word_substitution",
                                        "is_error": 1, "word_idx": k,
                                        "tier": "segmental"}


def err_near_miss(audio, sl, k, rng, pool):
    """Hardest tier: donor of SIMILAR DURATION from the same reciter, so
    duration cues are removed and only phonetic content distinguishes it."""
    pre, mid, post = _parts(audio, sl, k)
    cand = pool.get("same_reciter", [])
    if not cand or len(mid) < ms2s(80):
        return None
    target = len(mid)
    close = [c for c in cand if abs(len(c) - target) < 0.15 * target
             and len(c) >= ms2s(80)]
    if not close:
        return None
    donor = match_rms(rng.choice(close), rms(mid))
    return splice([pre, donor, post]), {"error_type": "near_miss_substitution",
                                        "is_error": 1, "word_idx": k,
                                        "tier": "segmental"}


GENERATORS = {
    "clean": err_clean,
    "control_seam": err_control,
    "word_deletion": err_word_deletion,
    "word_repetition": err_word_repetition,
    "truncation": err_truncation,
    "madd_shortening": err_madd_shortening,
    "madd_lengthening": err_madd_lengthening,
    "word_substitution": err_word_substitution,
    "near_miss_substitution": err_near_miss,
}

DEFAULT_MIX = {
    "clean": 0.20,
    "control_seam": 0.10,
    "word_deletion": 0.12,
    "word_repetition": 0.08,
    "truncation": 0.08,
    "madd_shortening": 0.12,
    "madd_lengthening": 0.10,
    "word_substitution": 0.10,
    "near_miss_substitution": 0.10,
}


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def cmd_build(args):
    rng = random.Random(args.seed)
    lib = Path(args.lib)
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    ayahs = load_manifest(args.manifest)
    suspects = load_suspects(
        Path(args.manifest).with_name(Path(args.manifest).stem + "_suspects.csv"),
        args.suspect_top_n)
    print(f"ayah-reciter pairs: {len(ayahs)}")
    print(f"suspect words excluded from ground truth: {len(suspects)}")

    # held-out reciters -- test generalization to an unseen voice
    reciters = sorted({k[0] for k in ayahs})
    holdout = set(args.holdout_reciter or [])
    if holdout:
        print(f"holdout reciters: {sorted(holdout)}")

    keys = [k for k in ayahs if len(ayahs[k]) >= 3]
    rng.shuffle(keys)

    mix = DEFAULT_MIX.copy()
    total = sum(mix.values())
    mix = {k: v / total for k, v in mix.items()}

    # donor pool of word audio, per reciter, built lazily
    donor_pool = defaultdict(list)
    audio_cache = {}

    def get_audio(key):
        if key not in audio_cache:
            if len(audio_cache) > 60:
                audio_cache.clear()
            audio_cache[key] = load_audio(lib / ayahs[key][0]["audio_path"])
        return audio_cache[key]

    # pre-build donor words from a sample of ayahs per reciter
    print("building donor pool...")
    per_reciter = defaultdict(list)
    for k in keys:
        per_reciter[k[0]].append(k)
    for rec, ks in per_reciter.items():
        for key in ks[:args.donor_ayahs]:
            try:
                a = get_audio(key)
            except Exception:
                continue
            for (s, e) in word_slices(a, ayahs[key]):
                if e > s and e <= len(a):
                    donor_pool[rec].append(a[s:e].copy())
    print("  " + ", ".join(f"{r}:{len(v)}" for r, v in sorted(donor_pool.items())))

    rows = []
    counts = defaultdict(int)
    attempts = 0
    targets = {t: int(round(args.n * f)) for t, f in mix.items()}

    while sum(counts.values()) < args.n and attempts < args.n * 25:
        attempts += 1
        remaining = [t for t in targets if counts[t] < targets[t]]
        if not remaining:
            break
        etype = rng.choice(remaining)
        key = rng.choice(keys)
        reciter, surah, ayah = key
        words = ayahs[key]

        try:
            audio = get_audio(key)
        except Exception:
            continue
        sl = word_slices(audio, words)
        if not sl or sl[-1][1] > len(audio) or any(e <= s for s, e in sl):
            continue

        # pick a target word, avoiding dubious ground truth and word 0
        choices = [i for i in range(len(words))
                   if (surah, ayah, words[i]["word_idx"]) not in suspects
                   and (i > 0 or args.allow_first_word)]
        if not choices:
            continue
        k = rng.choice(choices)

        pool = {"same_reciter": donor_pool.get(reciter, [])}
        res = GENERATORS[etype](audio, sl, k, rng, pool)
        if res is None:
            continue
        new_audio, label = res
        if len(new_audio) < ms2s(300):
            continue

        idx = sum(counts.values())
        fname = f"{idx:06d}_{surah:03d}{ayah:03d}_{etype}.wav"
        save_audio(out / "audio" / fname, new_audio)

        label.update({
            "id": idx, "file": f"audio/{fname}",
            "reciter": reciter, "surah": surah, "ayah": ayah,
            "n_words": len(words),
            "split": "holdout" if reciter in holdout else "dev",
            "orig_duration_ms": int(len(audio) / SR * 1000),
            "new_duration_ms": int(len(new_audio) / SR * 1000),
        })
        rows.append(label)
        counts[etype] += 1

        if sum(counts.values()) % 100 == 0:
            print(f"  {sum(counts.values())}/{args.n}")

    # write labels
    with open(out / "labels.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "seed": args.seed, "n": len(rows), "sr": SR,
        "crossfade_ms": XFADE_MS, "rms_matched": True,
        "manifest": str(args.manifest),
        "suspects_excluded": len(suspects),
        "holdout_reciters": sorted(holdout),
        "counts": dict(counts),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"wrote {len(rows)} items to {out}")
    tiers = defaultdict(int)
    for r in rows:
        tiers[r["tier"]] += 1
    for t, n in sorted(tiers.items()):
        print(f"  {t:12s} {n:5d}")
    print("\nper type:")
    for t, n in sorted(counts.items()):
        print(f"  {t:26s} {n:5d}")
    pos = sum(1 for r in rows if r["is_error"])
    print(f"\npositive rate: {pos/max(len(rows),1):.3f}  "
          f"(use this as the AUPRC baseline, not 0.5)")
    print("\nWhen you evaluate: report control_seam false-positive rate FIRST.")
    print("If it is not near the clean false-positive rate, the model is")
    print("reading splice artifacts and every other number is inflated.")


def main():
    ap = argparse.ArgumentParser(description="Suwayhib synthetic error injector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--manifest", default="manifest_full.csv")
    b.add_argument("--lib", default="reference_library")
    b.add_argument("--out", default="testset")
    b.add_argument("--n", type=int, default=2000)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--donor-ayahs", type=int, default=40,
                   help="ayahs per reciter mined for substitution donors")
    b.add_argument("--suspect-top-n", type=int, default=50,
                   help="top N rows of *_suspects.csv excluded as ground truth")
    b.add_argument("--holdout-reciter", nargs="*",
                   help="reciters marked split=holdout (unseen-voice test)")
    b.add_argument("--allow-first-word", action="store_true",
                   help="word 0 absorbs leading silence; excluded by default")
    b.set_defaults(func=cmd_build)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
