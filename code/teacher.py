#!/usr/bin/env python3
"""
Suwayhib v5 -- teacher target extraction.

    python teacher.py pilot   --n 60  --device cuda
    python teacher.py extract --n 4000 --device cuda --out targets.jsonl
    python teacher.py inspect --targets targets.jsonl

WHAT THIS DOES
--------------
Generates the supervision for distillation. For each ayah it builds several
augmented and/or deliberately corrupted variants, runs the v4 teacher
(whisper-base-quran, forced decoding) over each, and records the per-word
probability the teacher assigns.

The student will later be trained to reproduce those numbers from a fraction
of the compute.

WHY CORRUPTION IS NOT OPTIONAL
------------------------------
Clean audio gives the teacher ~1.000 almost everywhere -- measured, a
professional reciter scored median 1.0000 across 160 words. A student trained
on that learns to output 1.0 always: perfect training loss, never flags
anything. Deliberate errors are what create a RANGE of targets and therefore
a gradient.

The right corrupted FRACTION is an open question. Run `pilot`, look at the
target distribution, then set it. Do not guess it up front.

WHY AUGMENTATION HAPPENS AT THE WAVEFORM
----------------------------------------
The teacher must SEE the augmented audio, so its targets reflect it. v4a
cached features and thereby made waveform augmentation impossible, which
removed the only defence against a six-speaker training set. Not repeating
that.

WHY RECIPES AND NOT WAVEFORMS
-----------------------------
Each variant is stored as a RECIPE (source + seed + operations), not as
audio. render() regenerates it deterministically. Extraction and training
call the SAME function, so the audio the student sees is bit-identical to
the audio the teacher scored. Storing the waveforms would also be hundreds
of gigabytes.
"""

import argparse
import csv
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

SR = 16000
XFADE = int(0.02 * SR)      # 20ms equal-power crossfade at every splice


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def load_audio(path):
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"decode failed: {path}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def crossfade(a, b, n=XFADE):
    """Equal-power join. Preserves perceived loudness through the seam,
    unlike a linear fade which dips. Without this the student could learn to
    detect SPLICE SEAMS instead of mispronunciation."""
    n = min(n, len(a), len(b))
    if n <= 0:
        return np.concatenate([a, b])
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    mid = a[-n:] * np.cos(t) + b[:n] * np.sin(t)
    return np.concatenate([a[:-n], mid, b[n:]])


def splice(parts):
    out = parts[0]
    for p in parts[1:]:
        out = crossfade(out, p)
    return out


def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def match_rms(x, target, max_gain=4.0):
    cur = rms(x)
    if cur <= 1e-8 or target <= 1e-8:
        return x
    g = float(np.clip(target / cur, 1 / max_gain, max_gain))
    return x * g


def resample(x, rate):
    """rate>1 = faster/shorter. Linear interpolation: cheap, and it shifts
    formants along with duration, which is a crude but real speaker
    perturbation (VTLP-ish)."""
    if abs(rate - 1.0) < 1e-3 or len(x) < 8:
        return x
    n = max(8, int(len(x) / rate))
    return np.interp(np.linspace(0, len(x) - 1, n),
                     np.arange(len(x)), x).astype(np.float32)


def time_stretch(x, rate):
    """Pitch-preserving where available, so a madd error is a DURATION change
    and not a pitch change."""
    if len(x) < 512:
        return x
    try:
        import librosa
        return librosa.effects.time_stretch(y=x.astype(np.float32), rate=rate)
    except Exception:
        return resample(x, rate)


def add_noise(x, snr_db, seed):
    p = float(np.mean(x ** 2)) + 1e-10
    sigma = math.sqrt(p / (10 ** (snr_db / 10)))
    rng = np.random.default_rng(seed)
    return (x + rng.normal(0, sigma, len(x))).astype(np.float32)


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

def load_manifest(path):
    ay = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["multiword_segment"]):
                continue
            ay[(r["reciter"], int(r["surah"]), int(r["ayah"]))].append({
                "word_idx": int(r["word_idx"]),
                "start_ms": int(r["start_ms"]),
                "end_ms": int(r["end_ms"]),
                "audio_path": r["audio_path"],
            })
    for k in ay:
        ay[k].sort(key=lambda w: w["word_idx"])
    return dict(ay)


def ms(v):
    return int(v * SR / 1000)


# --------------------------------------------------------------------------
# the recipe: a deterministic description of one training variant
# --------------------------------------------------------------------------

AUGS = ["speed", "gain", "noise", "vtlp"]

CORRUPTIONS = [
    "none",
    "control_seam",     # NEGATIVE CONTROL: spliced, but not an error
    "same_ayah",        # swap in another word from THIS ayah -- the hardest
    "duration_match",   # same reciter, similar duration: no duration cue
    "random_word",      # any other word: the easy negative
    "madd_short",       # squeeze a long word
    "madd_long",        # stretch a word
    "deletion",         # drop a word entirely
]


def make_recipe(rng, key, words, corrupt_p, aug_p):
    """A recipe is everything needed to regenerate one variant."""
    n = len(words)
    rec = {"reciter": key[0], "surah": key[1], "ayah": key[2],
           "seed": rng.randrange(1 << 30), "augs": {}, "corruption": "none",
           "target_word": -1}

    if rng.random() < aug_p:
        r = random.Random(rec["seed"])
        if r.random() < 0.7:
            rec["augs"]["speed"] = round(r.uniform(0.85, 1.18), 4)
        if r.random() < 0.5:
            rec["augs"]["gain"] = round(r.uniform(0.6, 1.6), 4)
        if r.random() < 0.4:
            rec["augs"]["noise_snr"] = round(r.uniform(10, 30), 2)
        if r.random() < 0.4:
            rec["augs"]["vtlp"] = round(r.uniform(0.92, 1.09), 4)

    if n >= 2 and rng.random() < corrupt_p:
        # weight toward the HARD negatives: random wrong words are rejected
        # trivially and stop teaching within an epoch
        rec["corruption"] = rng.choices(
            CORRUPTIONS[1:],
            weights=[0.12, 0.26, 0.22, 0.09, 0.12, 0.10, 0.09])[0]
        rec["target_word"] = rng.randrange(n)
    return rec


def render(recipe, ayahs, lib, donor_pool=None):
    """Regenerate the exact audio for a recipe. Called at extraction time AND
    at training time -- one function, so the two cannot drift apart.

    Returns (audio, corrupted_word_idx or -1, spans)

    `spans` is [(start_sample, end_sample)] per word IN THE RENDERED AUDIO.
    This matters: corruption changes the length of the affected word, so
    EVERY WORD AFTER IT SHIFTS. Recomputing spans from the original manifest
    timings would put the window on the wrong audio for those words while
    still carrying a ~1.0 target -- label noise, which is exactly what stops
    a model from fitting even a tiny training set.
    """
    key = (recipe["reciter"], recipe["surah"], recipe["ayah"])
    words = ayahs[key]
    audio = load_audio(Path(lib) / words[0]["audio_path"])
    r = random.Random(recipe["seed"])

    spans = [[ms(w["start_ms"]), ms(w["end_ms"])] for w in words]
    spans = [[max(0, a), min(len(audio), b)] for a, b in spans]

    k = recipe["target_word"]
    corr = recipe["corruption"]
    if corr != "none" and 0 <= k < len(words):
        s, e = spans[k]
        pre, mid, post = audio[:s], audio[s:e], audio[e:]
        old_w = len(mid)

        if corr == "deletion":
            new_mid = np.zeros(0, dtype=np.float32)
            parts = [pre, post]
        elif corr in ("madd_short", "madd_long"):
            rate = r.uniform(1.8, 2.6) if corr == "madd_short" else r.uniform(0.4, 0.6)
            new_mid = match_rms(time_stretch(mid, rate), rms(mid))
            parts = [pre, new_mid, post]
        elif corr == "control_seam":
            # cut and rejoin AT THE SAME POINT: a splice seam, no error.
            # If the model flags these it is reading EDIT ARTIFACTS rather
            # than verifying text against audio, and every other number is
            # meaningless. Same control that validated v3.
            new_mid = mid
            parts = [pre, mid, post]
        else:
            donor = _pick_donor(r, corr, recipe, words, k, audio, spans, donor_pool)
            if donor is None:
                a2 = _apply_augs(audio, recipe, r)
                return a2, -1, _scale_spans(spans, len(audio), len(a2))
            new_mid = match_rms(donor, rms(mid))
            parts = [pre, new_mid, post]

        audio = splice(parts)

        # a crossfade join overlaps XFADE samples, shortening the result
        n_joins = len(parts) - 1
        fade = min(XFADE, old_w if old_w else XFADE)
        shift = (len(new_mid) - old_w) - n_joins * fade
        spans[k] = [s, s + len(new_mid)]
        for j in range(k + 1, len(spans)):
            spans[j] = [spans[j][0] + shift, spans[j][1] + shift]
        if corr == "deletion":
            spans[k] = [s, s]          # zero width: the word is not there

    out = _apply_augs(audio, recipe, r)
    return out, (k if corr != "none" else -1), _scale_spans(spans, len(audio), len(out))


def _scale_spans(spans, old_len, new_len):
    """Augmentation (speed) rescales the whole waveform uniformly, so the
    span map is an exact linear rescale -- not an approximation."""
    if old_len <= 0 or new_len == old_len:
        return [(int(a), int(b)) for a, b in spans]
    f = new_len / old_len
    return [(max(0, int(a * f)), min(new_len, int(b * f))) for a, b in spans]


def _pick_donor(r, corr, recipe, words, k, audio, spans, donor_pool):
    """Donors come from the SAME reciter, so a substitution can never be
    detected by noticing the voice changed."""
    target_len = spans[k][1] - spans[k][0]

    if corr == "same_ayah":
        others = [i for i in range(len(words)) if i != k]
        if not others:
            return None
        j = r.choice(others)
        a, b = spans[j]
        return audio[max(0, a):min(len(audio), b)].copy()

    pool = (donor_pool or {}).get(recipe["reciter"], [])
    if not pool:
        return None
    if corr == "duration_match":
        close = [d for d in pool
                 if abs(len(d) - target_len) < 0.15 * target_len
                 and len(d) > ms(80)]
        return r.choice(close).copy() if close else None
    return r.choice(pool).copy()


def _apply_augs(x, recipe, r):
    a = recipe.get("augs", {})
    if "vtlp" in a:
        # resample then resample back: length restored, formants shifted
        n0 = len(x)
        x = resample(x, a["vtlp"])
        x = np.interp(np.linspace(0, len(x) - 1, n0),
                      np.arange(len(x)), x).astype(np.float32)
    if "speed" in a:
        x = time_stretch(x, a["speed"])
    if "gain" in a:
        x = x * a["gain"]
    if "noise_snr" in a:
        x = add_noise(x, a["noise_snr"], recipe["seed"])
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def build_donor_pool(ayahs, lib, keys, per_reciter=250, seed=0):
    rng = random.Random(seed)
    pool = defaultdict(list)
    by_rec = defaultdict(list)
    for k in keys:
        by_rec[k[0]].append(k)
    for rec, ks in by_rec.items():
        rng.shuffle(ks)
        for key in ks:
            if len(pool[rec]) >= per_reciter:
                break
            try:
                a = load_audio(Path(lib) / ayahs[key][0]["audio_path"])
            except Exception:
                continue
            for w in ayahs[key]:
                s, e = ms(w["start_ms"]), ms(w["end_ms"])
                if 0 <= s < e <= len(a) and e - s > ms(80):
                    pool[rec].append(a[s:e].copy())
    return dict(pool)


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def run(args, n_items, out_path):
    import importlib.util
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("textscore", here / "textscore.py")
    TS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TS)

    teacher = TS.TextScorer(args.model, args.text, args.device)
    ayahs = load_manifest(args.manifest)

    # hold out the same two reciters as v3/v4, so every version stays
    # comparable on the same unseen voices
    HOLD = {"Husary_64kbps", "Minshawy_Murattal_128kbps"}
    keys = [k for k in ayahs
            if (k[0] in HOLD) == args.holdout and len(ayahs[k]) >= 2]
    rng = random.Random(args.seed)
    rng.shuffle(keys)

    print(f"ayah-reciter pairs available: {len(keys)} "
          f"({'HELD-OUT' if args.holdout else 'seen'} reciters)")
    print("building donor pool...")
    pool = build_donor_pool(ayahs, args.lib, keys, args.donors, args.seed)
    print("  " + ", ".join(f"{r}:{len(v)}" for r, v in sorted(pool.items())))

    written = 0
    attempts = 0
    failures = defaultdict(int)
    max_attempts = n_items * 20

    with open(out_path, "w", encoding="utf-8") as fh:
        while written < n_items and attempts < max_attempts:
            attempts += 1
            key = rng.choice(keys)
            recipe = make_recipe(rng, key, ayahs[key], args.corrupt_p, args.aug_p)
            try:
                audio, corrupted_idx, spans = render(
                    recipe, ayahs, args.lib, pool)
                if len(audio) < ms(400):
                    failures["audio too short"] += 1
                    continue
                scored = teacher.score(audio, key[1], key[2])
            except Exception as e:
                # NEVER swallow silently: a bare `except: continue` inside an
                # uncapped while-loop turned a one-line signature mismatch
                # into a 30-minute stall with zero output.
                failures[f"{type(e).__name__}: {str(e)[:60]}"] += 1
                continue
            if not scored:
                failures["not in text"] += 1
                continue

            fh.write(json.dumps({
                "recipe": recipe,
                "corrupted_word": corrupted_idx,
                "text": scored["text"],
                "words": [{"idx": w["idx"], "word": w["word"],
                           "target": round(w["worst"], 5),
                           "mean": round(w["mean"], 5),
                           "weak_token": w["weak_token"]}
                          for w in scored["words"]],
            }, ensure_ascii=False) + "\n")
            written += 1
            if written % 25 == 0:
                print(f"  {written}/{n_items}")

    if failures:
        print("\nfailures encountered:")
        for k, v in sorted(failures.items(), key=lambda kv: -kv[1]):
            print(f"  {v:6d}  {k}")
    if written < n_items:
        print(f"\nSTOPPED at {written}/{n_items} after {attempts} attempts. "
              f"See failures above.")
    print(f"wrote {out_path}  ({written} variants)")


def cmd_pilot(args):
    run(args, args.n, args.out)
    print("\nNow: python teacher.py inspect --targets " + args.out)
    print("Look at the target distribution BEFORE scaling up. The corrupted")
    print("fraction should be set by what the teacher actually produces.")


def cmd_extract(args):
    run(args, args.n, args.out)


# --------------------------------------------------------------------------

def cmd_inspect(args):
    rows = [json.loads(l) for l in open(args.targets, encoding="utf-8")]
    clean_t, corr_t = [], []
    by_corruption = defaultdict(list)
    for r in rows:
        ci = r["corrupted_word"]
        for w in r["words"]:
            if ci >= 0 and w["idx"] == ci:
                corr_t.append(w["target"])
                by_corruption[r["recipe"]["corruption"]].append(w["target"])
            else:
                clean_t.append(w["target"])

    def stats(v, name):
        if not v:
            return
        v = np.array(v)
        print(f"  {name:22s} n={len(v):6d}  median {np.median(v):.4f}  "
              f"p10 {np.quantile(v,0.10):.4f}  p90 {np.quantile(v,0.90):.4f}  "
              f"<0.1 {np.mean(v<0.1):.1%}")

    print(f"variants: {len(rows)}")
    print(f"words:    {len(clean_t)+len(corr_t)}\n")
    print("TEACHER TARGETS")
    stats(clean_t, "uncorrupted words")
    stats(corr_t, "corrupted words")
    print("\nby corruption type")
    for k in sorted(by_corruption):
        stats(by_corruption[k], k)

    if clean_t and corr_t:
        c, e = np.array(clean_t), np.array(corr_t)
        print(f"\nseparation: {np.mean(c > 0.5):.1%} of uncorrupted words "
              f"above 0.5, {np.mean(e < 0.1):.1%} of corrupted below 0.1")
        print(f"positive rate in this sample: {len(corr_t)/(len(c)+len(e)):.1%}")
        print("\nWHAT TO CHECK")
        print("  - are uncorrupted words clustered near 1.0? If not, the")
        print("    augmentation is too aggressive and is destroying signal.")
        print("  - do corrupted words actually drop? A corruption type whose")
        print("    median stays high is not teaching anything.")
        print("  - is the overall distribution BIMODAL? A student needs both")
        print("    ends. If almost everything sits near 1.0, raise --corrupt-p.")


def main():
    ap = argparse.ArgumentParser(description="v5 teacher target extraction")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn, dn in (("pilot", cmd_pilot, 60), ("extract", cmd_extract, 4000)):
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=dn)
        p.add_argument("--out", default=f"targets_{name}.jsonl")
        p.add_argument("--manifest", default="manifest_full.csv")
        p.add_argument("--lib", default="reference_library")
        p.add_argument("--model", default="whisper-base-quran")
        p.add_argument("--text", default="texts/quran-simple-plain.txt")
        p.add_argument("--device", default="cpu")
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--corrupt-p", type=float, default=0.5,
                       help="fraction of variants carrying a deliberate error")
        p.add_argument("--aug-p", type=float, default=0.8)
        p.add_argument("--donors", type=int, default=250)
        p.add_argument("--holdout", action="store_true",
                       help="use the HELD-OUT reciters instead of the seen ones")
        p.set_defaults(func=fn)

    i = sub.add_parser("inspect")
    i.add_argument("--targets", default="targets_pilot.jsonl")
    i.set_defaults(func=cmd_inspect)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
