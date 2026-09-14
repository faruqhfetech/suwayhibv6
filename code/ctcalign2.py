#!/usr/bin/env python3
"""
Suwayhib v5 -- streaming word verifier, rewrite 3: LIKELIHOOD RATIO.

    python ctcalign2.py check --spec recs/descriptions.txt --dir recs
    python ctcalign2.py scan  --n 60
    python ctcalign2.py walk  --audio recs/Recording_6.m4a --ref 78 9 --verbose

WHY REWRITE 3
=============
Rewrites 1 and 2 both scored an alignment in ABSOLUTE terms -- "how well do
these characters explain this audio". Both failed, for the same underlying
reason.

  rewrite 1 (mean log-prob)    confirmed EVERYTHING at every threshold from
                               -1 to -5. The score rose monotonically with
                               window length (0.53, 0.75, 0.79, 0.86, 0.894)
                               and never turned over, so argmax simply
                               returned the longest candidate.
  rewrite 2 (worst character)  ordering was right -- correct audio reached
                               -9.4 where the س->ص error reached -13.1 --
                               but calibration showed correct words on real
                               audio span -2.4 to -14.2, so no threshold
                               separates them. And Recording_4, a COMPLETELY
                               WRONG VERSE, scored -8.9/-7.7/-8.4: BETTER
                               than most correct words.

That last observation is the diagnostic one. CTC always finds *a* best path.
Without something to compare against, "the best path through this lattice"
says nothing about whether the word is actually there.

THE FIX: A COMPETING HYPOTHESIS
===============================
The classical keyword-spotting answer is a KEYWORD-FILLER model: decode
against both the keyword and a background/filler model, and detect when the
keyword hypothesis beats the filler. The score becomes a LIKELIHOOD RATIO:

    score = logP(expected characters | audio) - logP(best free path | audio)

Both terms grow with window length, so the growth CANCELS -- which is
exactly what kills the monotonic-climb failure that sank rewrite 1.

The filler is nearly free here: the free path is the per-frame argmax over
characters, summed. No extra model, no extra forward pass.

Interpretation:
    ratio ~ 0      the constrained path is about as good as the
                   unconstrained one -- the audio really does contain these
                   characters
    ratio << 0     the free path is far better -- the audio contains
                   something ELSE

Recording_4 should collapse under this, because on a different verse the
free path wins enormously.

STREAMING DISCIPLINE
====================
One DP column per frame, O(U) work and memory per word, no lookahead, no
re-scanning. The filler accumulates in the same pass. Exactly what the
device will run.
"""

import argparse
import csv
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("student", HERE / "student.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
sys.argv = _argv

SR = S.SR
NEG = -1e30


# --------------------------------------------------------------------------
# audio / model plumbing
# --------------------------------------------------------------------------

def load_audio(path):
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"decode failed: {path}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def trim_silence(audio, thresh_frac=0.08, pad_ms=100):
    if len(audio) < SR // 4:
        return audio
    win = int(0.02 * SR)
    env = np.convolve(np.abs(audio), np.ones(win) / win, mode="same")
    thr = max(float(np.percentile(env, 95)) * thresh_frac, 1e-4)
    on = np.where(env > thr)[0]
    if len(on) < SR // 8:
        return audio
    pad = int(pad_ms * SR / 1000)
    return audio[max(0, int(on[0]) - pad):min(len(audio), int(on[-1]) + pad)]


def load_model(run, device):
    ck = torch.load(Path(run) / "best.pt", map_location=device,
                    weights_only=False)
    vocab = S.CharVocab()
    vocab.stoi = ck["vocab"]
    vocab.itos = {i: c for c, i in vocab.stoi.items()}
    model = S.CTCATVerifier(len(vocab.stoi), ck["d"],
                            ck.get("blocks", 6)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, vocab, ck


def reference_durations(manifest="manifest_full.csv"):
    """Median word duration in FRAMES, per (surah, ayah, word_idx), across
    the eight reference reciters.

    This is the duration model. Word 0 of 78:9 aligned over 3.75s in
    Recording_5 -- swallowing the next two words -- because nothing in the
    likelihood ratio penalised a long alignment: the constrained path can
    park in a blank state, and blank is usually the free-path winner too, so
    `acc` and `fill` rise by the SAME amount and the ratio is unchanged.
    Extending across arbitrary audio was free.

    Excluding blank from the filler is not the fix -- it inverts the bias,
    making a blank frame score +4.6 and REWARDING extension. A duration
    prior is the classical answer (penalise a unit held longer than
    expected) and here it costs nothing: eight reciters have already said
    every one of these words.
    """
    import collections
    by = collections.defaultdict(list)
    try:
        with open(manifest, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if int(r["multiword_segment"]):
                    continue
                dur_ms = int(r["end_ms"]) - int(r["start_ms"])
                by[(int(r["surah"]), int(r["ayah"]), int(r["word_idx"]))].append(
                    dur_ms * SR / 1000 / S.HOP)
    except FileNotFoundError:
        return {}
    return {k: float(np.median(v)) for k, v in by.items() if v}


def reftext(path="texts/quran-simple-plain.txt"):
    spec = importlib.util.spec_from_file_location("reftext", HERE / "reftext.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    return R.RefText(HERE / path)


@torch.no_grad()
def frame_logposteriors(model, audio, device):
    if len(audio) < 400:
        return np.zeros((0, 1))
    mel = S.logmel(audio)
    X = torch.from_numpy(mel[None]).to(device)
    h = model.audio(X)
    return F.log_softmax(model.ctc(h), dim=-1)[0].cpu().numpy()


# --------------------------------------------------------------------------
# the verifier
# --------------------------------------------------------------------------

class WordVerifier:
    """CTC Viterbi over one word, scored against a FILLER hypothesis.

    States alternate blank and label:  - y1 - y2 - ... yU -   (L = 2U+1)

    Per state:
        acc     accumulated CONSTRAINED path log-probability
        fill    accumulated FREE path log-probability over the SAME frames
        n       frames consumed
        start   global frame where this alignment began
        ch      best frame per character (diagnostic only)

    Score is (acc - fill) / n: a per-frame log likelihood ratio. Both terms
    grow with length, so the ratio does not.
    """

    def __init__(self, char_ids, blank=0):
        self.y = [int(c) for c in char_ids]
        self.U = len(self.y)
        self.L = 2 * self.U + 1
        self.state = [blank] * self.L
        for i, c in enumerate(self.y):
            self.state[2 * i + 1] = c
        self.reset(0)

    def reset(self, t0=0):
        self.acc = np.full(self.L, NEG)
        self.fill = np.zeros(self.L)
        self.n = np.zeros(self.L, dtype=np.int64)
        self.start = np.full(self.L, -1, dtype=np.int64)
        self.ch = np.full((self.L, self.U), NEG)
        self.t = t0

    def step(self, logp):
        free = float(np.max(logp))          # the filler: best free character
        acc, fill, n, st, ch = self.acc, self.fill, self.n, self.start, self.ch
        n_acc = np.full(self.L, NEG)
        n_fill = np.zeros(self.L)
        n_n = np.zeros(self.L, dtype=np.int64)
        n_st = np.full(self.L, -1, dtype=np.int64)
        n_ch = np.full((self.L, self.U), NEG)

        for l in range(self.L):
            p_l = float(logp[self.state[l]])
            best, src = (0.0, -2) if l <= 1 else (NEG, -1)
            for d in (0, 1, 2):
                p = l - d
                if p < 0 or acc[p] <= NEG / 2:
                    continue
                if d == 2 and (l % 2 == 0 or p % 2 == 0
                               or self.state[l] == self.state[p]):
                    continue
                if acc[p] > best:
                    best, src = acc[p], p
            if src == -1:
                continue

            n_acc[l] = best + p_l
            if src == -2:
                n_st[l], n_fill[l], n_n[l] = self.t, free, 1
            else:
                n_st[l] = st[src]
                n_fill[l] = fill[src] + free      # filler over the SAME frames
                n_n[l] = n[src] + 1
                n_ch[l] = ch[src]
            if l % 2 == 1:
                k = (l - 1) // 2
                n_ch[l, k] = max(n_ch[l, k], p_l)

        self.acc, self.fill, self.n = n_acc, n_fill, n_n
        self.start, self.ch = n_st, n_ch
        self.t += 1

    def best(self):
        """Best COMPLETE alignment ending at the frame just fed."""
        out = None
        for l in (self.L - 1, self.L - 2):
            if l < 0 or self.acc[l] <= NEG / 2 or self.n[l] <= 0:
                continue
            if np.any(self.ch[l] <= NEG / 2):
                continue                        # some character never emitted
            ratio = (self.acc[l] - self.fill[l]) / self.n[l]
            if out is None or ratio > out["score"]:
                out = {"score": float(ratio), "start": int(self.start[l]),
                       "end": self.t - 1, "n": int(self.n[l]),
                       "acc": float(self.acc[l]), "fill": float(self.fill[l]),
                       "chars": self.ch[l].copy()}
        return out


def verify_word(logp, t0, char_ids, patience=40, max_frames=600,
                ref_frames=None, span_mult=2.5):
    """Feed frames from t0 until the ratio stops improving.

    Peak rather than first-crossing: rewrite 1 fired on first crossing,
    which only measured where the threshold sat because its score rose
    monotonically. The ratio genuinely peaks at the word boundary and
    declines once the alignment starts absorbing the next word.
    """
    v = WordVerifier(char_ids)
    v.reset(t0)
    # An alignment may not run longer than span_mult x what the reference
    # reciters take for this word. Without it, word 0 of 78:9 spanned 3.75s
    # against a reference median near 0.8s.
    cap = int(ref_frames * span_mult) if ref_frames else max_frames
    cap = max(cap, len(char_ids) + 2)          # CTC still needs room
    best, since, trace = None, 0, []
    for f in range(t0, min(len(logp), t0 + max_frames)):
        v.step(logp[f])
        b = v.best()
        if b is None:
            continue
        if b["n"] > cap:
            continue                            # too long to be this word
        trace.append((f, b["score"], b["start"], b["n"]))
        if best is None or b["score"] > best["score"]:
            best, since = b, 0
        else:
            since += 1
            if since >= patience:
                break
    return best, trace


def walk(logp, words, vocab, thr, verbose=False, durs=None, key=None,
         span_mult=2.5):
    """Strictly sequential cursor: wait for word k before considering k+1."""
    t, cursor, out = 0, 0, []
    while cursor < len(words):
        chars, n = vocab.encode(words[cursor])
        rf = None if not durs or not key else durs.get((key[0], key[1], cursor))
        b, trace = verify_word(logp, t, chars[:n], ref_frames=rf,
                               span_mult=span_mult)
        if verbose:
            print(f"\n  word {cursor} {words[cursor]}  from {t*S.HOP/SR:.2f}s")
            for f, sc, stt, nn in trace[::4][:16]:
                bar = "#" * int(max(0, min(28, (sc + 3) * 9)))
                print(f"      end {f*S.HOP/SR:5.2f}s  ratio {sc:7.3f}  "
                      f"span {nn*S.HOP/SR:4.2f}s  {bar}")
        if b is None or b["score"] <= thr:
            out.append({"idx": cursor, "word": words[cursor], "stalled": True,
                        "best": None if b is None else b["score"]})
            break
        out.append({"idx": cursor, "word": words[cursor], "score": b["score"],
                    "start": b["start"], "end": b["end"], "n": b["n"]})
        t, cursor = b["end"] + 1, cursor + 1
    return out, cursor


# --------------------------------------------------------------------------

def parse_spec(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = [x.strip() for x in re.split(r"[|\t]", line)]
        if len(p) < 2:
            continue
        sa = p[1].split()
        note = p[2] if len(p) > 2 else ""
        rows.append({"file": p[0], "surah": int(sa[0]), "ayah": int(sa[1]),
                     "note": note, "is_error": "error" in note.lower()})
    return rows


def cmd_check(a):
    """Per-word ratios for every recording, then a threshold sweep.

    An operating point must confirm every word of each CORRECT recitation
    while stalling somewhere in each ERROR one. Neither previous rewrite
    managed this at any threshold.
    """
    device = torch.device(a.device)
    model, vocab, ck = load_model(a.run, device)
    rt = reftext(a.text)
    durs = reference_durations(a.manifest)
    rows = parse_spec(a.spec)
    print(f"duration model: {len(durs)} words from {a.manifest}")

    cache = []
    print(f"model {a.run} epoch {ck.get('epoch','?')}\n")
    print("per-word likelihood ratios (near 0 = audio matches the expected")
    print("characters; strongly negative = audio contains something else)\n")
    for r in rows:
        p = Path(a.dir) / r["file"]
        if not p.exists():
            print(f"  missing {p}")
            continue
        words = rt.words(r["surah"], r["ayah"])
        logp = frame_logposteriors(model, trim_silence(load_audio(str(p))),
                                   device)
        cache.append((r, words, logp))

        t, per, spans = 0, [], []
        for wi, w in enumerate(words):
            chars, n = vocab.encode(w)
            rf = durs.get((r["surah"], r["ayah"], wi))
            b, _ = verify_word(logp, t, chars[:n], ref_frames=rf,
                               span_mult=a.span_mult)
            if b is None:
                per.append(float("nan")); spans.append(0.0)
                break
            per.append(b["score"])
            spans.append(b["n"] * S.HOP / SR)
            t = b["end"] + 1
        tag = "ERROR  " if r["is_error"] else "correct"
        print(f"  {r['file']:<18} {r['surah']}:{r['ayah']:<3} {tag}  " +
              "  ".join(f"{v:6.2f}" for v in per) +
              "   spans " + " ".join(f"{x:.2f}s" for x in spans))

    if not cache:
        return
    print(f"\n  {'thr':>7}  " +
          "".join(f"{Path(r['file']).stem[-4:]:>7}" for r, _, _ in cache) +
          "     verdict")
    for thr in a.thrs:
        cells, ok_c, ok_e = [], True, True
        for r, words, logp in cache:
            _, n = walk(logp, words, vocab, thr, durs=durs,
                        key=(r["surah"], r["ayah"]), span_mult=a.span_mult)
            cells.append(f"{n}/{len(words)}")
            if r["is_error"] and n == len(words):
                ok_e = False
            if not r["is_error"] and n < len(words):
                ok_c = False
        v = "SEPARATES" if (ok_c and ok_e) else (
            "correct ok, errors pass" if ok_c else
            ("errors ok, correct stalls" if ok_e else ""))
        print(f"  {thr:7.2f}  " + "".join(f"{c:>7}" for c in cells) + f"     {v}")


def cmd_scan(a):
    """Ratios on reference reciters -- correct by construction -- to fix the
    scale before any threshold is chosen. Same move that settled v4's text."""
    device = torch.device(a.device)
    model, vocab, ck = load_model(a.run, device)
    rt = reftext(a.text)
    durs = reference_durations(a.manifest)

    rows = {}
    with open(a.manifest, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if a.reciter in r["reciter"]:
                rows.setdefault((int(r["surah"]), int(r["ayah"])),
                                r["audio_path"])

    vals, done = [], 0
    for key in sorted(rows)[:a.n]:
        words = rt.words(*key)
        if not words:
            continue
        try:
            logp = frame_logposteriors(
                model, trim_silence(load_audio(Path(a.lib) / rows[key])), device)
        except Exception:
            continue
        if len(logp) < 10:
            continue
        t = 0
        for wi, w in enumerate(words):
            chars, n = vocab.encode(w)
            b, _ = verify_word(logp, t, chars[:n],
                               ref_frames=durs.get((key[0], key[1], wi)),
                               span_mult=a.span_mult)
            if b is None:
                break
            vals.append(b["score"])
            t = b["end"] + 1
        done += 1

    v = np.array(vals)
    if not len(v):
        print("no words scored"); return
    print(f"{a.reciter}: {done} ayahs, {len(v)} words -- ALL CORRECT\n")
    print(f"  median {np.median(v):7.3f}   p25 {np.quantile(v,.25):7.3f}   "
          f"p05 {np.quantile(v,.05):7.3f}   p01 {np.quantile(v,.01):7.3f}   "
          f"min {v.min():7.3f}")
    print("\n  A threshold at p05 stalls on 5% of CORRECT words. A stall costs")
    print("  one repetition; a false confirmation teaches something wrong, so")
    print("  p05-p01 is the region to look at.")


def cmd_walk(a):
    device = torch.device(a.device)
    model, vocab, ck = load_model(a.run, device)
    words = reftext(a.text).words(a.ref[0], a.ref[1])
    raw = load_audio(a.audio)
    logp = frame_logposteriors(model, trim_silence(raw), device)
    durs = reference_durations(a.manifest)
    res, cursor = walk(logp, words, vocab, a.thr, a.verbose, durs,
                       (a.ref[0], a.ref[1]), a.span_mult)

    print(f"\naudio  {a.audio}  {len(raw)/SR:.2f}s ({len(logp)} frames)")
    print(f"verse  {a.ref[0]}:{a.ref[1]}  {len(words)} words   thr {a.thr}\n")
    for r in res:
        if r.get("stalled"):
            b = "n/a" if r["best"] is None else f"{r['best']:.3f}"
            print(f"  [{r['idx']}] {r['word']:<20} STALLED   best ratio {b}")
        else:
            print(f"  [{r['idx']}] {r['word']:<20} ok  "
                  f"{r['start']*S.HOP/SR:5.2f}-{r['end']*S.HOP/SR:5.2f}s  "
                  f"ratio {r['score']:7.3f}")
    print(f"\n  confirmed {cursor}/{len(words)}")


def main():
    ap = argparse.ArgumentParser(
        description="streaming word verifier (keyword-filler likelihood ratio)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check")
    c.add_argument("--spec", default="recs/descriptions.txt")
    c.add_argument("--dir", default="recs")
    c.add_argument("--thrs", type=float, nargs="+",
                   default=[-0.05, -0.1, -0.2, -0.3, -0.5, -0.75, -1.0, -1.5])
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("scan")
    s.add_argument("--reciter", default="Husary")
    s.add_argument("--lib", default="reference_library")
    s.add_argument("--manifest", default="manifest_full.csv")
    s.add_argument("--n", type=int, default=60)
    s.set_defaults(func=cmd_scan)

    w = sub.add_parser("walk")
    w.add_argument("--audio", required=True)
    w.add_argument("--ref", nargs=2, type=int, required=True,
                   metavar=("SURAH", "AYAH"))
    w.add_argument("--thr", type=float, default=-0.3)
    w.add_argument("--verbose", action="store_true")
    w.set_defaults(func=cmd_walk)

    for p in (c, s, w):
        p.add_argument("--manifest", default="manifest_full.csv")
        p.add_argument("--span-mult", type=float, default=2.5,
                       help="an alignment may not exceed this multiple of "
                            "the reference reciters' median duration for "
                            "the word")
        p.add_argument("--run", default="runs/student")
        p.add_argument("--text", default="texts/quran-simple-plain.txt")
        p.add_argument("--device", default="cuda")

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
