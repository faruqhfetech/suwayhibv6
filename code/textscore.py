#!/usr/bin/env python3
"""
Suwayhib v4 -- forced-decoding scorer.

    python textscore.py calibrate --device cuda      # sanity, not tuning
    python textscore.py score --audio r.m4a --ref 78 9 --device cuda

WHAT THIS REPLACES
------------------
The whole embedding pipeline: pairs mining, the contrastive head, the 891 MB
feature cache, embeddings.npz, the per-word gallery, centroids, the spread
floor, leave-one-out calibration and word_thresholds.json. All of it is
gone.

That pipeline was asking a model to learn, from six professional reciters,
what a word sounds like independent of who says it. The Whisper model
fine-tuned on Qur'anic Arabic was ALREADY trained to do exactly that, and
its decoder answers the question directly.

HOW IT WORKS
------------
The verse text is known before the learner recites. So teacher-force the
decoder with that text and read the probability it assigns to each token.
A token the learner produced correctly gets a high probability; one they
mispronounced gets a low one. This is the GOP idea (Witt & Young 2000)
applied at Whisper's token level, and it needs no training, no gallery and
no per-word threshold.

MEASURED SCALE (base-quran, prepared simple-plain text)
  Husary vs text            0.991 - 1.000   (60 ayahs, mean 0.9994)
  learner, correct          0.98  - 1.00
  learner, deliberate s->s  0.000
  wrong verse entirely      0.000
Nothing lands between 0.05 and 0.98, so a threshold near 0.10 has room.
Contrast the embedding scorer, where correct words and errors overlapped
inside per-word thresholds ranging 5.6 to 23.4.

WHY worst-token AND NOT mean
  A single wrong diacritic in an eight-token word barely moves the mean but
  IS the error. Observed: sub-ata correct 0.994 worst vs deliberate sad
  0.000 worst, where the means were 0.999 and 0.012. The min separates by
  ~80x, the mean by ~80x too but with a much narrower safe band.
"""

import argparse
import importlib.util
import re
import subprocess
from pathlib import Path

import numpy as np

SR = 16000
HERE = Path(__file__).resolve().parent

MODELS = {
    "whisper-tiny-quran": "tarteel-ai/whisper-tiny-ar-quran",
    "whisper-base-quran": "tarteel-ai/whisper-base-ar-quran",
}


def _reftext_module():
    spec = importlib.util.spec_from_file_location("reftext", HERE / "reftext.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_audio(path):
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"could not decode {path}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


class TextScorer:
    def __init__(self, model="whisper-base-quran",
                 text="texts/quran-simple-plain.txt", device="cpu"):
        import torch
        from transformers import (WhisperForConditionalGeneration,
                                  WhisperProcessor)
        self.torch = torch
        self.device = torch.device(device)
        repo = MODELS.get(model, model)
        self.proc = WhisperProcessor.from_pretrained(repo)
        self.model = WhisperForConditionalGeneration.from_pretrained(repo)
        self.model.eval().to(self.device)
        self.tok = self.proc.tokenizer
        # Whisper's decoder always starts with a forced task prefix. Omitting
        # it makes the FIRST real token collapse to ~0 for everyone: Husary
        # scored logP -15 on a word he recites perfectly.
        self.prefix = self.tok.convert_tokens_to_ids(
            ["<|startoftranscript|>", "<|ar|>", "<|transcribe|>",
             "<|notimestamps|>"])
        RT = _reftext_module()
        self.ref = RT.RefText(text)
        n = sum(p.numel() for p in self.model.parameters())
        print(f"textscore: {repo} ({n/1e6:.1f}M), "
              f"{len(self.ref.text)} verses, "
              f"{self.ref.n_basmala_stripped} basmala stripped")

    # ------------------------------------------------------------------
    def _feats(self, audio):
        f = self.proc(audio, sampling_rate=SR,
                      return_tensors="pt").input_features
        return f.to(self.device)

    def transcribe(self, audio):
        with self.torch.no_grad():
            ids = self.model.generate(self._feats(audio), max_new_tokens=128,
                                      num_beams=1)
        t = self.proc.batch_decode(ids, skip_special_tokens=True)[0]
        return re.sub(r"<\|[^|]*\|>", "", t).strip()

    def score(self, audio, surah, ayah):
        """-> {words: [{word, idx, mean, worst, weak_token}], text, heard}"""
        text = self.ref.get(surah, ayah)
        if not text:
            return None
        torch = self.torch
        body = self.tok(text, add_special_tokens=False).input_ids
        ids = torch.tensor([self.prefix + body], device=self.device)
        with torch.no_grad():
            out = self.model(input_features=self._feats(audio),
                             decoder_input_ids=ids[:, :-1])
            lp = torch.log_softmax(out.logits[0].float(), dim=-1)
            gold = ids[0, 1:]
            v = lp[torch.arange(len(gold)), gold].cpu().numpy()

        skip = len(self.prefix) - 1        # forced positions carry no signal
        gold, v = gold[skip:], v[skip:]
        pieces = [self.tok.decode([int(t)]) for t in gold]

        words, cur, curlp, curtok = [], "", [], []
        for p, x in zip(pieces, v):
            if p.startswith("<|"):
                continue
            if p.startswith(" ") and cur:
                words.append((cur, curlp, curtok))
                cur, curlp, curtok = p.strip(), [x], [p.strip()]
            else:
                cur = (cur + p) if cur else p.strip()
                curlp.append(x); curtok.append(p.strip())
        if cur:
            words.append((cur, curlp, curtok))

        out_words = []
        for i, (w, lps, toks) in enumerate(words):
            lps = np.array(lps)
            j = int(np.argmin(lps))
            out_words.append({
                "word": w, "idx": i,
                "mean": float(np.exp(lps.mean())),
                "worst": float(np.exp(lps.min())),
                "weak_token": toks[j],
            })
        return {"words": out_words, "text": text}

    def flag(self, audio, surah, ayah, threshold=0.10):
        """The device's decision. Returns None if nothing is wrong."""
        r = self.score(audio, surah, ayah)
        if not r:
            return None
        bad = [w for w in r["words"] if w["worst"] < threshold]
        if not bad:
            return {"flag": False, "words": r["words"]}
        # every word below threshold is suspect; the weakest is the one to
        # point at
        worst = min(bad, key=lambda w: w["worst"])
        return {"flag": True, "words": r["words"], "target": worst,
                "n_bad": len(bad),
                # a whole-verse collapse is a different KIND of error: the
                # learner is reciting something else entirely
                # a wrong verse collapses EVERY word (measured: 3/3 at
                # 0.000). Two bad words in a three-word verse is just two
                # errors, so require the whole verse to fail.
                "wrong_verse": (len(r["words"]) >= 3
                                and len(bad) == len(r["words"]))}


# --------------------------------------------------------------------------

def cmd_score(a):
    s = TextScorer(a.model, a.text, a.device)
    audio = load_audio(a.audio)
    surah, ayah = a.ref
    print(f"\nheard : {s.transcribe(audio)}")
    r = s.score(audio, surah, ayah)
    if not r:
        raise SystemExit(f"{surah}:{ayah} not in the text")
    print(f"text  : {r['text']}\n")
    print(f"  {'word':<24} {'mean':>7} {'worst':>7}   weak token")
    for w in r["words"]:
        mark = "  <-- FLAG" if w["worst"] < a.threshold else ""
        print(f"  {w['word']:<24} {w['mean']:7.3f} {w['worst']:7.3f}   "
              f"{w['weak_token']!r}{mark}")
    d = s.flag(audio, surah, ayah, a.threshold)
    if d["flag"]:
        if d["wrong_verse"]:
            print(f"\n-> looks like a DIFFERENT VERSE "
                  f"({d['n_bad']}/{len(d['words'])} words fail)")
        else:
            print(f"\n-> point at word {d['target']['idx']}: "
                  f"{d['target']['word']}  ({d['target']['worst']:.3f}, "
                  f"weak on {d['target']['weak_token']!r})")
    else:
        print("\n-> nothing to say. Silence.")


def cmd_calibrate(a):
    """Confirms the scale on reference reciters. This is a SANITY CHECK, not
    a tuning step -- the separation is large enough that the threshold does
    not need fitting."""
    import csv
    s = TextScorer(a.model, a.text, a.device)
    rows = {}
    for r in csv.DictReader(open(a.manifest, encoding="utf-8")):
        if a.reciter in r["reciter"]:
            rows.setdefault((int(r["surah"]), int(r["ayah"])), r["audio_path"])
    worsts = []
    for k in sorted(rows)[:a.n]:
        if not s.ref.get(*k):
            continue
        r = s.score(load_audio(Path(a.lib) / rows[k]), *k)
        if r:
            worsts.extend(w["worst"] for w in r["words"])
    worsts = np.array(worsts)
    print(f"\n{a.reciter}: {len(worsts)} words across {a.n} ayahs")
    print(f"  worst-token  median {np.median(worsts):.4f}  "
          f"p05 {np.quantile(worsts,0.05):.4f}  min {worsts.min():.4f}")
    print(f"  below {a.threshold}: {(worsts < a.threshold).sum()} "
          f"({(worsts < a.threshold).mean():.2%})")
    print("\nThese words are CORRECT, so anything below the threshold is a")
    print("false positive the device would show a learner. If that rate is")
    print("above a few percent, lower the threshold.")


def main():
    ap = argparse.ArgumentParser(description="forced-decoding scorer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("score", cmd_score), ("calibrate", cmd_calibrate)):
        p = sub.add_parser(name)
        p.add_argument("--model", default="whisper-base-quran")
        p.add_argument("--text", default="texts/quran-simple-plain.txt")
        p.add_argument("--device", default="cpu")
        p.add_argument("--threshold", type=float, default=0.10)
        if name == "score":
            p.add_argument("--audio", required=True)
            p.add_argument("--ref", nargs=2, type=int, required=True,
                           metavar=("SURAH", "AYAH"))
        else:
            p.add_argument("--reciter", default="Husary")
            p.add_argument("--lib", default="reference_library")
            p.add_argument("--manifest", default="manifest_full.csv")
            p.add_argument("--n", type=int, default=40)
        p.set_defaults(func=fn)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
