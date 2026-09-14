#!/usr/bin/env python3
"""
Suwayhib v4 -- reference text preparation.

    python -m suwayhib.core.reftext check --texts texts --device cuda
    python -m suwayhib.core.reftext show 78 1

THE THREE RULES  (all derived empirically, not assumed)
-------------------------------------------------------
1. Use quran-simple-plain. Scoring a professional reciter against every
   Tanzil variant gave: simple-plain 0.87, simple 0.65, uthmani 0.25,
   the -min variants 0.00. The Tarteel models were fine-tuned on imlaei
   script, so Uthmani orthography penalises correct recitation heavily.

2. Strip U+0670 (ARABIC LETTER SUPERSCRIPT ALEF). This ONE character
   accounted for 9 of 11 mismatched ayahs: every alif-maqsura ending
   (مُوسَىٰ, إِلَىٰ, طَغَىٰ) and الرَّحْمَٰن. Removing it took Husary from
   11/60 failures to 2/60.

3. Strip the basmala from ayah 1 of every surah except 1 and 9. Tanzil
   prepends it; the audio does not contain it, so forced decoding is asked
   to find words that were never recited and the whole verse collapses.

   The basmala prefix is taken FROM THE TEXT ITSELF -- verse 1:1 IS the
   basmala -- rather than from a hand-typed constant. Hand-typing it failed
   twice: the superscript alef in الرَّحْمَٰن does not survive a shell paste
   reliably, so the literal never matched.

Diacritics are REQUIRED: simple-clean, and any variant with harakat
stripped, scored exactly 0.000. The model genuinely hears vowelling, which
is what makes the per-token diagnosis (kasra vs sukun) meaningful.
"""

import argparse
import re
import unicodedata
from pathlib import Path

SUPERSCRIPT_ALEF = "\u0670"
DEFAULT_TEXT = "texts/quran-simple-plain.txt"

AYAH_COUNTS = [
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128,
    111, 110, 98, 135, 112, 78, 118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73,
    54, 45, 83, 182, 88, 75, 85, 54, 53, 89, 59, 37, 35, 38, 29, 18, 45, 60,
    49, 62, 55, 78, 96, 29, 22, 24, 13, 14, 11, 11, 18, 12, 12, 30, 52, 52,
    44, 28, 28, 20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25, 22, 17, 19,
    26, 30, 20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3,
    6, 3, 5, 4, 5, 6,
]


def _raw_load(path):
    lines = [l.rstrip("\n") for l in
             Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [l for l in lines if l.strip() and not l.startswith("#")]

    out = {}
    if lines and re.match(r"^\d+\s*\|\s*\d+\s*\|", lines[0]):
        for l in lines:
            p = l.split("|", 2)
            if len(p) == 3:
                out[(int(p[0]), int(p[1]))] = p[2].strip()
        return out

    i = 0
    for s, n in enumerate(AYAH_COUNTS, start=1):
        for a in range(1, n + 1):
            if i < len(lines):
                out[(s, a)] = lines[i].strip()
            i += 1
    return out


class RefText:
    """Reference verse text, prepared for forced decoding."""

    def __init__(self, path=DEFAULT_TEXT):
        raw = _raw_load(path)
        if (1, 1) not in raw:
            raise SystemExit(f"{path}: no verse 1:1 -- wrong format?")

        # 1:1 IS the basmala, so take the prefix from the data. Normalising
        # both sides the same way means the match cannot fail on an invisible
        # character, which is exactly how two hand-written attempts failed.
        self.basmala = self._norm(raw[(1, 1)])

        self.text = {}
        stripped = 0
        for (s, a), t in raw.items():
            t = self._norm(t)
            if a == 1 and s not in (1, 9) and t.startswith(self.basmala):
                t = t[len(self.basmala):].strip()
                stripped += 1
            self.text[(s, a)] = t
        self.n_basmala_stripped = stripped

    @staticmethod
    def _norm(t):
        t = t.replace(SUPERSCRIPT_ALEF, "")     # rule 2, applied FIRST
        return re.sub(r"\s+", " ", t).strip()

    def get(self, surah, ayah):
        return self.text.get((surah, ayah))

    def words(self, surah, ayah):
        t = self.get(surah, ayah)
        return t.split() if t else []

    def report(self):
        print(f"verses loaded         : {len(self.text)}")
        print(f"basmala stripped from : {self.n_basmala_stripped} ayahs "
              f"(expected 112: all surahs except 1 and 9)")
        if self.n_basmala_stripped != 112:
            print("  WARNING: expected 112. Ayah 1 of the affected surahs "
                  "will false-flag, because the audio never contains the "
                  "basmala the text is asking the model to find.")


# --------------------------------------------------------------------------

def cmd_show(a):
    rt = RefText(a.text)
    rt.report()
    for k in [(1, 1), (a.surah, a.ayah), (2, 1), (9, 1)]:
        t = rt.get(*k)
        print(f"\n{k[0]}:{k[1]}  {t}")
        print(f"   words: {rt.words(*k)}")


def cmd_check(a):
    """Score a professional reciter -- correct by construction -- against the
    prepared text. He should sit near 1.000 everywhere. Anything low is the
    TEXT being wrong, not the recitation."""
    import csv
    import subprocess
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    SR = 16000
    repo = {"whisper-tiny-quran": "tarteel-ai/whisper-tiny-ar-quran",
            "whisper-base-quran": "tarteel-ai/whisper-base-ar-quran"
            }.get(a.model, a.model)
    dev = torch.device(a.device)
    proc = WhisperProcessor.from_pretrained(repo)
    model = WhisperForConditionalGeneration.from_pretrained(repo).eval().to(dev)
    tok = proc.tokenizer
    prefix = tok.convert_tokens_to_ids(
        ["<|startoftranscript|>", "<|ar|>", "<|transcribe|>",
         "<|notimestamps|>"])

    rt = RefText(a.text)
    rt.report()

    rows = {}
    for r in csv.DictReader(open(a.manifest, encoding="utf-8")):
        if a.reciter in r["reciter"]:
            rows.setdefault((int(r["surah"]), int(r["ayah"])), r["audio_path"])

    def worst(p, ref):
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(Path(a.lib) / p), "-f", "f32le",
             "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SR), "-"],
            capture_output=True).stdout
        au = np.frombuffer(raw, dtype=np.float32).copy()
        f = proc(au, sampling_rate=SR, return_tensors="pt").input_features.to(dev)
        ids = torch.tensor(
            [prefix + tok(ref, add_special_tokens=False).input_ids], device=dev)
        with torch.no_grad():
            lp = torch.log_softmax(
                model(input_features=f, decoder_input_ids=ids[:, :-1]
                      ).logits[0].float(), -1)
            g = ids[0, 1:]
            v = lp[torch.arange(len(g)), g].cpu().numpy()
        v = v[len(prefix) - 1:]
        pcs = [tok.decode([int(t)]) for t in g[len(prefix) - 1:]]
        keep = [x for p_, x in zip(pcs, v) if not p_.startswith("<|")]
        return float(np.exp(min(keep)))

    keys = sorted(rows)[:a.n]
    res = sorted((worst(rows[k], rt.get(*k)), k) for k in keys if rt.get(*k))
    bad = [x for x in res if x[0] < 0.5]

    print(f"\n{a.reciter} on {len(res)} ayahs -- he is CORRECT, so low = the "
          f"TEXT is wrong")
    print(f"  mean worst-token : {np.mean([w for w, _ in res]):.4f}")
    print(f"  below 0.5        : {len(bad)}")
    for w, k in res[:8]:
        mark = "  <-- TEXT MISMATCH" if w < 0.5 else ""
        print(f"    {w:.3f}  {k[0]}:{k[1]}{mark}")
        if w < 0.5:
            print(f"           {rt.get(*k)[:70]}")


def main():
    ap = argparse.ArgumentParser(description="reference text preparation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("show", cmd_show), ("check", cmd_check)):
        p = sub.add_parser(name)
        p.add_argument("--text", default=DEFAULT_TEXT)
        if name == "show":
            p.add_argument("surah", type=int, nargs="?", default=78)
            p.add_argument("ayah", type=int, nargs="?", default=1)
        else:
            p.add_argument("--model", default="whisper-base-quran")
            p.add_argument("--device", default="cpu")
            p.add_argument("--reciter", default="Husary")
            p.add_argument("--lib", default="reference_library")
            p.add_argument("--manifest", default="manifest_full.csv")
            p.add_argument("--n", type=int, default=60)
        p.set_defaults(func=fn)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
