#!/usr/bin/env python3
"""
Suwayhib v6 -- derive the phones.py <-> IqraEval symbol correspondence.

    python -m suwayhib.pipeline.align_phonemes probe   --n 200
    python -m suwayhib.pipeline.align_phonemes derive  --n 2000 --out manifest/iqra_alias.json

WHY THIS EXISTS
---------------
phones.py uses our own symbol names (H7, S9, q0, ...). IqraEval's released
data uses a 68-symbol MSA inventory in THEIR spelling (f, ii, h, i, nn, a,
x, A, y, r, aa, t, H, s, n, ...). Training on one alphabet and scoring
against QuranMB.v2 in the other is meaningless, and QuranMB.v2 is the only
externally-published number this project can be compared against.

The mapping is DERIVED, not declared. Iqra_train ships both the vowelised
sentence AND its phoneme_ref, so we can phonetise the same sentence with
phones.py and align the two sequences. Doing this over thousands of
sentences turns a guess into a measurement, and -- more usefully -- makes
DISAGREEMENTS visible, which is where the real information is.

WHAT TO EXPECT, AND WHY DISAGREEMENT IS NOT NECESSARILY OUR BUG
----------------------------------------------------------------
Some divergence is expected and legitimate:

  - phones.py deliberately does NOT model cross-word hamzat-wasl elision
    (documented limitation). IqraEval's phonetiser runs over whole
    sentences and does. So word-initial hamza after a vowel will differ.

  - IqraEval's inventory descends from Halabi's TTS phonetiser, which
    distinguishes vowels after emphatic vs non-emphatic consonants. Their
    paper says they collapsed that distinction for MSA -- but the released
    data appears to retain a separate 'A' symbol alongside 'a', so the
    collapse may be partial or applied at a different stage. Measuring
    tells us which.

  - Their 'nn'-style gemination doubling matches ours by construction, but
    only if our shadda-reordering fix (phones.py) fires the same way on
    their text, which uses a different diacritic ordering convention than
    Tanzil in places.

So: a symbol that maps 1:1 with high consistency is SETTLED. A symbol that
maps to two or more targets is a QUESTION to answer, not a bug to paper
over -- exactly the posture the project's suspects-list already takes with
manifest_full_suspects.csv.

METHOD
------
Per sentence, phonetise with phones.py and compare against phoneme_ref by
Levenshtein alignment (not naive zip -- the sequences will differ in
length, and a single insertion would otherwise corrupt every subsequent
pair). Accumulate a contingency table of ours -> theirs over all aligned
substitution/match positions, then report:

  - the modal target for each of our symbols, and how dominant it is
  - our symbols with no dominant target (ambiguous -- needs a rule, not
    an alias)
  - their symbols we never produce (coverage gaps in our inventory)

Only symbols above --min-confidence are written to the alias file. The
rest are reported and left out deliberately: a wrong alias is worse than a
missing one, because it silently corrupts every downstream label.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..core import phones as P


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------

def align(a, b):
    """Levenshtein alignment of two symbol lists.

    -> list of (a_sym or None, b_sym or None) pairs in order.

    Naive positional zip() would be wrong here: our sequence and theirs
    differ in length wherever either side inserts or drops a symbol (and
    they will, e.g. on hamzat-wasl), and a single offset would then
    mis-pair every remaining position, producing a contingency table that
    looks confidently wrong rather than obviously wrong.
    """
    n, m = len(a), len(b)
    # dp[i][j] = (cost, backpointer)
    dp = [[(0, None)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, "del")
    for j in range(1, m + 1):
        dp[0][j] = (j, "ins")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1][0] + (0 if a[i - 1] == b[j - 1] else 1)
            dele = dp[i - 1][j][0] + 1
            ins = dp[i][j - 1][0] + 1
            best = min(sub, dele, ins)
            op = "sub" if best == sub else ("del" if best == dele else "ins")
            dp[i][j] = (best, op)

    out, i, j = [], n, m
    while i > 0 or j > 0:
        op = dp[i][j][1]
        if op == "sub":
            out.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif op == "del":
            out.append((a[i - 1], None)); i -= 1
        else:
            out.append((None, b[j - 1])); j -= 1
    return list(reversed(out))


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def iter_samples(n, split="train", dataset="IqraEval/Iqra_train"):
    """Stream sentence/phoneme pairs WITHOUT decoding audio.

    remove_columns on 'audio' matters: decoding audio for every sample is
    slow and, in streaming mode, spawns decoder threads that crash noisily
    at interpreter shutdown. We only need the text columns here.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split, streaming=True)
    ds = ds.remove_columns([c for c in ("audio",) if c in ds.column_names])
    for i, r in enumerate(ds):
        if i >= n:
            break
        text = r.get("tashkeel_sentence") or r.get("sentence")
        ref = r.get("phoneme_ref")
        if text and ref:
            yield text, ref.split()


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_probe(a):
    """Eyeball a handful of alignments before trusting any aggregate."""
    for k, (text, theirs) in enumerate(iter_samples(a.n)):
        ours = []
        for w in text.split():
            ours.extend(P.word_to_phones_iqra(w))
        pairs = align(ours, theirs)
        print(f"\n--- sample {k} ---")
        print(f"text   : {text}")
        print(f"theirs : {' '.join(theirs)}")
        print(f"ours   : {' '.join(ours)}")
        diffs = [(x, y) for x, y in pairs if x != y]
        print(f"aligned: {len(pairs)}  differing: {len(diffs)}")
        if diffs:
            print("  " + "  ".join(f"{x or '-'}->{y or '-'}" for x, y in diffs[:14]))
        if k + 1 >= a.show:
            break
    return 0


def cmd_derive(a):
    table = defaultdict(Counter)      # ours -> Counter(theirs)
    ours_total = Counter()
    theirs_total = Counter()
    n_sent = 0
    n_pairs = n_diff = 0

    for text, theirs in iter_samples(a.n):
        ours = []
        for w in text.split():
            ours.extend(P.word_to_phones_iqra(w))
        if not ours:
            continue
        n_sent += 1
        theirs_total.update(theirs)
        ours_total.update(ours)
        for x, y in align(ours, theirs):
            if x is None or y is None:
                continue              # insertion/deletion: no correspondence
            table[x][y] += 1
            n_pairs += 1
            n_diff += (x != y)

    print(f"sentences        : {n_sent}")
    print(f"aligned positions: {n_pairs}   differing: {n_diff} "
          f"({n_diff / max(n_pairs, 1):.1%})")

    alias, ambiguous = {}, []
    print(f"\n{'ours':<8} {'->':2} {'theirs':<8} {'conf':>7} {'n':>8}   "
          f"runner-up")
    for sym in sorted(table, key=lambda s: -ours_total[s]):
        c = table[sym]
        tot = sum(c.values())
        top, n_top = c.most_common(1)[0]
        conf = n_top / tot
        rest = c.most_common(3)[1:]
        runner = "  ".join(f"{s}:{k}" for s, k in rest) or "-"
        mark = ""
        if conf >= a.min_confidence:
            if top != sym:
                alias[sym] = top
        else:
            ambiguous.append((sym, c.most_common(4)))
            mark = "  <-- AMBIGUOUS"
        print(f"{sym:<8} {'->':2} {top:<8} {conf:7.1%} {tot:8d}   "
              f"{runner}{mark}")

    if ambiguous:
        print(f"\n{len(ambiguous)} AMBIGUOUS symbols -- NOT written to the "
              f"alias file.")
        print("These need a phones.py RULE, not an alias: one of our symbols")
        print("is being used where their inventory makes a distinction we do")
        print("not. Look at the split before deciding.")
        for sym, common in ambiguous:
            print(f"  {sym:<8} " + "  ".join(f"{s}:{k}" for s, k in common))

    never = sorted(set(theirs_total) - {v for v in alias.values()}
                   - set(table.keys()))
    if never:
        print(f"\ntheir symbols we never align to ({len(never)}): "
              f"{' '.join(never)}")
        print("Coverage gap: either genuinely absent from this sample, or a")
        print("distinction phones.py does not make. Check before training.")

    print(f"\ntheir inventory size in sample: {len(theirs_total)} "
          f"(challenge states 68)")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({
            "alias": alias,
            "derived_from": {"dataset": "IqraEval/Iqra_train",
                             "sentences": n_sent, "pairs": n_pairs},
            "min_confidence": a.min_confidence,
            "ambiguous": [s for s, _ in ambiguous],
        }, indent=2, ensure_ascii=False))
        print(f"\nwrote {a.out}  ({len(alias)} aliases, "
              f"{len(ambiguous)} withheld as ambiguous)")
        print("Paste `alias` into phones.py IQRA_ALIAS once the ambiguous")
        print("ones are resolved -- a partial alias table silently corrupts")
        print("every label it touches, so do not wire it up half-done.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="derive IqraEval symbol alias")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--show", type=int, default=6)
    p.set_defaults(func=cmd_probe)

    d = sub.add_parser("derive")
    d.add_argument("--n", type=int, default=2000)
    d.add_argument("--min-confidence", type=float, default=0.90)
    d.add_argument("--out", default=None)
    d.set_defaults(func=cmd_derive)

    a = ap.parse_args()
    rc = a.func(a) or 0
    # The streaming datasets reader leaves an audio-decoder thread alive;
    # at interpreter finalisation it tries to release a GIL it no longer
    # legitimately holds and the runtime aborts with a core dump AFTER all
    # results are printed. Cosmetic, but alarming in a log. _exit skips
    # finalisation entirely. Flush first, or buffered output is lost.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
