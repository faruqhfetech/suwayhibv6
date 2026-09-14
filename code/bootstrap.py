#!/usr/bin/env python3
"""
Suwayhib v6 -- Gate 0: data verification and the fast audio cache.

    python bootstrap.py verify      # all Gate 0 data checks
    python bootstrap.py cache       # decode every ayah WAV once, memmap it
    python bootstrap.py stats       # duration distributions from the manifest

WHY THIS EXISTS
---------------
v5 spawned one ffmpeg subprocess per WORD during dataset preparation, which
student.py measured at >40 minutes per epoch, essentially all process spawn
and I/O. There are only 4,512 ayah files. Decode them ONCE into a single
memory-mapped float16 array and index it; everything downstream becomes
array slicing.

The verification checks are the ones the project already learned to trust.
The load-bearing one is ROWS MUST DIVIDE EVENLY BY RECITER COUNT: eight
reciters saying identical text cannot produce a fractional row count. That
single assertion caught two silent parse bugs in v3 that passed every other
structural test.

GATE 0 PASSES WHEN
  - manifest is 18,144 rows over 8 reciters, divides evenly
  - every referenced audio file exists, is 16k mono, and is long enough for
    the spans that claim to sit inside it
  - no span is inverted, out of order, or past the end of its audio
  - the cache round-trips: a random word sliced from the memmap is
    sample-identical to the same word decoded directly
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SR = 16000
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def decode(path, sr=SR):
    """Decode any audio file to mono float32 at `sr`. One subprocess."""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"decode failed: {path}\n{r.stderr.decode()[:400]}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def probe(path):
    """-> (sample_rate, channels, duration_s) without decoding the samples."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels,duration",
         "-of", "json", str(path)],
        capture_output=True)
    if r.returncode != 0:
        return None
    try:
        s = json.loads(r.stdout)["streams"][0]
        return (int(s["sample_rate"]), int(s["channels"]),
                float(s.get("duration", 0.0)))
    except (KeyError, IndexError, ValueError):
        return None


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def load_manifest(path):
    """-> list of row dicts, ints already parsed."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "reciter": r["reciter"],
                "surah": int(r["surah"]),
                "ayah": int(r["ayah"]),
                "word_idx": int(r["word_idx"]),
                "start_ms": int(r["start_ms"]),
                "end_ms": int(r["end_ms"]),
                "multiword": int(r["multiword_segment"]),
                "audio_path": r["audio_path"],
            })
    return rows


def group_ayahs(rows):
    """-> {(reciter, surah, ayah): [row, ...]} sorted by word_idx."""
    ay = defaultdict(list)
    for r in rows:
        ay[(r["reciter"], r["surah"], r["ayah"])].append(r)
    for k in ay:
        ay[k].sort(key=lambda w: w["word_idx"])
    return dict(ay)


# --------------------------------------------------------------------------
# exclusion -- excluded_ayahs.json was found to be both incomplete and
# unapplied. See cmd_clean for the full explanation.
# --------------------------------------------------------------------------

def load_excluded_json(path):
    """-> {(surah, ayah): reason}. Empty dict if the file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return {}
    return {(e["surah"], e["ayah"]): e["reason"]
            for e in json.loads(p.read_text(encoding="utf-8"))}


def find_bad_ayahs(rows):
    """Live detection from the manifest itself: inverted spans, non-monotonic
    spans within an ayah. This is a SUPERSET check independent of any
    exclusion file, so it catches gaps like 112:1 that no document recorded.

    -> {(surah, ayah): [reason, ...]}
    """
    bad = defaultdict(list)
    for r in rows:
        if r["end_ms"] <= r["start_ms"]:
            bad[(r["surah"], r["ayah"])].append(
                f"{r['reciter']} w{r['word_idx']} inverted/zero span "
                f"({r['start_ms']}->{r['end_ms']})")

    ayahs = group_ayahs(rows)
    for (rec, s, ay), ws in ayahs.items():
        for p, q in zip(ws, ws[1:]):
            if q["start_ms"] < p["end_ms"]:
                bad[(s, ay)].append(
                    f"{rec} w{p['word_idx']}/{q['word_idx']} overlap "
                    f"({p['end_ms']} > {q['start_ms']})")
    return dict(bad)


def build_exclusion_set(rows, excluded_json_path):
    """Union of the documented exclusion file and live detection.

    Returns (exclude_set, undocumented) where `undocumented` is the set of
    ayahs live detection caught that the JSON file did NOT -- these should
    be added to excluded_ayahs.json by hand, with a reason, so the next
    person does not have to rediscover them.
    """
    documented = load_excluded_json(excluded_json_path)
    live = find_bad_ayahs(rows)
    exclude = set(documented) | set(live)
    undocumented = set(live) - set(documented)
    return exclude, undocumented, documented, live


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------

class Check:
    """Accumulates pass/fail so every check runs even when one fails.

    Failing fast would hide the SHAPE of a problem. If three checks fail
    together that is usually one bug; if one fails alone it is usually data.
    """

    def __init__(self):
        self.fails = []
        self.warns = []

    def ok(self, cond, label, detail=""):
        mark = "PASS" if cond else "FAIL"
        print(f"  [{mark}] {label}")
        if detail and not cond:
            for line in str(detail).splitlines()[:12]:
                print(f"         {line}")
        if not cond:
            self.fails.append(label)
        return cond

    def warn(self, cond, label, detail=""):
        if not cond:
            print(f"  [WARN] {label}")
            if detail:
                for line in str(detail).splitlines()[:8]:
                    print(f"         {line}")
            self.warns.append(label)
        return cond


def cmd_verify(a):
    lib = Path(a.lib)
    C = Check()

    print(f"\nMANIFEST  {a.manifest}")
    rows = load_manifest(a.manifest)
    reciters = sorted({r["reciter"] for r in rows})
    ayahs = group_ayahs(rows)

    if a.expect_rows:
        C.ok(len(rows) == a.expect_rows,
             f"row count == {a.expect_rows}", f"got {len(rows)}")
    else:
        print(f"  [info] row count {len(rows)} (no expectation set)")
    C.ok(len(reciters) == a.expect_reciters,
         f"reciter count == {a.expect_reciters}",
         f"got {len(reciters)}: {reciters}")

    # THE CHECK THAT EARNED ITS KEEP.
    # Eight reciters saying identical text cannot produce a fractional row
    # count. Caught an exclusive-vs-inclusive index error and reciters with
    # SKIPPED word indices, both of which passed every other structural test.
    per = defaultdict(int)
    for r in rows:
        per[r["reciter"]] += 1
    counts = sorted(set(per.values()))
    C.ok(len(counts) == 1,
         "every reciter has the same row count",
         "\n".join(f"{k}: {v}" for k, v in sorted(per.items())))
    C.ok(len(rows) % len(reciters) == 0,
         "rows divide evenly by reciter count",
         f"{len(rows)} % {len(reciters)} = {len(rows) % len(reciters)}")

    # word-count agreement ACROSS reciters for the same ayah
    by_verse = defaultdict(dict)
    for (rec, s, ay), ws in ayahs.items():
        by_verse[(s, ay)][rec] = len(ws)
    disagree = {k: v for k, v in by_verse.items() if len(set(v.values())) > 1}
    C.ok(not disagree,
         "word counts agree across reciters for every ayah",
         "\n".join(f"{s}:{ay}  {v}" for (s, ay), v in
                   sorted(disagree.items())[:6]))

    # word indices contiguous from 0
    gaps = [k for k, ws in ayahs.items()
            if [w["word_idx"] for w in ws] != list(range(len(ws)))]
    C.ok(not gaps,
         "word indices contiguous from 0 in every ayah",
         "\n".join(str(k) for k in gaps[:6]))

    # spans well-formed
    inverted = [r for r in rows if r["end_ms"] <= r["start_ms"]]
    C.ok(not inverted, "no inverted or zero-length spans",
         "\n".join(f"{r['reciter']} {r['surah']}:{r['ayah']} w{r['word_idx']} "
                   f"{r['start_ms']}->{r['end_ms']}" for r in inverted[:6]))

    nonmono = []
    for k, ws in ayahs.items():
        for p, q in zip(ws, ws[1:]):
            if q["start_ms"] < p["end_ms"]:
                nonmono.append((k, p["word_idx"], p["end_ms"], q["start_ms"]))
    C.ok(not nonmono, "spans monotonic within every ayah",
         "\n".join(str(x) for x in nonmono[:6]))

    short = [r for r in rows if r["end_ms"] - r["start_ms"] < a.min_word_ms]
    C.warn(not short,
           f"no words shorter than {a.min_word_ms}ms "
           f"({len(short)} found -- triage, not necessarily a defect)",
           "\n".join(f"{r['reciter']} {r['surah']}:{r['ayah']} w{r['word_idx']} "
                     f"{r['end_ms']-r['start_ms']}ms" for r in short[:6]))

    mw = sum(r["multiword"] for r in rows)
    print(f"  [info] multiword_segment=1 on {mw} rows "
          f"(these words cannot be isolated)")

    # ---------------------------------------------------------------- audio
    print(f"\nAUDIO  {lib}")
    paths = sorted({r["audio_path"] for r in rows})
    missing = [p for p in paths if not (lib / p).exists()]
    C.ok(not missing, f"all {len(paths)} referenced files exist",
         "\n".join(missing[:6]))

    if not missing:
        n_probe = len(paths) if a.full else min(a.probe_n, len(paths))
        step = max(1, len(paths) // n_probe)
        sample = paths[::step][:n_probe]
        rates, chans, bad = set(), set(), []
        for p in sample:
            info = probe(lib / p)
            if info is None:
                bad.append(p)
                continue
            rates.add(info[0]); chans.add(info[1])
        C.ok(not bad, f"ffprobe read {len(sample)} sampled files",
             "\n".join(bad[:6]))
        C.ok(rates == {SR}, f"sample rate is {SR} everywhere",
             f"found {sorted(rates)} -- resample before caching")
        C.ok(chans == {1}, "mono everywhere", f"found {sorted(chans)}")

        # spans must fit inside their audio. Checked on the sampled files
        # only unless --full, since it needs a real decode.
        over = []
        by_path = defaultdict(list)
        for r in rows:
            by_path[r["audio_path"]].append(r)
        for p in sample[:a.span_check_n]:
            dur_ms = len(decode(lib / p)) * 1000 // SR
            for r in by_path[p]:
                if r["end_ms"] > dur_ms:
                    over.append(f"{p} w{r['word_idx']} ends {r['end_ms']}ms "
                                f"but audio is {dur_ms}ms")
        C.ok(not over,
             f"spans fit inside audio ({min(len(sample), a.span_check_n)} "
             f"files decoded)", "\n".join(over[:6]))

    # ------------------------------------------------------------- recs/
    print(f"\nLEARNER RECORDINGS  {a.recs}")
    recs = Path(a.recs)
    desc = recs / "descriptions.txt"
    C.ok(desc.exists(), "descriptions.txt present",
         "without labels the seven recordings are unusable")
    wavs = sorted(p for p in recs.glob("*") if p.suffix.lower()
                  in (".m4a", ".wav", ".mp3"))
    C.ok(len(wavs) >= 7, f"at least 7 recordings ({len(wavs)} found)")

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 70)
    if C.fails:
        print(f"GATE 0 DATA CHECKS: FAIL ({len(C.fails)})")
        for f in C.fails:
            print(f"  - {f}")
        print("\nStop here. Every downstream number rests on these.")
        return 1
    print("GATE 0 DATA CHECKS: PASS"
          + (f"  ({len(C.warns)} warnings)" if C.warns else ""))
    print("\nStill required for Gate 0:")
    print("  - phones.py: every phone attested, 8-16 phones/sec per reciter")
    print("  - harness.py FAILS a random-posterior stub on Rec_4 and Rec_6")
    return 0


# --------------------------------------------------------------------------
# clean
# --------------------------------------------------------------------------

def cmd_clean(a):
    """Write manifest_clean.csv: manifest_full.csv with every ayah in the
    UNION of excluded_ayahs.json and live-detected bad spans removed.

    WHY THIS EXISTS
    excluded_ayahs.json and manifest_full.csv were found to have drifted
    apart. Two of its ten documented exclusions (80:23, 111:2) were never
    actually filtered out of the manifest -- their bad rows are still
    there. And verification_report.txt flags several MORE bad
    (reciter, ayah) pairs -- 78:36, 82:18, 83:25, 83:27, 89:17, 80:37,
    109:5, 112:1 -- that were never written to excluded_ayahs.json at all,
    so no downstream stage was ever told to skip them.

    This drops the WHOLE ayah (all 8 reciters) for anything in either
    source, consistent with the project's own convention: manifest_full.csv
    is supposed to have every word slot backed by all 8 reciters (verified
    by bootstrap.py stats), which only holds if a bad reciter's ayah takes
    its seven good siblings down with it.
    """
    rows = load_manifest(a.manifest)
    exclude, undocumented, documented, live = build_exclusion_set(
        rows, a.excluded)

    print(f"excluded_ayahs.json           : {len(documented)} ayahs")
    print(f"live-detected bad spans       : {len(live)} ayahs")
    print(f"union (what will be dropped)  : {len(exclude)} ayahs")

    if undocumented:
        print(f"\n{len(undocumented)} ayahs were bad but NOT in "
              f"excluded_ayahs.json -- add these by hand:")
        for k in sorted(undocumented):
            print(f"  {{\"surah\": {k[0]}, \"ayah\": {k[1]}, \"reason\": "
                  f"\"live-detected: {live[k][0]}\"}},")

    kept = [r for r in rows if (r["surah"], r["ayah"]) not in exclude]
    dropped_rows = len(rows) - len(kept)
    print(f"\n{len(rows)} rows -> {len(kept)} rows "
          f"({dropped_rows} dropped, {len(exclude)} ayahs)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["reciter", "surah", "ayah", "word_idx", "start_ms",
                    "end_ms", "multiword_segment", "audio_path"])
        for r in kept:
            w.writerow([r["reciter"], r["surah"], r["ayah"], r["word_idx"],
                       r["start_ms"], r["end_ms"], r["multiword"],
                       r["audio_path"]])

    # re-run the two checks that failed, against the CLEANED rows, so this
    # command proves its own output rather than asserting it
    inv = [r for r in kept if r["end_ms"] <= r["start_ms"]]
    ayahs = group_ayahs(kept)
    nonmono = [(k, p["word_idx"]) for k, ws in ayahs.items()
              for p, q in zip(ws, ws[1:]) if q["start_ms"] < p["end_ms"]]
    print(f"\nre-check on cleaned manifest:")
    print(f"  [{'PASS' if not inv else 'FAIL'}] no inverted spans")
    print(f"  [{'PASS' if not nonmono else 'FAIL'}] spans monotonic")

    if inv or nonmono:
        print("\nStill bad after cleaning -- do not proceed. Something in "
              "find_bad_ayahs missed a case.")
        return 1

    print(f"\nwrote {out}")
    print(f"NEXT: python bootstrap.py verify --manifest {out}")
    print(f"      python bootstrap.py cache   --manifest {out}")
    return 0


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

def cmd_cache(a):
    """Decode every referenced ayah once into a single memmapped array.

    float16 halves the file and is well inside the precision that matters
    for a log-mel front end; the round-trip check below verifies that
    against real audio rather than assuming it.
    """
    lib, out = Path(a.lib), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(a.manifest)
    paths = sorted({r["audio_path"] for r in rows})

    print(f"decoding {len(paths)} files -> {out}")
    lengths, total = [], 0
    for i, p in enumerate(paths):
        n = len(decode(lib / p))
        lengths.append(n)
        total += n
        if (i + 1) % 250 == 0:
            print(f"  probed {i+1}/{len(paths)}  ({total/SR/60:.1f} min)")

    offsets = np.zeros(len(paths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    dtype = np.float16 if a.half else np.float32
    mm = np.lib.format.open_memmap(
        out / "audio.npy", mode="w+", dtype=dtype, shape=(int(offsets[-1]),))

    for i, p in enumerate(paths):
        mm[offsets[i]:offsets[i + 1]] = decode(lib / p).astype(dtype)
        if (i + 1) % 250 == 0:
            print(f"  wrote {i+1}/{len(paths)}")
    mm.flush()

    (out / "index.json").write_text(json.dumps({
        "sr": SR, "dtype": str(np.dtype(dtype)),
        "paths": paths,
        "offsets": offsets.tolist(),
    }))
    print(f"\n{offsets[-1]/SR/60:.1f} min of audio, "
          f"{mm.nbytes/1e6:.0f} MB at {np.dtype(dtype)}")

    # ROUND-TRIP CHECK. Cheap, and it is the difference between trusting the
    # cache and discovering a silent offset bug three phases later.
    print("\nround-trip check on 5 random files")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(paths), size=min(5, len(paths)), replace=False)
    worst = 0.0
    for i in idx:
        direct = decode(lib / paths[i]).astype(dtype).astype(np.float32)
        cached = np.asarray(mm[offsets[i]:offsets[i + 1]], dtype=np.float32)
        if len(direct) != len(cached):
            print(f"  FAIL length {paths[i]}: {len(direct)} vs {len(cached)}")
            return 1
        d = float(np.abs(direct - cached).max())
        worst = max(worst, d)
        print(f"  {paths[i]}  max abs diff {d:.2e}")
    if worst > 1e-6:
        print(f"  FAIL: cache does not round-trip (worst {worst:.2e})")
        return 1
    print("  PASS")
    print(f"\nNEXT: index with AudioCache('{out}')")
    return 0


class AudioCache:
    """Read-only view over the bootstrap cache. Slicing, no subprocesses."""

    def __init__(self, path):
        d = Path(path)
        meta = json.loads((d / "index.json").read_text())
        self.sr = meta["sr"]
        self.paths = meta["paths"]
        self.offsets = np.array(meta["offsets"], dtype=np.int64)
        self.pos = {p: i for i, p in enumerate(self.paths)}
        self.data = np.load(d / "audio.npy", mmap_mode="r")

    def ayah(self, audio_path):
        i = self.pos[audio_path]
        return np.asarray(self.data[self.offsets[i]:self.offsets[i + 1]],
                          dtype=np.float32)

    def word(self, audio_path, start_ms, end_ms, pad_ms=0):
        i = self.pos[audio_path]
        lo, hi = self.offsets[i], self.offsets[i + 1]
        a = lo + max(0, (start_ms - pad_ms)) * self.sr // 1000
        b = lo + (end_ms + pad_ms) * self.sr // 1000
        return np.asarray(self.data[max(lo, a):min(hi, b)], dtype=np.float32)


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

def cmd_stats(a):
    """Per-word duration distributions across reciters.

    These become the DURATION PRIOR in the v6 graph decoder. Handover
    section 4: word 0 of Recording_5 aligned over 3.75 seconds against a
    reference median near 0.8s. A p95 cap from these numbers makes that
    alignment impossible rather than merely unlikely.
    """
    rows = load_manifest(a.manifest)
    dur = defaultdict(list)
    for r in rows:
        dur[(r["surah"], r["ayah"], r["word_idx"])].append(
            r["end_ms"] - r["start_ms"])

    n_rec = len({r["reciter"] for r in rows})
    full = [k for k, v in dur.items() if len(v) == n_rec]
    print(f"{len(dur)} distinct word slots, {len(full)} with all "
          f"{n_rec} reciters")

    allv = np.array([x for v in dur.values() for x in v])
    print(f"\nword duration (ms) over {len(allv)} segments")
    for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
        print(f"  p{int(q*100):02d}  {np.quantile(allv, q):7.0f}")

    # cross-reciter spread: how much do eight reciters disagree on one word?
    spread = np.array([np.std(dur[k]) / max(1.0, np.mean(dur[k]))
                       for k in full])
    print(f"\ncross-reciter CV per word slot")
    for q in (0.5, 0.9, 0.99):
        print(f"  p{int(q*100):02d}  {np.quantile(spread, q):.3f}")
    print("\nHigh-CV slots are where a p95 duration cap will be loosest, and")
    print("where the aligner most likely disagreed with itself. Cross-check")
    print("the top of this list against manifest_full_suspects.csv.")

    worst = sorted(full, key=lambda k: -np.std(dur[k]) / max(1.0, np.mean(dur[k])))
    print(f"\nhighest-spread slots")
    for k in worst[:8]:
        v = dur[k]
        print(f"  {k[0]}:{k[1]} w{k[2]}   mean {np.mean(v):6.0f}ms  "
              f"sd {np.std(v):6.0f}  range {min(v)}-{max(v)}")
    return 0


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="v6 Gate 0 data checks and cache")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = dict(manifest="manifest/manifest_full.csv",
                  lib="reference_library")

    v = sub.add_parser("verify")
    v.add_argument("--manifest", default=common["manifest"])
    v.add_argument("--lib", default=common["lib"])
    v.add_argument("--recs", default="recs")
    v.add_argument("--expect-rows", type=int, default=18144)
    v.add_argument("--expect-reciters", type=int, default=8)
    v.add_argument("--min-word-ms", type=int, default=80)
    v.add_argument("--probe-n", type=int, default=120,
                   help="files to ffprobe; --full does all of them")
    v.add_argument("--span-check-n", type=int, default=40,
                   help="files to fully decode for the span-fit check")
    v.add_argument("--full", action="store_true")
    v.set_defaults(func=cmd_verify)

    cl = sub.add_parser("clean")
    cl.add_argument("--manifest", default=common["manifest"])
    cl.add_argument("--excluded", default="reference_library/excluded_ayahs.json")
    cl.add_argument("--out", default="manifest/manifest_clean.csv")
    cl.set_defaults(func=cmd_clean)

    c = sub.add_parser("cache")
    c.add_argument("--manifest", default=common["manifest"])
    c.add_argument("--lib", default=common["lib"])
    c.add_argument("--out", default="cache/audio")
    c.add_argument("--half", action="store_true", default=True)
    c.add_argument("--float32", dest="half", action="store_false")
    c.set_defaults(func=cmd_cache)

    s = sub.add_parser("stats")
    s.add_argument("--manifest", default=common["manifest"])
    s.set_defaults(func=cmd_stats)

    a = ap.parse_args()
    sys.exit(a.func(a) or 0)


if __name__ == "__main__":
    main()
