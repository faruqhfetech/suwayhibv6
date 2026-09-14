#!/usr/bin/env python3
"""
Suwayhib v6 -- the demo.

    python -m suwayhib.pipeline.demo                          # every recs/ item, in order
    python -m suwayhib.pipeline.demo --file Recording_4.m4a    # one specific item
    python -m suwayhib.pipeline.demo --speed 4                 # 4x realtime
    python -m suwayhib.pipeline.demo --play                    # also play the audio

WHAT THIS PROVES, AND WHAT IT DOES NOT
-----------------------------------------
This is the first time anything in the project is watched running rather
than read as a table. It answers: does the decoder, wired end to end,
actually behave like the product -- a green edge that only ever moves
forward, that stops without accusation when something is wrong?

WHAT IT DOES NOT PROVE: true incremental encoding. XLS-R is a full
self-attention transformer over the whole waveform, not a chunked or
causal encoder -- there is no "encode the next 20ms" operation to call.
So this demo computes the WHOLE utterance's frame posteriors once,
upfront, and then feeds them into StreamingDecoder.step() one frame at a
time, paced to real wall-clock time. The DECODER is genuinely running
incrementally, exactly as it would in production -- that is the part
this project has spent real effort getting right, and it is what this
demo actually exercises. The ENCODER is not yet incremental; making it
so (a chunked/causal audio pipeline feeding the browser AudioWorklet) is
Phase 6's job, not this one's. Stated here rather than left implicit,
because quietly presenting a precomputed replay as "streaming" would be
the kind of thing this project has otherwise been careful not to do.

RENDERING, AND WHY THERE IS NO RED
-------------------------------------
Confirmed words are green. The word currently being evaluated is
underlined. Everything after it is dim. When the edge stops, the stuck
word gets a plain marker and a text line explaining what happened --
never a colour that reads as "wrong". This is not a UI nicety; it is the
same principle from handover section 1, restated here because a demo is
where it is actually visible: a false stall costs the learner one
repetition, but "your Qur'an recitation is being marked wrong on
screen" is a cost this project has decided is never worth paying, even
when the flag might be correct.

WHY THE WHOLE SCREEN REDRAWS RATHER THAN AN IN-PLACE \\r UPDATE
------------------------------------------------------------------
Arabic is RTL and terminal width calculation for mixed-direction, mixed
-width ANSI-coloured text is fragile across terminals. Clearing and
reprinting the full block every redraw is slower but never garbles.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..core import phones as P
from ..core import reftext as RT
from . import decode as D
from . import harness as H
from . import score as S

FRAMES_PER_SEC = 50


# --------------------------------------------------------------------------
# terminal rendering
# --------------------------------------------------------------------------

GREEN = "\033[32m"
DIM = "\033[2m"
UNDERLINE = "\033[4m"
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[H"


def supports_color():
    return sys.stdout.isatty()


def render(words, confirmed_upto, cursor_word, stalled, elapsed_s,
          total_s, note, use_color=True):
    """One full frame of the terminal display.

    confirmed_upto: number of words confirmed so far (monotonic)
    cursor_word: index currently being evaluated (== confirmed_upto
                unless the whole ayah is done)
    """
    def c(code, s):
        return f"{code}{s}{RESET}" if use_color else s

    parts = []
    for i, w in enumerate(words):
        if i < confirmed_upto:
            parts.append(c(GREEN + BOLD, w))
        elif i == cursor_word:
            parts.append(c(UNDERLINE, w))
        else:
            parts.append(c(DIM, w))
    # Arabic is RTL; joining left-to-right in source order and letting the
    # terminal's own bidi handling take over is the standard approach and
    # is what every other Arabic-producing tool in this project already
    # assumes (phones.py's printed samples, textscore.py's output).
    line = "  ".join(parts)

    bar_w = 30
    filled = int(bar_w * min(1.0, elapsed_s / max(total_s, 0.01)))
    bar = "#" * filled + "-" * (bar_w - filled)

    out = [line, "", f"[{bar}] {elapsed_s:4.1f}s / {total_s:4.1f}s"]
    if stalled:
        out.append("")
        out.append(c(BOLD, "  the edge has stopped -- "
                            "repeat the last word to continue"))
    if note:
        out.append("")
        out.append(c(DIM, note))
    return "\n".join(out)


# --------------------------------------------------------------------------
# audio playback (best-effort, never fatal)
# --------------------------------------------------------------------------

def start_playback(path):
    """ffplay, if available. Returns the Popen handle or None. Failure
    here should never stop the visual demo -- audio is a nice-to-have,
    the decoder trace is the point."""
    if not shutil.which("ffplay"):
        print("  (ffplay not found -- running visual-only, no audio)")
        return None
    try:
        return subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
             str(path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"  (audio playback failed: {e} -- running visual-only)")
        return None


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def run_one(item, tau, dwell_cap, commit_lag, speed, redraw_every, play,
           use_color):
    d, words_ph, logp = item

    proc = start_playback(d.get("_audio_path")) if play else None

    dec = D.StreamingDecoder(words_ph, blank_id=0, tau=tau,
                             dwell_cap=dwell_cap, commit_lag=commit_lag)
    n_frames = logp.shape[0]
    total_s = n_frames / FRAMES_PER_SEC

    print(f"{CLEAR}{d.get('file', '?')}  [{d.get('kind', '?')}]")
    if d.get("note"):
        print(f"  {d['note']}")
    print()
    time.sleep(0.6)

    t_wall0 = time.time()
    last_draw = 0
    stalled_note = ""

    for t in range(n_frames):
        ev = dec.step(logp[t])
        if ev["confirmed"]:
            stalled_note = ""
        if ev["stalled"] and not stalled_note:
            stalled_note = ("green edge stopped -- acoustic evidence no "
                           "longer supports the expected word")

        if t - last_draw >= redraw_every or ev["confirmed"] or dec.done:
            elapsed = t / FRAMES_PER_SEC
            screen = render(d["words_text"], dec.cursor,
                            min(dec.cursor, len(d["words_text"]) - 1),
                            ev["stalled"], elapsed, total_s, stalled_note,
                            use_color)
            print(CLEAR + screen, flush=True)
            last_draw = t

        if dec.done:
            break

        # pace to real time, scaled by --speed. Decoder work itself is
        # cheap (a handful of small-state Viterbi updates per frame); the
        # sleep is what makes this a REPLAY rather than an instant batch
        # dump, which is the whole point of a demo meant to be watched.
        target_wall = t_wall0 + (t / FRAMES_PER_SEC) / speed
        rem = target_wall - time.time()
        if rem > 0:
            time.sleep(rem)

    final = render(d["words_text"], dec.cursor,
                   min(dec.cursor, len(d["words_text"]) - 1),
                   not dec.done, n_frames / FRAMES_PER_SEC, total_s,
                   "" if dec.done else "stopped -- did not reach the end",
                   use_color)
    print(CLEAR + final)
    print()
    print(f"  confirmed {dec.cursor}/{len(words_ph)} words"
          + ("  (complete)" if dec.done else "  (stalled)"))

    if proc is not None:
        proc.wait()
    return dec.done


def prepare_items(a):
    """Same construction as decode.py's prepare(), plus the raw audio
    path and the WORD TEXT (Arabic strings, for rendering) alongside the
    phone-id sequences decode.py needs."""

    scorer = S.LiveScorer(a.ckpt, a.device)
    rt = RT.RefText(a.text)
    items = []
    for d in H.load_descriptions(a.descriptions):
        if a.file and d["file"] != a.file:
            continue
        path = Path(a.recs) / d["file"]
        if not path.exists():
            continue
        words_text = rt.words(d["surah"], d["ayah"])
        words_ph = [[scorer.vocab[p] for p in P.word_to_phones_iqra(w)
                    if p in scorer.vocab] for w in words_text]
        audio = S.load_audio(path)
        logp = D.frame_logp(scorer, audio)
        d = dict(d)
        d["words_text"] = words_text
        d["_audio_path"] = path
        items.append((d, words_ph, logp))
    return items


def main():
    ap = argparse.ArgumentParser(description="Suwayhib v6 live demo")
    ap.add_argument("--ckpt", default="runs/head_v2/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recs", default="recs")
    ap.add_argument("--descriptions", default="recs/descriptions.txt")
    ap.add_argument("--text", default="texts/quran-simple-plain.txt")
    ap.add_argument("--file", default=None,
                    help="run only this one file, e.g. Recording_4.m4a")
    ap.add_argument("--tau", type=float, default=-1.4)
    ap.add_argument("--dwell-cap", type=int, default=40)
    ap.add_argument("--commit-lag", type=int, default=8)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed multiplier; 1.0 = realtime")
    ap.add_argument("--redraw-every", type=int, default=3,
                    help="frames between screen redraws, to avoid "
                         "flicker at 50fps")
    ap.add_argument("--play", action="store_true",
                    help="also play the audio via ffplay, best-effort")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--pause-between", type=float, default=2.0)
    a = ap.parse_args()

    use_color = supports_color() and not a.no_color
    print("preparing items (encoding audio through the frozen head)...")
    items = prepare_items(a)
    if not items:
        print("no matching recordings found")
        return 1

    for i, item in enumerate(items):
        run_one(item, a.tau, a.dwell_cap, a.commit_lag, a.speed,
               a.redraw_every, a.play, use_color)
        if i < len(items) - 1:
            time.sleep(a.pause_between)
    return 0


if __name__ == "__main__":
    sys.exit(main())
