#!/usr/bin/env python3
"""
Suwayhib v6 -- grapheme-to-phoneme.

    python -m suwayhib.core.phones word "بِسْمِ"
    python -m suwayhib.core.phones test  --text ../texts/quran-simple-plain.txt
    python -m suwayhib.core.phones rate  --manifest ../manifest/manifest_clean.csv

WHAT THIS IS, AND WHY IT IS NOT A PORT OF HALABI'S CODE
---------------------------------------------------------
The IQRA 2026 challenge phoneme inventory (68 phonemes, MSA, gemination by
symbol doubling e.g. /bb/) is BASED ON Halabi & Wald 2016, implemented as
nawarhalabi/Arabic-Phonetiser and forked by the organisers as
Iqra-Eval/MSA_phonetiser.

That original phonetiser is licensed CC-BY-NC 4.0 (NonCommercial) and is
Python 2 with an external stress-annotation dependency this project does
not need (stress is a TTS/prosody concern, not MDD). Vendoring it would
also entangle Suwayhib's licence posture, which the project has otherwise
been careful about (see ATTRIBUTION.txt).

This is therefore an ORIGINAL implementation: the same linguistic rules,
informed by reading the published rule-set and by IqraEval's documented MSA
simplification (collapse the emphatic/non-emphatic VOWEL allophone split;
keep gemination as symbol-doubling), written fresh against this project's
own `reftext.py` output. It is designed to be used only as a training-data
labelling step -- never shipped in the deployed model or app -- which is
also how the organisers use their own tool.

INPUT CONTRACT
--------------
Text MUST already be reftext.py-normalised quran-simple-plain: superscript
alef (U+0670) stripped, basmala stripped from ayah 1 where applicable.
Running this on raw Tanzil text will silently mis-handle alif-maqsura
endings, per reftext.py's own measurement (11/60 -> 2/60 failures from that
one codepoint).

THE SYMBOL SET IS OURS, NOT IQRAEVAL'S VERBATIM SPELLING
----------------------------------------------------------
We do not have IqraEval's exact 68-symbol table (their HF Space did not
render in a fetchable form). Comparability to QuranMB.v2 does not require
using their exact spelling internally -- it requires a 1:1 mapping at
scoring time. IQRA_ALIAS below is an explicit, empty-by-default seam for
that mapping. Fill it in once by phonetising a handful of words with this
module and diffing against the phoneme column already present in the
Iqra_train dataset (huggingface.co/IqraEval) -- that is a direct empirical
check against their actual output, not a guess from documentation.

RULES IMPLEMENTED
------------------
  - unambiguous consonants: direct map
  - hamza (all seat forms: hamza-on-alif, hamza-on-waw, hamza-on-ya, alone,
    madda): unified to one phoneme, since the seat is an orthographic
    convention and the sound is one glottal stop in MSA recitation
  - shadda: GEMINATION by doubling the preceding consonant symbol, per the
    IqraEval convention (their /bb/ example), not a separate length feature
  - sun-letter assimilation: the definite-article lam is silent when
    followed by a shadda-marked letter (Tanzil spells this either with an
    explicit sukun on the lam, or with no diacritic on it at all before the
    shadda letter; both are handled)
  - ta marbuta: realised as a vowel + /t/ when it carries its own vowel
    diacritic (mid-utterance, connected recitation), or /h/ when it carries
    none (utterance-final, pausal form)
  - tanwin (fathatan/dammatan/kasratan): vowel + /n/. Fathatan's orthographic
    carrier alif (the bare alif conventionally written after fathatan, e.g.
    kitaaban is spelled with alif but SAID as short /an/) is detected and
    skipped rather than pronounced as a long vowel
  - madda (آ): hamza + long aa
  - long vowels: alif/waw/ya after a diacritic-less letter that itself
    carries the matching short vowel are treated as vowel LENGTHENING, not
    a separate consonant; waw/ya elsewhere are consonants
  - MSA SIMPLIFICATION (per IqraEval): the emphatic/non-emphatic allophone
    split on vowels is NOT modelled. /a/ is /a/ regardless of an adjacent
    emphatic consonant. This is a deliberate reduction of Halabi's original
    inventory, matching the challenge's stated adaptation.

WHAT IS NOT IMPLEMENTED, ON PURPOSE, FOR NOW
----------------------------------------------
  - cross-WORD hamzat-wasl elision (connected speech eliding a word-initial
    hamza after a preceding vowel). This needs sentence context this
    function does not have. Flagged, not silently wrong: words are
    phonetised in isolation.
  - the small fixed-pronunciation irregular-word table Halabi's original
    carries (a few dozen high-frequency function words with idiosyncratic
    readings, e.g. haaka-family demonstratives). Coverage testing (see
    `test`) will surface which Juz Amma words this actually matters for;
    add them by hand if the coverage run shows a problem, rather than
    porting a table blind.

SELF-TEST, WHICH IS THE POINT OF THIS FILE
--------------------------------------------
`test` runs every ayah in quran-simple-plain.txt through the G2P and
reports: every phone attested at least once, no word producing zero
phones, and the phones-per-word distribution. `rate` cross-references the
manifest to give phones-per-second per reciter, which should land at
8-16 -- a systematically wrong G2P shows up immediately as an impossible
rate, before a single frame of audio is trained on.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import reftext as RT

# --------------------------------------------------------------------------
# Buckwalter transliteration -- a standard, public scheme (LDC), not
# Halabi's invention. Arabic Unicode <-> ASCII working representation.
# --------------------------------------------------------------------------

_AR2BW = {
    '\u0628': 'b', '\u0630': '*', '\u0637': 'T', '\u0645': 'm',
    '\u062a': 't', '\u0631': 'r', '\u0638': 'Z', '\u0646': 'n',
    '\u062b': '^', '\u0632': 'z', '\u0639': 'E', '\u0647': 'h',
    '\u062c': 'j', '\u0633': 's', '\u063a': 'g', '\u062d': 'H',
    '\u0642': 'q', '\u0641': 'f', '\u062e': 'x', '\u0635': 'S',
    '\u0634': '$', '\u062f': 'd', '\u0636': 'D', '\u0643': 'k',
    '\u0623': '>', '\u0621': "'", '\u0626': '}', '\u0624': '&',
    '\u0625': '<', '\u0622': '|', '\u0627': 'A', '\u0649': 'Y',
    '\u0629': 'p', '\u064a': 'y', '\u0644': 'l', '\u0648': 'w',
    '\u064b': 'F', '\u064c': 'N', '\u064d': 'K', '\u064e': 'a',
    '\u064f': 'u', '\u0650': 'i', '\u0651': '~', '\u0652': 'o',
    # BUG FOUND BY TESTING: to_buckwalter() maps unknown characters to ''
    # (silently dropping them), so DAGGER_ALIF must be listed here or every
    # branch in word_to_phones that checks for it is dead code -- which it
    # was, on the first test pass. Identity passthrough: the working string
    # stays non-ASCII for this one codepoint, which is fine since the main
    # loop compares against the literal character, not an ASCII code.
    '\u0670': '\u0670',
}


def to_buckwalter(word):
    return ''.join(_AR2BW.get(c, '') for c in word)


# --------------------------------------------------------------------------
# phone inventory (OUR symbols -- see module docstring)
# --------------------------------------------------------------------------

HAMZA = 'q0'          # unified glottal stop, all seat forms + madda's onset
DAGGER_ALIF = '\u0670'   # superscript alef -- a LONG-VOWEL marker, not noise.
                        # reftext.py strips this for Whisper's tokenizer;
                        # that is the wrong input for this module. See
                        # load_words_for_g2p below.
CONSONANTS = {
    'b': 'b', 't': 't', '^': 'th', 'j': 'j', 'H': 'H7', 'x': 'kh',
    'd': 'd', '*': 'dh', 'r': 'r', 'z': 'z', 's': 's', '$': 'sh',
    'S': 'S9', 'D': 'D9', 'T': 'T9', 'Z': 'Z9', 'E': 'ay3', 'g': 'gh',
    'f': 'f', 'q': 'q', 'k': 'k', 'm': 'm', 'n': 'n', 'h': 'h',
    "'": HAMZA, '>': HAMZA, '<': HAMZA, '}': HAMZA, '&': HAMZA,
}
DIACRITICS = set('oauiFNK~')          # sukun, fatha, damma, kasra, tanwin x3, shadda
SHORT_VOWEL = {'a': 'a', 'u': 'u', 'i': 'i'}   # MSA: no emphatic allophone split
LONG_VOWEL = {'a': 'aa', 'u': 'uu', 'i': 'ii'}
TANWIN = {'F': ('a', 'n'), 'N': ('u', 'n'), 'K': ('i', 'n')}

# BUG FOUND BY TRAINING: this set is hand-maintained and drifted from
# what word_to_phones actually emits. 'l' is appended directly by the
# lam/sun-letter branch and never appears in CONSONANTS, so PHONE_SET
# omitted it -- and 'l' is the most frequent consonant in the corpus
# (38,890 occurrences), which caused 76% of reference items to be dropped
# as OOV during the first training attempt. 'w' and 'y' had already been
# patched in by hand for the same underlying reason, which was the warning
# sign.
#
# Kept for reference and for the `test` command's never-attested check,
# but NOT used to size the model -- see iqra_inventory(), which derives
# the inventory from actual G2P output instead of asserting it.
PHONE_SET = (set(CONSONANTS.values()) | {HAMZA} | set(SHORT_VOWEL.values())
             | set(LONG_VOWEL.values()) | {'w', 'y', 'l'})

# --------------------------------------------------------------------------
# a small number of high-frequency words are genuinely irregular and are
# not decomposable from the general rules -- this is Halabi's own reason
# for carrying a ~40-entry fixed-word table. We carry exactly ONE, because
# it is disproportionately frequent in Qur'anic text and the general rules
# cannot get it right by construction (its madd is a historical
# contraction, not a rule-governed lengthening): allaah / the divine name,
# bare or with the most common single-letter attached prepositions.
# Anything else the coverage test flags should be added here BY HAND, once
# seen -- a triage queue, not a claim of completeness.
# --------------------------------------------------------------------------

_ALLAH_PREFIX_PHONES = {
    '': [], 'b': ['b', 'i'], 'w': ['w', 'a'], 'f': ['f', 'a'],
    'l': ['l', 'i'], 'k': ['k', 'a'], 't': ['t', 'a'],
}


def _consonant_skeleton(bw):
    return re.sub(r'[oauiFNK~]', '', bw)


def _fixed_allah(bw):
    """-> phone list, or None if this word is not the divine name.

    /ʔalˈlaːh/: hamza, SHORT a, geminated l, LONG aa, h, case vowel.

    BUG FOUND BY TESTING: the first version emitted the long vowel BEFORE
    the gemination (hamza, aa, l, l, h -- "aallah") instead of after
    (hamza, a, l, l, aa, h -- "allaah"). Caught by checking the real IPA
    rather than trusting that the list "looked" plausible.

    When a prefix is attached (bi-llaahi, wa-llaahi), the word-initial
    hamza elides in connected speech -- the prefix's own vowel carries the
    syllable instead. Real Qur'anic recitation says "billaahi", not
    "bi-allaahi". Detected here by pre_ph being non-empty.
    """
    skel = _consonant_skeleton(bw)
    for pre, pre_ph in _ALLAH_PREFIX_PHONES.items():
        if skel == pre + 'Allh':
            tail = bw[-1] if bw and bw[-1] in SHORT_VOWEL else 'a'
            core = (['l', 'l', LONG_VOWEL['a'], 'h', SHORT_VOWEL[tail]]
                    if pre_ph else
                    [HAMZA, SHORT_VOWEL['a'], 'l', 'l', LONG_VOWEL['a'],
                     'h', SHORT_VOWEL[tail]])
            return pre_ph + core
    return None

# DERIVED by align_phonemes.py over 2000 Iqra_train sentences, not
# declared. Only symbols whose modal target was reached at high
# confidence are listed; anything that split ambiguously was withheld,
# because a wrong alias silently corrupts every label it touches while a
# missing one fails loudly.
#
# Confidences at derivation time (before the gemination and emphatic
# fixes, which should raise all of these):
#   ay3 -> E  99.4%    kh -> x  97.6%    gh -> g  98.9%
#   H7  -> H  97.7%    dh -> *  91.5%    q0 -> <  91.5%
#   th  -> ^  88.8%    sh -> $  80.9%
#   S9  -> S  72.9%    T9 -> T  69.2%    D9 -> D  78.9%   Z9 -> Z  78.3%
#
# The four emphatics sat lowest not because the rename is uncertain --
# S9->S is unambiguous -- but because gemination misalignment was
# scattering their counts. Re-run `align_phonemes.py derive` after this
# change; if any of them has not risen well above 90%, that is a real
# signal worth chasing rather than a threshold to lower.
IQRA_ALIAS = {
    'ay3': 'E',      # ain
    'kh': 'x',       # kha
    'gh': 'g',       # ghain
    'H7': 'H',       # ha (hah)
    'dh': '*',       # dhal
    'th': '^',       # tha
    'sh': '$',       # shin
    'q0': '<',       # hamza / glottal stop
    'S9': 'S',       # sad
    'T9': 'T',       # ta (emphatic)
    'D9': 'D',       # dad
    'Z9': 'Z',       # za (emphatic)
}


# --------------------------------------------------------------------------
# emphatic backing and gemination -- both DERIVED, not assumed.
#
# align_phonemes.py over 2000 Iqra_train sentences showed our symbols
# splitting systematically against theirs:
#     a  -> a 83.9%,  A:1429
#     i  -> i 89.7%,  I:396
#     u  -> u 88.8%,  U:330
#     aa -> aa 85.6%, AA:514
#     ii -> ii 84.4%, II:159
#     uu -> uu 87.4%, UU:86
# and every consonant showing a doubled runner-up (ll:326, dd:118, bb:140,
# nn:387, tt:103, rr:137, mm:107, ...).
#
# So IqraEval's released inventory DOES retain the Halabi emphatic vowel
# allophones, notwithstanding the challenge paper's statement that the
# distinction was collapsed for MSA. Measuring the data settled a question
# the documentation got wrong -- which is the whole reason the alias table
# was derived rather than declared.
# --------------------------------------------------------------------------

# Consonants that back an adjacent vowel. The four classical emphatics
# (mufakhkham) plus q: confirmed in the data as
#   "m i n n U T f a"     -- u before T  -> U
#   "q AA l a"            -- aa after q  -> AA
#   "f A q A dd a r a h"  -- a around q  -> A
EMPHATIC = {'S9', 'D9', 'T9', 'Z9', 'q', 'kh', 'gh'}

# r (tafkhim) is context-dependent in real tajweed -- backed next to fatha
# or damma, thin next to kasra -- and the sample shows r's vowel behaviour
# is NOT cleanly one way (r -> r 90.9% with A:14 scattered). Left OUT
# deliberately: a rule that is right 60% of the time is worse than no rule,
# because it moves errors around instead of removing them. Revisit with a
# targeted count if the residual disagreement after this change is
# concentrated on r.
EMPHATIC_MAYBE = {'r'}

BACKED = {'a': 'A', 'i': 'I', 'u': 'U',
          'aa': 'AA', 'ii': 'II', 'uu': 'UU'}


def apply_emphatic_backing(phones, window=1):
    """Back any vowel adjacent to an emphatic consonant.

    `window` is how many positions either side count as adjacent. The data
    shows backing reaching across a neighbouring consonant in places
    ("f A q A dd a r a h": the a BEFORE q is backed too), so window=1 on
    both sides is the starting point, not a claim of correctness -- re-run
    align_phonemes.py derive after changing it and keep whichever number
    lowers the disagreement rate.
    """
    out = list(phones)
    for i, p in enumerate(out):
        if p not in BACKED:
            continue
        lo, hi = max(0, i - window), min(len(out), i + window + 1)
        if any(out[j] in EMPHATIC for j in range(lo, hi) if j != i):
            out[i] = BACKED[p]
    return out


def merge_gemination(phones):
    """Collapse adjacent identical CONSONANTS into a doubled symbol.

    IqraEval writes gemination as one doubled symbol (nn, ll, bb, dd, ...)
    where we emit the consonant twice. This is a representational
    difference, not an alias: it changes sequence LENGTH, which is why it
    dominated the raw disagreement count and also corrupted the alignment
    around every geminate.

    Vowels are excluded. Two identical adjacent vowels are not gemination
    -- and where we produced them (zuwwijat -> 'z uu uu i j a t' against
    their 'z uu w i j a t') it was a separate bug in the waw rule, fixed
    there rather than papered over here.
    """
    vowels = set(SHORT_VOWEL.values()) | set(LONG_VOWEL.values()) | set(
        BACKED.values())
    out = []
    for p in phones:
        if out and out[-1] == p and p not in vowels:
            out[-1] = p + p          # n n -> nn, matching their convention
        else:
            out.append(p)
    return out


# --------------------------------------------------------------------------
# G2P
# --------------------------------------------------------------------------

class G2PWarning:
    """Collects non-fatal oddities per call so `test` can report them without
    stopping the run. A random model must fail your metric; a G2P must
    surface its own uncertainty."""

    def __init__(self):
        self.empty_words = []
        self.unmapped_chars = Counter()


def word_to_phones(ar_word, warn=None):
    """Arabic word (harakat present, reftext-normalised) -> list of phones.

    Word-isolated: no cross-word assimilation. See docstring for what this
    deliberately does not model.
    """
    bw = to_buckwalter(ar_word)
    if not bw:
        return []

    fixed = _fixed_allah(bw)
    if fixed is not None:
        return fixed

    # fathatan's orthographic carrier alif: a bare 'A' immediately after 'F'
    # is spelling, not sound (kitaaban is written with alif, said as /an/).
    bw = bw.replace('FA', 'F')

    # BUG FOUND BY TESTING (word_to_phones on الشَّمْسِ): Arabic text sources
    # do not agree on shadda/harakah combining-mark order -- some encode
    # consonant+shadda+vowel, others consonant+vowel+shadda. The gemination
    # rule below doubles whatever phone was appended LAST, so if the vowel
    # was processed before the shadda, gemination doubled the VOWEL instead
    # of the consonant ('sh','a','a' instead of 'sh','sh','a'). Canonicalise
    # the order before the main pass: shadda always immediately follows its
    # base consonant, before any vowel/tanwin diacritic on that consonant.
    bw = re.sub(r'([auiFNK])~', r'~\1', bw)

    n = len(bw)
    phones = []
    i = 0
    while i < n:
        c = bw[i]
        c1 = bw[i + 1] if i + 1 < n else ''
        c2 = bw[i + 2] if i + 2 < n else ''
        cm1 = bw[i - 1] if i > 0 else ''

        # -- shadda: geminate whatever consonant precedes it -------------
        if c == '~':
            if phones:
                phones.append(phones[-1])
            i += 1
            continue

        # -- sukun: explicit "no vowel"; nothing to emit ------------------
        if c == 'o':
            i += 1
            continue

        # -- tanwin: vowel + n, consumed as one diacritic -----------------
        if c in TANWIN:
            v, tail = TANWIN[c]
            phones.append(SHORT_VOWEL[v])
            phones.append('n')
            i += 1
            continue

        # -- madda: hamza + long aa ---------------------------------------
        if c == '|':
            phones.append(HAMZA)
            phones.append(LONG_VOWEL['a'])
            i += 1
            continue

        # -- lam: sun-letter assimilation ---------------------------------
        if c == 'l':
            # silent when it carries no vowel of its own and the NEXT
            # letter is shadda-marked (the assimilated consonant supplies
            # its own gemination via the shadda rule above)
            if c1 not in SHORT_VOWEL and c1 != '|' and c2 == '~':
                i += 1
                continue
            phones.append('l')
            i += 1
            continue

        # -- ta marbuta: vowelled -> t+vowel; bare -> h (pausal) ----------
        if c == 'p':
            if c1 in SHORT_VOWEL or c1 in TANWIN:
                phones.append('t')
            else:
                phones.append('h')
            i += 1
            continue

        # -- waw / ya: consonant vs vowel-lengthening ----------------------
        if c in ('w', 'y'):
            prev_is_matching_short = (
                (c == 'w' and cm1 == 'u') or (c == 'y' and cm1 == 'i'))
            # SHADDA'D waw/ya, refined twice against align_phonemes.py.
            #
            # First attempt lengthened unconditionally, giving 'z uu uu i
            # j a t' for zuwwijat -- the lengthening fired AND the
            # gemination rule then doubled the vowel. Second attempt swung
            # too far the other way and treated every shadda'd waw/ya as a
            # plain geminate ('z u ww i j a t'), which showed up in the
            # 2000-sentence table as yy -> y at 53.8% and ww -> w at 65.2%,
            # i.e. wrong about half the time.
            #
            # The actual rule, read off the contexts side by side:
            #     zuwwijat  damma + shadda'd WAW -> 'uu' + single 'w'
            #     qawwim    fatha + shadda'd WAW -> 'a'  + geminate 'w w'
            #     ayyuhaa   fatha + shadda'd YA  -> 'a'  + geminate 'y y'
            # When the preceding short vowel MATCHES the letter (u+waw,
            # i+ya), the first half of the gemination is absorbed into
            # lengthening the vowel and only one consonant surfaces. When
            # it does not match, the gemination is real and both surface.
            if prev_is_matching_short and c1 == '~':
                # matching vowel + shadda: lengthen, emit ONE consonant,
                # and consume the shadda here so the generic shadda branch
                # does not double what we just emitted.
                if phones and phones[-1] == SHORT_VOWEL.get(
                        'u' if c == 'w' else 'i'):
                    phones[-1] = LONG_VOWEL['u' if c == 'w' else 'i']
                phones.append('w' if c == 'w' else 'y')
                i += 2
                continue
            if prev_is_matching_short and \
                    c1 not in DIACRITICS - {'~'}:
                # lengthens the preceding short vowel rather than adding a
                # consonant: kutubu-hu style /u/ + w -> /uu/. The short
                # vowel phone was already appended when 'u'/'i' was
                # processed; replace it with the long form.
                if phones and phones[-1] in LONG_VOWEL.values():
                    pass
                elif phones and phones[-1] == SHORT_VOWEL.get(
                        'u' if c == 'w' else 'i'):
                    phones[-1] = LONG_VOWEL['u' if c == 'w' else 'i']
                else:
                    phones.append('w' if c == 'w' else 'y')
                i += 1
                continue
            # consonantal elsewhere
            phones.append('w' if c == 'w' else 'y')
            i += 1
            continue

        # -- dagger alef (superscript alef): ALWAYS a long vowel, never
        # elidable. This is the codepoint reftext.py strips for Whisper's
        # tokenizer -- see load_words_for_g2p, which deliberately does not
        # go through that strip, because this module needs the length
        # information it carries (madd in mUsaa, ar-rahmaan, etc.) --------
        if c == DAGGER_ALIF:
            if phones and phones[-1] == SHORT_VOWEL['a']:
                phones[-1] = LONG_VOWEL['a']
            elif phones and phones[-1] in LONG_VOWEL.values():
                # BUG FOUND BY TESTING (muusaa, ilaa): alif-maqsura (Y)
                # immediately before a dagger alif already performed the
                # lengthening -- the two marks together spell ONE long
                # vowel, not two. Without this branch, ilaa came out
                # ['q0','i','l','aa','aa'], five phones for a four-phone
                # word. The dagger alif here is a redundant reinforcing
                # mark and is a no-op.
                pass
            else:
                phones.append(LONG_VOWEL['a'])
            i += 1
            continue

        # -- alif / alif-maqsura: lengthens a preceding /a/ ----------------
        if c in ('A', 'Y'):
            if phones and phones[-1] == SHORT_VOWEL['a']:
                phones[-1] = LONG_VOWEL['a']
            else:
                # BUG FOUND BY TESTING: this fallback used to emit a LONG
                # vowel, which is wrong for the overwhelmingly common case
                # -- a bare word-initial alif with nothing to lengthen is
                # almost always the definite article's hamzat-wasl (elided
                # in connected speech, realised as SHORT /a/ in isolation),
                # not a long vowel. Confirmed against "al-qamari": the old
                # code produced 'aa l q a m a r i' (wrong) instead of
                # 'a l q a m a r i'.
                phones.append(SHORT_VOWEL['a'])
            i += 1
            continue

        # -- short vowels ---------------------------------------------------
        if c in SHORT_VOWEL:
            phones.append(SHORT_VOWEL[c])
            i += 1
            continue

        # -- plain consonants -----------------------------------------------
        if c in CONSONANTS:
            phones.append(CONSONANTS[c])
            i += 1
            continue

        if warn is not None:
            warn.unmapped_chars[c] += 1
        i += 1

    if not phones and warn is not None:
        warn.empty_words.append(ar_word)
    return phones


def to_iqra_label(phone):
    """Rename one symbol into IqraEval's spelling. Handles the doubled
    symbols produced by merge_gemination by aliasing the base and
    re-doubling, so H7H7 -> HH rather than falling through unmapped."""
    if phone in IQRA_ALIAS:
        return IQRA_ALIAS[phone]
    n = len(phone)
    if n % 2 == 0:
        half = phone[:n // 2]
        if phone == half + half and half in IQRA_ALIAS:
            return IQRA_ALIAS[half] * 2
    return phone


def strip_final_tanwin(phones, bw=None):
    """Drop a word-final tanwin (short vowel + n) -- PAUSAL FORM.

    IqraEval's phonetiser renders word-final tanwin in pausal form, i.e.
    omits it: khayraatun is transcribed 'x A y r aa t', nutfatin as
    'n U T f a'. We render it, so every tanwin-final word contributed a
    spurious two-symbol tail to the alignment.

    This is a genuine CONVENTION difference, not a defect on either side:
    both readings are correct Arabic, and which one is right depends on
    whether the word is uttered in pause or connected. IqraEval chose
    pausal; we follow them ONLY in the iqra-convention output, keeping
    our own representation unchanged, because for Suwayhib's actual task
    the learner IS reciting connected and the tanwin IS pronounced.

    BUG FOUND BY TESTING: an earlier version inferred tanwin from the
    PHONE SEQUENCE alone (ends in vowel + n) and so ate the final 'i n'
    of min (m i n), a word with no tanwin at all -- silently deleting two
    real phones. The phone sequence genuinely cannot tell the two apart;
    only the orthography can. So `bw` (the Buckwalter form) is required,
    and the strip fires ONLY when a tanwin diacritic (F/N/K) is present.
    """
    if bw is None or not any(c in bw for c in TANWIN):
        return phones
    vowels = set(SHORT_VOWEL.values()) | set(BACKED.values())
    if len(phones) >= 2 and phones[-1] == 'n' and phones[-2] in vowels:
        return phones[:-2]
    return phones


def strip_initial_wasl(phones, bw=None):
    """Drop a word-initial hamza+short-vowel that is an elidable wasl.

    IqraEval phonetises whole SENTENCES, so the definite article's hamzat
    wasl elides after a preceding word: al-nufuus becomes 'nn u f uu s'
    with no leading vowel. phones.py phonetises words in ISOLATION (a
    documented limitation) and so renders it.

    Only applied in the iqra-convention output, and only to the
    article-like pattern. Note this is an APPROXIMATION of real elision:
    strictly, the hamza elides only when something precedes it in the
    utterance, which word-isolated phonetisation cannot know. Sentence-
    level phonetisation would remove the guesswork -- worth doing if the
    residual disagreement after this change is concentrated here.

    Like strip_final_tanwin, this checks the ORTHOGRAPHY rather than
    guessing from the phone sequence: it requires the word to actually
    begin with alif+lam. A sequence-only version mis-fired on words that
    merely resembled the pattern once gemination had been merged.
    """
    if bw is None or not bw.startswith('Al'):
        return phones
    v = set(SHORT_VOWEL.values()) | set(BACKED.values())
    if phones and phones[0] in v:
        return phones[1:]
    return phones


def word_to_phones_iqra(ar_word, warn=None, pausal=True, elide_wasl=True):
    """Phone list in IQRAEVAL's convention: emphatic backing applied,
    gemination merged into doubled symbols, symbols renamed via IQRA_ALIAS.

    Use this for anything compared against IqraEval data or QuranMB.v2;
    use word_to_phones() for our own internal representation.

    Kept as a separate function rather than a flag on word_to_phones so
    that which convention is in play is visible at every call site. The
    two are NOT interchangeable -- they differ in sequence LENGTH -- and
    silently mixing them is exactly the failure the alias machinery exists
    to prevent.
    """
    ph = word_to_phones(ar_word, warn)
    ph = apply_emphatic_backing(ph)
    ph = merge_gemination(ph)
    bw = to_buckwalter(ar_word)
    if pausal:
        ph = strip_final_tanwin(ph, bw)
    if elide_wasl:
        ph = strip_initial_wasl(ph, bw)
    return [to_iqra_label(p) for p in ph]


def text_to_phones(words, warn=None):
    """List of Arabic words -> flat list of phones, word boundaries lost.
    For per-word sequences (needed by the graph decoder) call
    word_to_phones per word instead; this is for corpus-level stats only."""
    out = []
    for w in words:
        out.extend(word_to_phones(w, warn))
    return out


# --------------------------------------------------------------------------
# the training inventory
# --------------------------------------------------------------------------

def iqra_inventory():
    """Every symbol word_to_phones_iqra can emit. -> sorted list.

    THIS, not len(PHONE_SET), is the CTC output dimension. PHONE_SET holds
    only the 33 base symbols; the iqra convention additionally emits the
    27 consonant geminates that merge_gemination produces (nn, ll, bb, SS,
    TT, ...) and the 6 backed vowel allophones. Sizing a model from
    PHONE_SET would silently drop every geminate label.

    Vowels are excluded from gemination by construction (merge_gemination
    only doubles consonants), so the count is base + consonant geminates,
    not base squared.

    Yields 66, which is a useful cross-check: the challenge documents a
    68-phoneme inventory, and align_phonemes.py measured exactly 66
    distinct symbols in a 2000-sentence sample of the real data. Two
    independent routes to the same number. The 2 unaccounted for are
    likely symbols too rare to appear in that sample, or distinctions this
    G2P does not make -- worth a look before training, but not a blocker,
    since the model only needs to cover what the labels contain.
    """
    base = set(PHONE_SET) | set(BACKED.values())
    vowels = (set(SHORT_VOWEL.values()) | set(LONG_VOWEL.values())
              | set(BACKED.values()))
    gem = {c + c for c in base - vowels}
    return sorted({to_iqra_label(p) for p in base} |
                  {to_iqra_label(c[:len(c) // 2]) * 2 for c in gem})


def observed_inventory(text_path="texts/quran-simple-plain.txt"):
    """The inventory DERIVED from running the G2P over the whole corpus.

    iqra_inventory() asserts what should be emitted; this measures what IS
    emitted. They should agree, and when they do not, this one is right --
    a hand-maintained set drifts (it already did: 'l' was missing, and
    'w'/'y' had been patched in by hand earlier for the same reason).

    Use this to build the training vocabulary. It costs one pass over
    6,236 verses and removes an entire class of silent failure.
    """
    text, _ = load_words_for_g2p(text_path)
    seen = set()
    for t in text.values():
        for w in t.split():
            seen.update(word_to_phones_iqra(w))
    return sorted(seen)


def build_vocab(extra=()):
    """-> (symbol -> index) with 0 reserved for CTC blank.

    `extra` lets a caller fold in symbols observed in real labels that
    this G2P does not generate, so an unexpected symbol in the data widens
    the vocabulary rather than crashing training or being silently mapped
    to the wrong class.
    """
    syms = sorted(set(iqra_inventory()) | set(extra))
    return {s: i + 1 for i, s in enumerate(syms)}


def build_vocab_observed(text_path="texts/quran-simple-plain.txt",
                         extra=()):
    """Vocabulary from measured G2P output, unioned with the asserted
    inventory so nothing shrinks unexpectedly, plus any `extra` symbols a
    caller has seen in real labels (e.g. IqraEval's phoneme_ref)."""
    syms = sorted(set(observed_inventory(text_path)) |
                  set(iqra_inventory()) | set(extra))
    return {s: i + 1 for i, s in enumerate(syms)}


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_word(a):
    warn = G2PWarning()
    ph = word_to_phones(a.word, warn)
    print(f"{a.word}")
    print(f"  buckwalter : {to_buckwalter(a.word)}")
    print(f"  phones     : {' '.join(ph)}")
    if warn.unmapped_chars:
        print(f"  UNMAPPED   : {dict(warn.unmapped_chars)}")
    if not ph:
        print("  WARNING: zero phones produced")


def load_words_for_g2p(text_path):
    """Same basmala-stripping as reftext.RefText, but deliberately does NOT
    strip U+0670 (dagger alif / superscript alef). RefText strips it because
    Whisper's tokenizer needs that; this module needs the opposite, because
    the dagger alif is the ONLY marker of madd length in words like
    mUsaa/ilaa/ar-rahmaan once the full alif is absent from the spelling.
    Using RefText's output here would silently delete vowel-length
    information this G2P exists to model.

    -> {(surah, ayah): "word word word"}, same shape as RefText.text
    """

    def norm_keep_dagger(t):
        return re.sub(r"\s+", " ", t).strip()

    raw = RT._raw_load(text_path)
    if (1, 1) not in raw:
        raise SystemExit(f"{text_path}: no verse 1:1 -- wrong format?")
    basmala = norm_keep_dagger(raw[(1, 1)])

    text, stripped = {}, 0
    for (s, ay), t in raw.items():
        t = norm_keep_dagger(t)
        if ay == 1 and s not in (1, 9) and t.startswith(basmala):
            t = t[len(basmala):].strip()
            stripped += 1
        text[(s, ay)] = t
    return text, stripped


def cmd_test(a):
    text, n_basmala = load_words_for_g2p(a.text)
    print(f"basmala stripped from {n_basmala} ayahs (expect 112)")
    warn = G2PWarning()
    inventory = Counter()
    per_word_len = []
    n_words = 0

    for (s, ay), t in text.items():
        for w in t.split():
            n_words += 1
            ph = word_to_phones(w, warn)
            per_word_len.append(len(ph))
            inventory.update(ph)

    print(f"words phonetised        : {n_words}")
    print(f"distinct phones in use  : {len(inventory)}")
    print(f"empty-phone words       : {len(warn.empty_words)}")
    if warn.empty_words:
        print("  " + "  ".join(warn.empty_words[:10]))
    print(f"unmapped characters     : {dict(warn.unmapped_chars)}")

    lens = sorted(per_word_len)
    if lens:
        import statistics as st
        print(f"\nphones per word")
        print(f"  mean {st.mean(lens):.2f}  median {st.median(lens)}  "
              f"min {min(lens)}  max {max(lens)}")

    print(f"\nphone inventory ({len(inventory)} symbols), by frequency")
    for ph, n in inventory.most_common():
        print(f"  {ph:6s} {n:7d}")

    never_seen = PHONE_SET - set(inventory)
    print(f"\ndefined but NEVER attested: {sorted(never_seen) or 'none'}")
    if never_seen:
        print("  (expected only if a phone is genuinely rare in Juz Amma "
              "vocabulary and this is run on a subset -- check against the "
              "full 6236-ayah text before treating this as a bug)")

    ok = not warn.empty_words and not warn.unmapped_chars
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{'no empty words, no unmapped characters' if ok else 'see above'}")
    return 0 if ok else 1


def cmd_rate(a):
    """Phones-per-second per reciter. Gate 0 target: 8-16."""
    import csv
    text, _ = load_words_for_g2p(a.text)

    dur_ms = defaultdict(int)
    with open(a.manifest, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rec = r["reciter"]
            dur_ms[rec] += int(r["end_ms"]) - int(r["start_ms"])

    # phone counts per reciter: same text for every reciter, so compute the
    # per-ayah phone count once and multiply by each reciter's word-slot
    # coverage rather than re-running the G2P per reciter
    ayah_phones = {k: len(text_to_phones(t.split())) for k, t in text.items()}

    per_ayah_ms = defaultdict(lambda: defaultdict(int))
    with open(a.manifest, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            per_ayah_ms[r["reciter"]][(int(r["surah"]), int(r["ayah"]))] += (
                int(r["end_ms"]) - int(r["start_ms"]))

    print(f"{'reciter':<32} {'phones/sec':>10}")
    for rec, ayahs in per_ayah_ms.items():
        tot_ms = sum(ayahs.values())
        tot_ph = sum(ayah_phones.get(k, 0) for k in ayahs)
        rate = tot_ph / (tot_ms / 1000) if tot_ms else 0.0
        flag = "" if 8 <= rate <= 16 else "  <-- outside 8-16"
        print(f"  {rec:<30} {rate:10.2f}{flag}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="v6 grapheme-to-phoneme")
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("word")
    w.add_argument("word")
    w.set_defaults(func=cmd_word)

    t = sub.add_parser("test")
    t.add_argument("--text", default="../texts/quran-simple-plain.txt")
    t.set_defaults(func=cmd_test)

    r = sub.add_parser("rate")
    r.add_argument("--text", default="../texts/quran-simple-plain.txt")
    r.add_argument("--manifest", default="../manifest/manifest_clean.csv")
    r.set_defaults(func=cmd_rate)

    a = ap.parse_args()
    sys.exit(a.func(a) or 0)


if __name__ == "__main__":
    main()
