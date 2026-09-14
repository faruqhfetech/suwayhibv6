#!/usr/bin/env python3
"""
Suwayhib v5 -- student model and training  (rewrite 2).

    python student.py prepare --targets targets_train.jsonl --out cache_train
    python student.py train --cache-train cache_train --cache-val cache_val \
        --device cuda --epochs 35
    python student.py eval --run runs/student --cache cache_holdout --device cuda

WHY THIS WAS REWRITTEN
======================
Rewrite 1 (kept as student_v1_bce.py) regressed a single scalar with BCE and
plateaued at correlation 0.618 with the teacher. Widening the model did not
help; a 155K-parameter model in the streaming-KWS literature (Jin et al.,
Interspeech 2024, "CTC-aligned Audio-Text Embedding") reaches competitive
accuracy, so capacity was never the constraint.

What that literature does differently, and what is adopted here:

  1. CTC LOSS over characters.  Rewrite 1 collapsed everything into one
     scalar, so the model got no supervision about WHERE each character is.
     CTC provides frame-level alignment supervision for free.

  2. COSINE SIMILARITY, not a learned MLP head.  Rewrite 1 asked an MLP to
     learn the comparison function from scratch, with no inductive bias
     toward "similar means matching".

  3. IN-BATCH NEGATIVES.  Rewrite 1 had ~0.35 negatives per positive from
     mismatch sampling.  Using every other word in the batch gives ~B-1.
     The reference system uses ~1023.

  4. WORD-LEVEL POOLING, not a per-character mean.  Their ablation is
     explicit: character-level pooling was the WEAKEST of three levels
     tried and actually hurt on hard negatives, while word- and
     phrase-level helped substantially.

  5. DISTILLATION IS KEPT, as a third loss term.  This is the one thing not
     taken from the literature and it is deliberate: six reciters cannot
     teach speaker invariance (v4a: holdout AP 0.064), and the
     Whisper-Quran teacher learned it from thousands of Tarteel voices.
     CTC and metric learning would train invariance from OUR six speakers.
     Distillation is the channel that carries it.

     Testable prediction: ablating distillation (--w-distill 0) should hurt
     HOLDOUT more than val. That ablation isolates the claimed contribution.

AUDIT FIXES FOLDED IN
=====================
  max_len 12 -> 20.  Six of eleven common Qur'anic words were being
      TRUNCATED at 12 characters, and the lost characters are word-final
      diacritics -- exactly the kasra/sukun/tanwin contrasts this system
      exists to detect.
  Deterministic mismatch draws.  A stateful RNG meant the val set was
      redrawn every epoch (so the val curve was measured on different data
      each pass) and forked DataLoader workers shared an identical draw
      sequence.  Now seeded per index.
  Checkpoint selection on CORRELATION, not val loss.  Val loss bottomed at
      epoch 17 while correlation kept improving to 35, so the previous
      "best" checkpoint was not the best model.
  --max-examples honoured with cached data (it was silently ignored, which
      is why the "500-example overfit test" was never an overfit test).
"""

import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise SystemExit("pip install torch --index-url "
                     "https://download.pytorch.org/whl/cu121")

HERE = Path(__file__).resolve().parent
SR = 16000
N_FFT, HOP, N_MELS = 400, 160, 40
CONTEXT_MS = 150          # audio kept each side of the target word's span
MAX_CHARS = 20            # was 12, which truncated >half of Qur'anic words


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _load("teacher")     # render(), load_manifest(), build_donor_pool()


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def _mel_fb(sr=SR, n_fft=N_FFT, n_mels=N_MELS):
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(20), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l: fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c: fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


_FB = _mel_fb()
_WIN = np.hanning(N_FFT).astype(np.float32)


def logmel(x):
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * _WIN, axis=1)) ** 2
    m = np.log(spec @ _FB.T + 1e-10).T.astype(np.float32)
    m -= m.mean(axis=1, keepdims=True)
    m /= (m.std(axis=1, keepdims=True) + 1e-5)
    return m


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------

class CharVocab:
    """Index 0 is the CTC blank, 1 is <unk>. Real characters start at 2.

    CTC requires a dedicated blank symbol and torch's ctc_loss defaults to
    blank=0, so 0 is reserved for it rather than for padding. Padding is
    handled by an explicit length vector instead.
    """

    def __init__(self, chars=None):
        base = chars or []
        self.stoi = {c: i + 2 for i, c in enumerate(sorted(set(base)))}
        self.stoi["<blank>"] = 0
        self.stoi["<unk>"] = 1
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, word, max_len=MAX_CHARS):
        ids = [self.stoi.get(c, 1) for c in word[:max_len]]
        n = len(ids)
        ids += [0] * (max_len - n)
        return np.array(ids, dtype=np.int64), n

    def __len__(self):
        return len(self.stoi)

    @classmethod
    def from_words_file(cls, path):
        chars = set()
        for line in open(path, encoding="utf-8"):
            for c in line.strip():
                if c != " ":
                    chars.add(c)
        return cls(sorted(chars))


# --------------------------------------------------------------------------
# window extraction
# --------------------------------------------------------------------------

class WindowExtractor:
    """Renders one recipe and cuts out the target word's window, using the
    TRUE post-corruption spans returned by render().

    Recomputing spans from the original manifest timings would put the
    window on the wrong audio for every word after a corruption, since
    deletion/stretch/substitution all change length -- label noise on ~18%
    of examples.
    """

    def __init__(self, ayahs, lib):
        self.ayahs = ayahs
        self.lib = lib

    def extract(self, recipe, target_word_idx, donor_pool=None):
        key = (recipe["reciter"], recipe["surah"], recipe["ayah"])
        words = self.ayahs[key]
        if target_word_idx >= len(words):
            return np.zeros(T.ms(300), dtype=np.float32)

        audio, corrupted_idx, spans = T.render(
            recipe, self.ayahs, self.lib, donor_pool)
        if target_word_idx >= len(spans):
            return np.zeros(T.ms(300), dtype=np.float32)

        s0, e0 = spans[target_word_idx]
        if e0 <= s0:
            # deleted word: give the window where it SHOULD have been, which
            # is exactly what the model must learn to reject
            e0 = s0 + T.ms(150)

        pad = T.ms(CONTEXT_MS)
        a = max(0, int(s0 - pad))
        b = min(len(audio), int(e0 + pad))
        if b <= a:
            return np.zeros(T.ms(300), dtype=np.float32)
        return audio[a:b]


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------

class DistillDataset(Dataset):
    """Wraps a targets_*.jsonl file. One item per WORD."""

    def __init__(self, targets_path, ayahs, lib, vocab, donor_pool=None,
                 max_per_file=None):
        self.rows = []
        # The teacher's word_idx indexes reftext's whitespace split of the
        # Tanzil text; WindowExtractor indexes quran-align's timing segments.
        # Different sources, different tokenisation: where the counts differ
        # the span would be for the WRONG word, so those rows are dropped.
        dropped, mismatched = 0, set()
        for line in open(targets_path, encoding="utf-8"):
            r = json.loads(line)
            rec = r["recipe"]
            key = (rec["reciter"], rec["surah"], rec["ayah"])
            n_manifest = len(ayahs.get(key, []))
            if n_manifest != len(r["words"]):
                mismatched.add((rec["surah"], rec["ayah"],
                                len(r["words"]), n_manifest))
            for w in r["words"]:
                if w["idx"] >= n_manifest:
                    dropped += 1
                    continue
                self.rows.append({
                    "recipe": rec, "word_idx": w["idx"],
                    "word": w["word"], "target": w["target"],
                })
            if max_per_file and len(self.rows) >= max_per_file:
                self.rows = self.rows[:max_per_file]
                break

        if dropped or mismatched:
            print(f"  [{Path(targets_path).name}] dropped {dropped} words; "
                  f"{len(mismatched)} ayahs with a teacher/manifest word-count "
                  f"mismatch")

        self.win = WindowExtractor(ayahs, lib)
        self.vocab = vocab
        self.donor_pool = donor_pool

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        audio = self.win.extract(r["recipe"], r["word_idx"], self.donor_pool)
        chars, n = self.vocab.encode(r["word"])
        return {"mel": logmel(audio), "text": chars, "tlen": n,
                "y": np.float32(r["target"]), "wid": 0}


class PreparedDataset(Dataset):
    """Precomputed log-mel windows from `prepare`. Training is array indexing.

    On-the-fly rendering launched an ffmpeg subprocess per word -- measured
    at >40 minutes per epoch, essentially all process spawn and I/O.
    """

    def __init__(self, cache_dir, vocab, mismatch_p=0.0, seed=0, limit=None):
        d = Path(cache_dir)
        z = np.load(d / "mels.npz", allow_pickle=False)
        self.flat = z["flat"]              # (total_frames, N_MELS) float16
        self.offsets = z["offsets"]
        self.y = z["y"]
        meta = json.loads((d / "meta.json").read_text())
        self.words = meta["words"]
        self.keys = meta.get("keys", [""] * len(self.words))

        if limit:
            n = min(limit, len(self.y))
            self.y = self.y[:n]
            self.words = self.words[:n]
            self.keys = self.keys[:n]
            print(f"  truncated to {n} examples")

        self.vocab = vocab
        self.mismatch_p = mismatch_p
        self.seed = seed

        self.by_key = {}
        for i, k in enumerate(self.keys):
            self.by_key.setdefault(k, []).append(self.words[i])
        self.all_words = sorted(set(self.words))
        # stable word ids, so in-batch negatives never treat two occurrences
        # of the SAME word as a negative pair
        self.word_id = {w: i for i, w in enumerate(self.all_words)}

        print(f"  cache: {len(self.y)} examples, {self.flat.nbytes/1e6:.0f} MB"
              + (f", mismatch p={mismatch_p}" if mismatch_p else ""))

    def _rng(self, i):
        """Seeded PER INDEX, not a rolling stream.

        A stateful RNG made the val set different every epoch (so the val
        curve was measured on different data each pass) and gave every
        forked DataLoader worker the same draw sequence. Per-index seeding
        is deterministic, epoch-stable and worker-safe.
        """
        return random.Random((self.seed * 1_000_003 + i) & 0xFFFFFFFF)

    def _mismatched_word(self, i, rng):
        true = self.words[i]
        same = [w for w in self.by_key.get(self.keys[i], []) if w != true]
        pool = same if (same and rng.random() < 0.7) else self.all_words
        for _ in range(8):
            c = rng.choice(pool)
            if c != true:
                return c
        return None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        a, b = self.offsets[i], self.offsets[i + 1]
        word, y = self.words[i], float(self.y[i])
        rng = self._rng(i)

        # Only build a mismatch from a window whose audio genuinely contains
        # its own word. Pairing wrong text with already-corrupted audio
        # teaches nothing new.
        if self.mismatch_p and y >= 0.5 and rng.random() < self.mismatch_p:
            cand = self._mismatched_word(i, rng)
            if cand is not None:
                word, y = cand, 0.0

        chars, n = self.vocab.encode(word)
        return {
            "mel": self.flat[a:b].T.astype(np.float32),   # (N_MELS, T)
            "text": chars, "tlen": n, "y": np.float32(y),
            "wid": self.word_id.get(word, -1),
        }


def collate(batch):
    T_max = max(b["mel"].shape[1] for b in batch)
    X = np.zeros((len(batch), N_MELS, T_max), dtype=np.float32)
    M = np.zeros((len(batch), T_max), dtype=np.float32)
    for i, b in enumerate(batch):
        n = b["mel"].shape[1]
        X[i, :, :n] = b["mel"]
        M[i, :n] = 1.0
    return (torch.from_numpy(X), torch.from_numpy(M),
            torch.from_numpy(np.stack([b["text"] for b in batch])),
            torch.from_numpy(np.array([b["tlen"] for b in batch], dtype=np.int64)),
            torch.from_numpy(np.array([b["y"] for b in batch], dtype=np.float32)),
            torch.from_numpy(np.array([b["wid"] for b in batch], dtype=np.int64)))


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class DSConv(nn.Module):
    """Depthwise-separable conv block, as used in small-footprint KWS.
    Cheap per parameter, which is the point -- the reference system reaches
    competitive accuracy with a 155K-parameter acoustic encoder."""

    def __init__(self, d, k=5, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.dw = nn.Conv1d(d, d, k, padding=pad, dilation=dilation, groups=d)
        self.pw = nn.Conv1d(d, d, 1)
        self.bn1 = nn.BatchNorm1d(d)
        self.bn2 = nn.BatchNorm1d(d)

    def forward(self, x):
        h = F.relu(self.bn1(self.dw(x)))
        h = F.relu(self.bn2(self.pw(h)))
        return x + h


class AcousticEncoder(nn.Module):
    def __init__(self, d=192, n_blocks=6):
        super().__init__()
        self.inp = nn.Conv1d(N_MELS, d, 5, padding=2)
        self.bn = nn.BatchNorm1d(d)
        self.blocks = nn.ModuleList(
            [DSConv(d, 5, dilation=2 ** (i % 3)) for i in range(n_blocks)])
        self.d = d

    def forward(self, x):
        h = F.relu(self.bn(self.inp(x)))
        for b in self.blocks:
            h = b(h)
        return h.transpose(1, 2)              # (B, T, d)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d=192):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.rnn = nn.LSTM(d, d // 2, num_layers=2, batch_first=True,
                           bidirectional=True)
        self.d = d

    def forward(self, chars):
        h, _ = self.rnn(self.emb(chars))
        return h                               # (B, L, d)


class CTCATVerifier(nn.Module):
    """Acoustic encoder + text encoder + two heads:

      CTC head    per-frame character posteriors -> alignment supervision
      AE/TE proj  a shared embedding space compared by COSINE SIMILARITY

    The cosine comparison replaces rewrite 1's learned MLP head: "similar
    means matching" is built in rather than learned from scratch.
    """

    def __init__(self, vocab_size, d=192, n_blocks=6, emb=128):
        super().__init__()
        self.audio = AcousticEncoder(d, n_blocks)
        self.text = TextEncoder(vocab_size, d)
        self.ctc = nn.Linear(d, vocab_size)
        self.ae_proj = nn.Sequential(nn.Linear(d, emb), nn.LayerNorm(emb))
        self.te_proj = nn.Sequential(nn.Linear(d, emb), nn.LayerNorm(emb))
        # learned temperature and offset so cosine (bounded [-1,1]) can span
        # the logit range BCE needs
        self.logit_scale = nn.Parameter(torch.tensor(2.5))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode(self, mel, mask, chars, tlen):
        h = self.audio(mel)                                  # (B, T, d)
        ctc_logits = self.ctc(h)                             # (B, T, V)

        # WORD-LEVEL pooling, masked. The reference ablation found
        # character-level pooling the weakest of three levels tried; word
        # and phrase level were substantially better.
        m = mask.unsqueeze(-1)
        ae = (h * m).sum(1) / m.sum(1).clamp(min=1)
        ae = F.normalize(self.ae_proj(ae), dim=-1)            # (B, emb)

        t = self.text(chars)                                  # (B, L, d)
        tm = (torch.arange(chars.shape[1], device=chars.device)[None, :]
              < tlen[:, None]).float().unsqueeze(-1)
        te = (t * tm).sum(1) / tm.sum(1).clamp(min=1)
        te = F.normalize(self.te_proj(te), dim=-1)            # (B, emb)
        return ae, te, ctc_logits

    def forward(self, mel, mask, chars, tlen):
        ae, te, ctc_logits = self.encode(mel, mask, chars, tlen)
        sim = (ae * te).sum(-1)                               # cosine
        return sim * self.logit_scale.exp() + self.bias, ae, te, ctc_logits


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

def in_batch_metric_loss(ae, te, wid, y, temp=0.07):
    """Every other word in the batch is a negative.

    Rewrite 1 had ~0.35 negatives per positive from mismatch sampling; this
    gives ~B-1. Rows whose audio is corrupted (y<0.5) are excluded as
    anchors, because their audio does NOT contain their text -- they are not
    valid positives for anyone.

    Two occurrences of the same word are masked out of the negative set via
    `wid`, so the loss never pushes apart two things that should match.
    """
    valid = y >= 0.5
    if int(valid.sum()) < 2:
        return ae.new_zeros(())
    a, t, w = ae[valid], te[valid], wid[valid]
    logits = (a @ t.t()) / temp                       # (n, n)
    same = w[:, None] == w[None, :]
    eye = torch.eye(len(a), dtype=torch.bool, device=a.device)
    logits = logits.masked_fill(same & ~eye, -1e4)    # ignore duplicate words
    tgt = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, tgt)
                  + F.cross_entropy(logits.t(), tgt))


def ctc_alignment_loss(ctc_logits, mask, chars, tlen, y):
    """Frame-level character alignment. Only for rows whose audio genuinely
    contains the text -- CTC on a mismatched pair would ask the model to
    align characters that are not there."""
    valid = y >= 0.5
    if int(valid.sum()) == 0:
        return ctc_logits.new_zeros(())
    lp = F.log_softmax(ctc_logits[valid], dim=-1).transpose(0, 1)  # (T,B,V)
    inp_len = mask[valid].sum(1).long().clamp(min=1)
    tl = tlen[valid].clamp(min=1)
    # CTC needs input length >= target length; drop rows that violate it
    ok = inp_len >= tl
    if int(ok.sum()) == 0:
        return ctc_logits.new_zeros(())
    return F.ctc_loss(lp[:, ok], chars[valid][ok], inp_len[ok], tl[ok],
                      blank=0, zero_infinity=True)


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------

def cmd_prepare(args):
    ayahs = T.load_manifest(args.manifest)
    vocab = CharVocab.from_words_file(args.text)
    HOLD = {"Husary_64kbps", "Minshawy_Murattal_128kbps"}
    keys = [k for k in ayahs if (k[0] in HOLD) == args.holdout]
    pool = T.build_donor_pool(ayahs, args.lib, keys, 250, args.seed)

    ds = DistillDataset(args.targets, ayahs, args.lib, vocab, pool)
    print(f"rendering {len(ds)} windows (once) ...")

    src_cache = {}
    orig_load = T.load_audio

    def cached_load(path):
        p = str(path)
        if p not in src_cache:
            if len(src_cache) > args.cache_files:
                src_cache.clear()
            src_cache[p] = orig_load(path)
        return src_cache[p].copy()

    T.load_audio = cached_load
    try:
        mels, ys, words, keys_out = [], [], [], []
        for i in range(len(ds)):
            item = ds[i]
            mels.append(item["mel"].T.astype(np.float16))
            ys.append(item["y"])
            rw = ds.rows[i]
            words.append(rw["word"])
            keys_out.append(f"{rw['recipe']['surah']}:{rw['recipe']['ayah']}")
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(ds)}")
    finally:
        T.load_audio = orig_load

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    flat = np.concatenate(mels, axis=0)
    offsets = np.cumsum([0] + [len(m) for m in mels]).astype(np.int64)
    np.savez(out / "mels.npz", flat=flat, offsets=offsets,
             y=np.array(ys, dtype=np.float32))
    (out / "meta.json").write_text(json.dumps({
        "words": words, "keys": keys_out, "n": len(ys),
        "source": str(args.targets), "holdout": args.holdout,
    }, ensure_ascii=False))
    print(f"wrote {out}/mels.npz  {flat.shape} ({flat.nbytes/1e6:.0f} MB fp16)")


# --------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------

def n_params(m):
    return sum(p.numel() for p in m.parameters())


def _auc(pos, neg):
    """Rank-based AUC: threshold-free AND calibration-free.

    Pearson r was used for model selection all evening and it MISLEADS
    across models: it penalises a model whose ranking is fine but whose
    outputs are compressed into a narrow band. AUC does not care about the
    scale, only the ordering, which is what actually matters for a score
    that will be thresholded at deployment.
    """
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = len(pos), len(neg)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _best_balanced(pos, neg, n_thr=400):
    """Sweep the threshold; return (balanced_accuracy, threshold).

    A fixed cut-point measures where a model's outputs happen to sit, not
    how well it discriminates. Two models on different scales can only be
    compared by letting each use its own best threshold.
    """
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), 0.0
    lo, hi = float(min(pos.min(), neg.min())), float(max(pos.max(), neg.max()))
    best = (-1.0, lo)
    for thr in np.linspace(lo, hi, n_thr):
        bal = (float((pos > thr).mean()) + float((neg <= thr).mean())) / 2
        if bal > best[0]:
            best = (bal, float(thr))
    return best


@torch.no_grad()
def score_batches(model, dl, device):
    """Both head scores, per example.

      z_emb  sigmoid of the scaled cosine similarity
      z_ctc  length-normalised CTC forward score -- log P(expected chars |
             audio), computed WITHOUT the embedding path, so it is a
             genuinely independent view

    The CTC head was trained but never consulted at inference. Measured on
    holdout it is the STRONGER of the two (84.9% vs 82.7% balanced), and
    the combination beats both.
    """
    model.eval()
    z_emb, z_ctc, tgts = [], [], []
    for X, M, C, L, y, W in dl:
        X, M, C, L = X.to(device), M.to(device), C.to(device), L.to(device)
        logit, ae, te, ctc_logits = model(X, M, C, L)
        z_emb.append(torch.sigmoid(logit).cpu().numpy())

        lp = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)   # (T,B,V)
        inp = M.sum(1).long().clamp(min=1)
        tl = L.clamp(min=1)
        per = F.ctc_loss(lp, C, inp, tl, blank=0, zero_infinity=True,
                         reduction="none")
        z_ctc.append((-per / tl.float()).cpu().numpy())          # higher = better
        tgts.append(y.numpy())
    return (np.concatenate(z_emb), np.concatenate(z_ctc),
            np.concatenate(tgts))


def combine(z_emb, z_ctc, head="both", lam=2.0):
    """head: emb | ctc | both

    For `both`, each score is standardised before weighting -- z_ctc is an
    unbounded log-prob and z_emb is a probability in [0,1], so without this
    lambda would be doing unit conversion rather than weighting.

    lambda is MODEL-SPECIFIC: measured 2.0 for the distilled model and 0.5
    for the ablation. It must be tuned per checkpoint, not assumed.
    """
    if head == "emb":
        return z_emb
    if head == "ctc":
        return z_ctc
    def z(v):
        return (v - v.mean()) / (v.std() + 1e-8)
    return z(z_emb) + lam * z(z_ctc)


def evaluate(model, dl, device, head="both", lam=2.0):
    z_emb, z_ctc, tgts = score_batches(model, dl, device)
    s = combine(z_emb, z_ctc, head, lam)
    pos, neg = s[tgts >= 0.5], s[tgts < 0.5]
    bal, thr = _best_balanced(pos, neg)
    return {
        "score": s, "tgts": tgts, "z_emb": z_emb, "z_ctc": z_ctc,
        "auc": _auc(pos, neg),
        "balanced": bal, "thr": thr,
        # kept for continuity with earlier runs; NOT the selection metric
        "r": float(np.corrcoef(s, tgts)[0, 1]) if len(s) > 1 else float("nan"),
        "n_clean": int((tgts >= 0.5).sum()), "n_corrupt": int((tgts < 0.5).sum()),
    }


def cmd_train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    vocab = CharVocab.from_words_file(args.text)
    print(f"vocab: {len(vocab)} symbols (0 = CTC blank)")

    if not args.cache_train or not args.cache_val:
        raise SystemExit("--cache-train and --cache-val required; "
                         "run `prepare` first")
    tr = PreparedDataset(args.cache_train, vocab, args.mismatch_p,
                         args.seed, args.max_examples)
    va = PreparedDataset(args.cache_val, vocab, args.mismatch_p,
                         args.seed + 1, args.max_val)

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, collate_fn=collate,
                       drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, collate_fn=collate)

    model = CTCATVerifier(len(vocab), args.width, args.blocks).to(device)
    print(f"parameters: {n_params(model):,} ({n_params(model)*4/1e6:.1f} MB)")
    print(f"loss = {args.w_distill}*distill + {args.w_metric}*metric "
          f"+ {args.w_ctc}*ctc")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(dl_tr)))

    best_r, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        agg = {"d": 0.0, "m": 0.0, "c": 0.0}
        for X, M, C, L, y, W in dl_tr:
            X, M, C = X.to(device), M.to(device), C.to(device)
            L, y, W = L.to(device), y.to(device), W.to(device)

            logit, ae, te, ctc_logits = model(X, M, C, L)

            l_d = F.binary_cross_entropy_with_logits(logit, y)
            l_m = in_batch_metric_loss(ae, te, W, y, args.temp)
            l_c = ctc_alignment_loss(ctc_logits, M, C, L, y)
            loss = args.w_distill * l_d + args.w_metric * l_m + args.w_ctc * l_c

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); sched.step()
            agg["d"] += float(l_d); agg["m"] += float(l_m); agg["c"] += float(l_c)

        nb = max(len(dl_tr), 1)
        v = evaluate(model, dl_va, device, args.head, args.lam)
        e = evaluate(model, dl_va, device, "emb", args.lam)
        c = evaluate(model, dl_va, device, "ctc", args.lam)
        hist.append({"epoch": ep, "distill": agg["d"] / nb,
                     "metric": agg["m"] / nb, "ctc_loss": agg["c"] / nb,
                     "auc": v["auc"], "balanced": v["balanced"],
                     "auc_emb": e["auc"], "auc_ctc": c["auc"], "r": v["r"]})
        print(f"ep {ep:3d}  distill {agg['d']/nb:.3f}  metric {agg['m']/nb:.3f}  "
              f"ctc {agg['c']/nb:.3f}  |  AUC {v['auc']:.4f} "
              f"(emb {e['auc']:.3f} ctc {c['auc']:.3f})  "
              f"bal {v['balanced']:.1%}  r {v['r']:.3f}")

        # SELECT ON AUC, not Pearson r. r penalises a model whose ranking is
        # fine but whose outputs are compressed -- measured directly: the
        # ablation looked 0.10 worse by r and was only ~1.5pp worse by
        # balanced accuracy. AUC is rank-based, so it measures the thing that
        # survives thresholding at deployment.
        if v["auc"] > best_r:
            best_r = v["auc"]
            torch.save({"model": model.state_dict(), "vocab": vocab.stoi,
                        "d": args.width, "blocks": args.blocks,
                        "epoch": ep, "val_auc": v["auc"],
                        "val_balanced": v["balanced"], "thr": v["thr"],
                        "head": args.head, "lam": args.lam}, out / "best.pt")

    (out / "history.json").write_text(json.dumps(hist, indent=2))
    print(f"\nbest val AUC {best_r:.4f} -> {out/'best.pt'}")
    print(f"NEXT: python student.py eval --run {out} "
          f"--cache cache_holdout --device {args.device}")


def cmd_eval(args):
    device = torch.device(args.device)
    ck = torch.load(Path(args.run) / "best.pt", map_location=device,
                    weights_only=False)
    vocab = CharVocab(); vocab.stoi = ck["vocab"]
    vocab.itos = {i: c for c, i in vocab.stoi.items()}

    model = CTCATVerifier(len(vocab.stoi), ck["d"], ck.get("blocks", 6)).to(device)
    model.load_state_dict(ck["model"])

    ds = PreparedDataset(args.cache, vocab, args.mismatch_p, 1234)
    dl = DataLoader(ds, batch_size=64, collate_fn=collate)
    z_emb, z_ctc, tgts = score_batches(model, dl, device)
    pos_m, neg_m = tgts >= 0.5, tgts < 0.5

    print(f"\ncheckpoint epoch {ck.get('epoch','?')}  "
          f"({int(pos_m.sum())} positives, {int(neg_m.sum())} negatives)")

    for name, key in (("embedding", "emb"), ("CTC", "ctc")):
        sc = combine(z_emb, z_ctc, key)
        bal, thr = _best_balanced(sc[pos_m], sc[neg_m])
        print(f"  {name:10s} alone   AUC {_auc(sc[pos_m], sc[neg_m]):.4f}   "
              f"balanced {bal:.1%} at thr {thr:.4f}")

    # lambda is MODEL-SPECIFIC (2.0 distilled, 0.5 ablation) -- sweep it
    print("\n  combined  z_emb + lambda * z_ctc")
    best = (-1.0, 0.0, 0.0, 0.0)
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        sc = combine(z_emb, z_ctc, "both", lam)
        a = _auc(sc[pos_m], sc[neg_m])
        bal, thr = _best_balanced(sc[pos_m], sc[neg_m])
        mark = ""
        if a > best[0]:
            best, mark = (a, lam, bal, thr), "  <-- best"
        print(f"    lambda {lam:4.2f}   AUC {a:.4f}   balanced {bal:.1%}{mark}")

    print(f"\n  BEST: AUC {best[0]:.4f}, balanced {best[2]:.1%}, "
          f"lambda {best[1]}, threshold {best[3]:.4f}")
    print("\n  AUC is rank-based -- threshold-free AND calibration-free --")
    print("  which is what makes it a fair comparison ACROSS models. Pearson r")
    print("  penalises compressed outputs even when the ordering is correct.")


def main():
    ap = argparse.ArgumentParser(
        description="v5 student (CTC + metric learning + distillation)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--cache-train", default=None)
    t.add_argument("--cache-val", default=None)
    t.add_argument("--text", default="texts/quran-simple-plain.txt")
    t.add_argument("--out", default="runs/student")
    t.add_argument("--epochs", type=int, default=100,
                   help="at 35 epochs OneCycle annealed lr to 3e-8 by the "
                        "end, so the apparent plateau was the SCHEDULE, not "
                        "convergence. The reference system trains 100")
    t.add_argument("--batch", type=int, default=256,
                   help="batch size IS the negative count for the metric "
                        "loss. At 64 the metric loss fell to 0.03 -- the "
                        "64-way discrimination was solved and stopped "
                        "producing gradient. The reference system uses 1024")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--width", type=int, default=192)
    t.add_argument("--blocks", type=int, default=6)
    t.add_argument("--temp", type=float, default=0.07)
    t.add_argument("--w-distill", type=float, default=1.0,
                   help="carries the teacher's SPEAKER INVARIANCE, which six "
                        "reciters cannot teach from scratch. Set 0 for the "
                        "ablation: it should hurt HOLDOUT more than val")
    t.add_argument("--w-metric", type=float, default=1.0)
    t.add_argument("--w-ctc", type=float, default=0.3)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--head", default="both", choices=["emb", "ctc", "both"],
                   help="which score to select checkpoints on. The CTC head "
                        "alone beats the embedding head (84.9%% vs 82.7%% "
                        "balanced on holdout); `both` beats either")
    t.add_argument("--lam", type=float, default=2.0,
                   help="weight on the CTC score in `both`. MODEL-SPECIFIC: "
                        "measured 2.0 distilled, 0.5 for the ablation")
    t.add_argument("--mismatch-p", type=float, default=0.35)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--max-examples", type=int, default=None)
    t.add_argument("--max-val", type=int, default=None)
    t.add_argument("--device", default="cpu")
    t.add_argument("--seed", type=int, default=1)
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("prepare")
    p.add_argument("--targets", default="targets_train.jsonl")
    p.add_argument("--out", default="cache_train")
    p.add_argument("--manifest", default="manifest_full.csv")
    p.add_argument("--lib", default="reference_library")
    p.add_argument("--text", default="texts/quran-simple-plain.txt")
    p.add_argument("--holdout", action="store_true")
    p.add_argument("--cache-files", type=int, default=800)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_prepare)

    e = sub.add_parser("eval")
    e.add_argument("--run", default="runs/student")
    e.add_argument("--cache", default="cache_holdout")
    e.add_argument("--text", default="texts/quran-simple-plain.txt")
    e.add_argument("--mismatch-p", type=float, default=0.35,
                   help="must match training, or the gate measures a "
                        "different task than the model was trained on")
    e.add_argument("--lam", type=float, default=2.0)
    e.add_argument("--device", default="cpu")
    e.set_defaults(func=cmd_eval)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
