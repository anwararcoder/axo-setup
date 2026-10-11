"""
StanzaOrder: restoring the printed line order of scrambled German stanzas (1600-1900) for unseen poets

Task
  Each row of test.csv is one stanza of 4-10 lines in scrambled order (labels L1..Ln). Output the labels in
  printed order. Score per stanza: the share of its n-1 consecutive printed pairs that are also consecutive
  (same direction) in the prediction; the final score is the mean over stanzas.

Approach
  1. Line features, computed from the line's own text: the punctuation it ends with (period prints close a
     sense unit with a full stop or colon and continue it with a comma or virgule), its first word (a
     vocabulary of the 100 most frequent line-initial words of train.csv, counted in code), the syllable
     count, masculine/feminine ending, and how many other lines of the same stanza it rhymes with.
     Rhyme is detected with hand-written rules on word endings (no dictionary): the spelling variants of the
     prints are normalised (long s, e-above umlauts, ey/ei, th/t, doubled letters, final devoicing) and two
     lines rhyme when their last words agree from the last stressed vowel on.
  2. Pair features for every ordered pair of lines (a, b) of the same stanza: rhyme, shared final letters,
     same ending type, syllable difference.
  3. A small neural network, created with random initialisation and trained in this file on train.csv only,
     turns the features into three distributions per stanza: which line is printed first, and for every line
     which line follows it or whether the stanza ends after it. Loss: cross-entropy against the printed order.
  4. Rhyme and metre are symmetric: they say which lines belong next to each other but not in which direction,
     and a reversed run of lines scores zero. The direction comes from syntax and sense, so a character
     language model (a causal transformer, 4 layers, width 256, randomly initialised and trained here on the
     printed stanzas of train.csv) scores for every stanza: log P(line j | stanza start), log P(line b | line a
     printed just before it) and log P(stanza end | line a). Summed along an order, these are the language
     model's log-likelihood of the stanza read in that order (with one line of context).
  5. Decoding: the order with the largest total score (the network's log-probabilities of first line, each
     successor and end, plus 0.3 times the language model's log-likelihoods) is found exactly by dynamic
     programming over subsets of lines (Held-Karp; at most 2^10 x 10 states per stanza).
  Each test stanza is featurised, scored and decoded on its own: no information passes between test rows.

Deterministic runtime plan (keep this section; the platform checks for it)
  * One fixed device (cuda:0). No device selection, no fallback, no multi-GPU path, no data-loader workers,
    no mixed precision, no retry logic.
  * Fixed seeds for Python, NumPy and torch, set again before the model is created; a fixed number of epochs
    and a fixed one-cycle schedule; no early stopping; the clock is read only to print progress.
  * Deterministic backends: use_deterministic_algorithms(True), cuDNN deterministic, benchmarking off, TF32 off,
    fixed cuBLAS workspace.
  * No result depends on the iteration order of a set or on Python's hash seed: vocabularies are sorted by
    count and then alphabetically, and every loop runs over sorted or positional indices.
  * Attention, character embeddings and every other layer are plain matrix products, so no kernel or
    convolution algorithm is chosen at run time.

Rules followed
  * From scratch: no pretrained weights or embeddings, no external text, no rhyme or pronunciation
    dictionary. Every learned parameter is created from RANDOM INITIALISATION in this file and trained here on
    the provided train.csv. The rhyme rules are hand-written string rules on word endings.
  * test.csv is read only after training and is used only for the single prediction pass; each test stanza
    is predicted from its own lines alone.
"""
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # assigned (never setdefault), before torch is imported

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# One fixed plan: the amount of work, the device and the paths never depend on the clock, a score, the GPU or
# CPU count, CUDA availability or files found on disk.
CONFIG = dict(
    seeds=[0],
    epochs=30,
    batch=256,          # stanzas per optimisation step
    lr=2e-3,
    weight_decay=0.05,
    hidden=128,
    first_vocab=100,    # most frequent line-initial words of train.csv used as one-hot features
    lm_layers=4,        # character language model (causal transformer) trained from scratch on train.csv
    lm_dim=256,
    lm_heads=4,
    lm_dropout=0.1,
    lm_epochs=12,
    lm_batch=128,       # windows per optimisation step
    lm_lr=1e-3,
    lm_max_line=100,    # characters kept per line (99 % of lines are shorter than 65)
    lm_weight=0.3,      # weight of the language-model log-likelihoods next to the network's log-probabilities
)
DEVICE = "cuda:0"
MAX_LINES = 10          # the description: stanzas have 4-10 lines


def make_deterministic(seed):
    """Fixed seeds and deterministic kernels. Called before the model is built."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def progress_logger():
    """Returns on_epoch(fraction_done): prints elapsed time and the estimated time left."""
    start, last = time.time(), [0.0]

    def on_epoch(fraction):
        elapsed = time.time() - start
        if last[0] == 0.0 or fraction - last[0] >= 0.1 or fraction >= 1.0:
            last[0] = fraction
            left = elapsed / fraction * (1.0 - fraction)
            print(f"{fraction:6.1%} done, {elapsed / 60:.1f} min elapsed, about {left / 60:.1f} min left", flush=True)
    return on_epoch


# ----------------------------------------------------------------------------
# Text rules (hand-written; no dictionary, no learned component)
# ----------------------------------------------------------------------------
E_ABOVE = {"a": "ä", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü"}
VOWEL_GROUP = re.compile(r"[aeiou123]+")
# Ordered spelling rules. Diphthongs become one placeholder symbol each (1 = ei/ai, 2 = eu, 3 = au) so that the
# period spellings ey/ei/ay/ai and eu/äu compare equal; ie is a long i; th, dt, tz, ck, ph are single sounds.
SPELLING_RULES = [("eu", "2"), ("ei", "1"), ("ey", "1"), ("ai", "1"), ("ay", "1"), ("au", "3"), ("ie", "i"),
                  ("y", "i"), ("th", "t"), ("dt", "t"), ("tz", "z"), ("ck", "k"), ("ph", "f")]
END_PUNCT = [",", ";", ":", ".", "!", "?", "/", "-", "letter", "other"]
CLOSERS = "\"'»«„“”‘’)]"
DASHES = "-–—"


def clean(text):
    """Long s -> s, a/o/u with a small e above -> umlaut, Unicode NFC."""
    text = text.replace("ſ", "s")
    text = re.sub(r"([aouAOU])ͤ", lambda m: E_ABOVE[m.group(1)], text)
    return unicodedata.normalize("NFC", text)


def phonetic(word):
    """A rough sound spelling of one word, so that rhymes spelled differently in the prints compare equal."""
    w = word.lower().replace("ß", "ss").replace("ä", "e").replace("ö", "e").replace("ü", "i")
    w = "".join(ch for ch in unicodedata.normalize("NFD", w) if unicodedata.category(ch) != "Mn")
    w = re.sub(r"[^a-z]", "", w)
    for a, b in SPELLING_RULES:
        w = w.replace(a, b)
    w = re.sub(r"([aeiou123])h", r"\1", w)   # lengthening h after a vowel is silent
    w = re.sub(r"(.)\1+", r"\1", w)            # doubled letters
    if w[-1:] in ("d", "b", "g"):              # final devoicing: Tod/Not, Hand/bekannt
        w = w[:-1] + {"d": "t", "b": "p", "g": "k"}[w[-1]]
    return w


def rhyme_parts(word):
    """(rhyme key, feminine ending flag, syllables) of a line's last word.
    A feminine ending is an unstressed final e-syllable (Sonne, Lieder); the rhyme then starts one vowel earlier."""
    p = phonetic(word)
    groups = list(VOWEL_GROUP.finditer(p))
    if not groups:
        return p, False, 0
    last = groups[-1]
    feminine = len(groups) >= 2 and last.group() == "e" and len(p) - last.end() <= 2
    key = p[groups[-2].start():] if feminine else p[last.start():]
    return key, feminine, len(groups)


def syllables(text):
    return sum(len(VOWEL_GROUP.findall(phonetic(w))) for w in re.findall(r"[^\W\d_]+", text))


def words_of(text):
    return re.findall(r"[^\W\d_]+", text)


def end_punct(text):
    """Class of the line's final punctuation, ignoring closing quotes and brackets."""
    s = text.rstrip()
    closed = False
    while s and s[-1] in CLOSERS:
        s, closed = s[:-1].rstrip(), True
    ch = s[-1:] if s else ""
    if ch in DASHES:
        cls = "-"
    elif ch in END_PUNCT:
        cls = ch
    elif ch.isalpha():
        cls = "letter"
    else:
        cls = "other"
    return cls, closed


def first_word(text):
    w = words_of(text)
    return w[0].lower() if w else ""


# ----------------------------------------------------------------------------
# Features. One stanza -> line features [10, Fl], pair features [10, 10, Fp], padded to MAX_LINES.
# Every feature of a stanza is computed from that stanza's own lines and the training vocabulary only.
# ----------------------------------------------------------------------------
def build_vocab(train_table, size):
    """The most frequent line-initial words of the training stanzas (count, then alphabetical)."""
    counts = Counter()
    for lines_json in train_table["lines"]:
        for line in json.loads(lines_json):
            counts[first_word(clean(line))] += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {w: i for i, (w, _) in enumerate(ranked[:size])}


def stanza_features(lines, vocab):
    n = len(lines)
    lines = [clean(x) for x in lines]
    last_words = [(words_of(x) or [""])[-1] for x in lines]
    parts = [rhyme_parts(w) for w in last_words]
    keys = [p[0] for p in parts]
    fem = [p[1] for p in parts]
    syl = [syllables(x) for x in lines]
    med_syl = float(np.median(syl))
    v = len(vocab)
    n_line = len(END_PUNCT) + 2 + 4 + 7 + 9 + (v + 1)
    L = np.zeros((MAX_LINES, n_line), np.float32)
    for i, text in enumerate(lines):
        cls, closed = end_punct(text)
        rhymes = sum(1 for k in range(n) if k != i and keys[k] and keys[k] == keys[i])
        f = [0.0] * n_line
        f[END_PUNCT.index(cls)] = 1.0
        o = len(END_PUNCT)
        f[o] = float(closed)
        f[o + 1] = float(text.lstrip()[:1] in CLOSERS + DASHES)          # line opens with a quote or dash
        o += 2
        f[o + min(rhymes, 3)] = 1.0                                       # rhyme partners in this stanza
        o += 4
        f[o + n - 4] = 1.0                                                # stanza size 4..10
        o += 7
        f[o] = syl[i] / 12.0
        f[o + 1] = (syl[i] - med_syl) / 4.0
        f[o + 2] = float(fem[i])
        f[o + 3] = len(text) / 60.0
        f[o + 4] = text.count(",") / 3.0
        f[o + 5] = float("?" in text)
        f[o + 6] = float("!" in text)
        f[o + 7] = text.count("/") / 2.0                                  # virgule inside the line
        f[o + 8] = float(len(words_of(text)) <= 3)
        o += 9
        f[o + vocab.get(first_word(text), v)] = 1.0
        L[i] = f
    n_pair = 8
    P = np.zeros((MAX_LINES, MAX_LINES, n_pair), np.float32)
    ph = [phonetic(w) for w in last_words]
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            P[a, b] = [
                float(bool(keys[a]) and keys[a] == keys[b]),
                float(len(ph[a]) >= 2 and ph[a][-2:] == ph[b][-2:]),
                float(len(ph[a]) >= 3 and ph[a][-3:] == ph[b][-3:]),
                float(last_words[a].lower() == last_words[b].lower()),
                float(fem[a] == fem[b]),
                (syl[b] - syl[a]) / 5.0,
                abs(syl[b] - syl[a]) / 5.0,
                float(syl[a] == syl[b]),
            ]
    return L, P


def featurise(table, vocab):
    """Row-aligned arrays for a table of stanzas."""
    S = len(table)
    Ls, Ps = [], []
    n = table["n_lines"].to_numpy().astype(np.int64)
    for lines_json in table["lines"]:
        L, P = stanza_features(json.loads(lines_json), vocab)
        Ls.append(L)
        Ps.append(P)
    mask = np.arange(MAX_LINES)[None, :] < n[:, None]
    return dict(L=np.stack(Ls), P=np.stack(Ps), mask=mask, n=n, ids=table["example_id"].tolist(), size=S)


def targets(table):
    """first[s] = index of the printed first line; succ[s, i] = index of the line after line i, or MAX_LINES
    (the END column) for the last printed line, or -100 for padding."""
    S = len(table)
    first = np.zeros(S, np.int64)
    succ = np.full((S, MAX_LINES), -100, np.int64)
    for s, order in enumerate(table["order"]):
        idx = [int(lab[1:]) - 1 for lab in order.split()]
        first[s] = idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            succ[s, a] = b
        succ[s, idx[-1]] = MAX_LINES
    return first, succ


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class PairOrderNet(nn.Module):
    """Shared line encoder; a pair scorer for "b follows a"; an end score and a start score per line."""

    def __init__(self, n_line, n_pair, hidden):
        super().__init__()
        self.line = nn.Sequential(nn.Linear(n_line, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pair = nn.Sequential(nn.Linear(2 * hidden + n_pair, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.end = nn.Linear(hidden, 1)
        self.start = nn.Linear(hidden, 1)

    def forward(self, L, P, mask):
        h = self.line(L)                                              # [B, N, H]
        B, N, H = h.shape
        hi = h[:, :, None, :].expand(B, N, N, H)
        hj = h[:, None, :, :].expand(B, N, N, H)
        z = self.pair(torch.cat([hi, hj, P], dim=-1)).squeeze(-1)     # [B, N, N]
        eye = torch.eye(N, dtype=torch.bool, device=L.device)[None]
        valid = mask[:, :, None] & mask[:, None, :] & ~eye
        z = z.masked_fill(~valid, -1e9)
        succ = torch.cat([z, self.end(h)], dim=-1)                    # column N = "stanza ends here"
        start = self.start(h).squeeze(-1).masked_fill(~mask, -1e9)
        return F.log_softmax(succ, dim=-1), F.log_softmax(start, dim=-1)


def fit(train, first, succ, cfg, seed, on_epoch=None):
    """Train one network on the given stanzas for exactly cfg["epochs"] passes."""
    make_deterministic(seed)
    L = torch.tensor(train["L"], device=DEVICE)
    P = torch.tensor(train["P"], device=DEVICE)
    M = torch.tensor(train["mask"], device=DEVICE)
    T1 = torch.tensor(first, device=DEVICE)
    T2 = torch.tensor(succ, device=DEVICE)
    model = PairOrderNet(L.shape[-1], P.shape[-1], cfg["hidden"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    S = L.shape[0]
    steps_per_epoch = (S + cfg["batch"] - 1) // cfg["batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch,
                                                pct_start=0.1)
    gen = torch.Generator().manual_seed(seed)                         # CPU generator: shuffling is reproducible
    model.train()
    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(S, generator=gen).to(DEVICE)
        for k in range(steps_per_epoch):
            b = perm[k * cfg["batch"]:(k + 1) * cfg["batch"]]
            logp_succ, logp_start = model(L[b], P[b], M[b])
            loss = F.nll_loss(logp_succ.reshape(-1, MAX_LINES + 1), T2[b].reshape(-1), ignore_index=-100) \
                + F.nll_loss(logp_start, T1[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if on_epoch:
            on_epoch((epoch + 1) / cfg["epochs"])
    model.eval()
    return model


@torch.no_grad()
def log_probs(models, data, batch=512):
    """Average over models of the start and successor log-probabilities, as float64 NumPy arrays."""
    S = data["size"]
    succ_all = np.zeros((S, MAX_LINES, MAX_LINES + 1))
    start_all = np.zeros((S, MAX_LINES))
    for model in models:
        for k in range(0, S, batch):
            L = torch.tensor(data["L"][k:k + batch], device=DEVICE)
            P = torch.tensor(data["P"][k:k + batch], device=DEVICE)
            M = torch.tensor(data["mask"][k:k + batch], device=DEVICE)
            ls, lt = model(L, P, M)
            succ_all[k:k + batch] += ls.double().cpu().numpy() / len(models)
            start_all[k:k + batch] += lt.double().cpu().numpy() / len(models)
    return start_all, succ_all


def decode(start, succ, n):
    """Exact best order of one group of stanzas with the same size n (Held-Karp over subsets).
    start [B, n], trans [B, n, n] (diagonal excluded), end [B, n]; returns [B, n] line indices."""
    B = start.shape[0]
    trans = succ[:, :n, :n].copy()
    trans[:, np.arange(n), np.arange(n)] = -np.inf
    end = succ[:, :n, MAX_LINES]
    full = 1 << n
    dp = np.full((B, full, n), -np.inf)
    back = np.zeros((B, full, n), np.int64)
    rows = np.arange(B)
    for j in range(n):
        dp[:, 1 << j, j] = start[:, j]
    for mask in range(1, full):                       # increasing masks: every subset is finished before use
        for j in range(n):
            if not (mask >> j) & 1 or mask == (1 << j):
                continue
            prev = mask ^ (1 << j)
            cand = dp[:, prev, :] + trans[:, :, j]
            k = cand.argmax(axis=1)
            dp[:, mask, j] = cand[rows, k]
            back[:, mask, j] = k
    last = (dp[:, full - 1, :] + end).argmax(axis=1)
    orders = np.zeros((B, n), np.int64)
    mask = np.full(B, full - 1)
    cur = last.copy()
    for pos in range(n - 1, -1, -1):
        orders[:, pos] = cur
        prev_cur = back[rows, mask, cur]
        mask = mask ^ (1 << cur)
        cur = prev_cur
    return orders


# ----------------------------------------------------------------------------
# Character language model, trained from scratch on the training stanzas in printed order. It supplies the
# direction evidence that rhyme and metre cannot: how naturally line b continues line a (syntax, sense).
# ----------------------------------------------------------------------------
BOS, MID, EOS, NL = "\x02", "\x01", "\x03", "\n"   # stanza start, window start inside a stanza, stanza end, line end


class Block(nn.Module):
    """Pre-norm transformer block with causal self-attention written as plain matrix products."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.heads)
        att = att.masked_fill(causal[:T, :T], float("-inf")).softmax(-1)
        x = x + self.drop(self.proj((att @ v).transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class CharLM(nn.Module):
    """Causal character transformer; the character embedding is a one-hot matrix product and is tied to the
    output layer."""

    def __init__(self, n_chars, cfg, max_len):
        super().__init__()
        d = cfg["lm_dim"]
        self.n_chars = n_chars
        self.emb = nn.Parameter(torch.randn(n_chars, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, cfg["lm_heads"], cfg["lm_dropout"]) for _ in range(cfg["lm_layers"])])
        self.ln = nn.LayerNorm(d)
        self.register_buffer("causal", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1))

    def forward(self, ids):
        x = F.one_hot(ids, self.n_chars).float() @ self.emb + self.pos[: ids.shape[1]]
        for block in self.blocks:
            x = block(x, self.causal)
        return self.ln(x) @ self.emb.T


def lm_chars(train_table, max_line):
    """Character inventory of the training lines (count, then code point); 0 = padding, 1 = unseen."""
    counts = Counter()
    for lines_json in train_table["lines"]:
        for line in json.loads(lines_json):
            counts.update(clean(line)[:max_line])
    chars = {c: i + 2 for i, (c, _) in enumerate(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))}
    for c in (BOS, MID, EOS, NL):
        if c not in chars:
            chars[c] = len(chars) + 2
    return chars


def encode_windows(windows, chars, max_len):
    """(context, target) strings -> padded ids and a mask of the target positions."""
    ids = np.zeros((len(windows), max_len), np.int64)
    target = np.zeros((len(windows), max_len), bool)
    for i, (ctx, tgt) in enumerate(windows):
        seq = [chars.get(c, 1) for c in ctx + tgt][:max_len]
        ids[i, :len(seq)] = seq
        target[i, len(ctx):len(seq)] = True
    return ids, target


def train_windows(train_table, max_line):
    """The training text in the three shapes that are scored later: the first line after the stanza start,
    every printed line pair (line b after line a), and the stanza end after the last line."""
    windows = []
    for lines_json, order in zip(train_table["lines"], train_table["order"]):
        lines = [clean(x)[:max_line] for x in json.loads(lines_json)]
        o = [int(lab[1:]) - 1 for lab in order.split()]
        windows.append((BOS, lines[o[0]] + NL))
        windows += [(MID + lines[a] + NL, lines[b] + NL) for a, b in zip(o[:-1], o[1:])]
        windows.append((MID + lines[o[-1]] + NL, EOS))
    return windows


def fit_lm(train_table, cfg, seed, on_epoch=None):
    """Train the character LM for exactly cfg["lm_epochs"] passes over the training windows."""
    make_deterministic(seed)
    chars = lm_chars(train_table, cfg["lm_max_line"])
    max_len = 2 * cfg["lm_max_line"] + 4
    ids, _ = encode_windows(train_windows(train_table, cfg["lm_max_line"]), chars, max_len)
    n_chars = len(chars) + 2
    model = CharLM(n_chars, cfg, max_len).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lm_lr"], weight_decay=0.05)
    N = len(ids)
    steps_per_epoch = (N + cfg["lm_batch"] - 1) // cfg["lm_batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lm_lr"],
                                                total_steps=cfg["lm_epochs"] * steps_per_epoch, pct_start=0.05)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for epoch in range(cfg["lm_epochs"]):
        perm = torch.randperm(N, generator=gen).numpy()
        for k in range(steps_per_epoch):
            rows = np.sort(perm[k * cfg["lm_batch"]:(k + 1) * cfg["lm_batch"]])
            T = int((ids[rows] > 0).sum(1).max())                     # longest window of the batch
            x = torch.tensor(ids[rows, :T], device=DEVICE)
            keep = (x[:, 1:] > 0).float()                             # every real character after the first
            logits = model(x[:, :-1])
            nll = F.cross_entropy(logits.reshape(-1, n_chars), x[:, 1:].reshape(-1), reduction="none")
            loss = (nll.reshape(keep.shape) * keep).sum() / keep.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if on_epoch:
            on_epoch((epoch + 1) / cfg["lm_epochs"])
    model.eval()
    return dict(model=model, chars=chars, max_len=max_len)


@torch.no_grad()
def lm_scores(lm, table, cfg, batch=256):
    """For every stanza on its own: log P(line j | stanza start), log P(line b | line a before it) and
    log P(stanza end | line a), in the layout of the network's start / successor arrays."""
    S = len(table)
    start = np.full((S, MAX_LINES), -1e9)
    succ = np.full((S, MAX_LINES, MAX_LINES + 1), -1e9)
    windows, where = [], []
    for s, lines_json in enumerate(table["lines"]):
        lines = [clean(x)[:cfg["lm_max_line"]] for x in json.loads(lines_json)]
        n = len(lines)
        for a in range(n):
            windows.append((BOS, lines[a] + NL))
            where.append((s, -1, a))
            windows.append((MID + lines[a] + NL, EOS))
            where.append((s, a, MAX_LINES))
            for b in range(n):
                if b != a:
                    windows.append((MID + lines[a] + NL, lines[b] + NL))
                    where.append((s, a, b))
    ids, target = encode_windows(windows, lm["chars"], lm["max_len"])
    values = np.zeros(len(windows))
    for k in range(0, len(windows), batch):
        rows = slice(k, k + batch)
        T = int((ids[rows] > 0).sum(1).max())
        x = torch.tensor(ids[rows, :T], device=DEVICE)
        logp = F.log_softmax(lm["model"](x[:, :-1]), dim=-1).gather(-1, x[:, 1:, None]).squeeze(-1)
        mask = torch.tensor(target[rows, 1:T], device=DEVICE)
        values[rows] = (logp * mask).sum(1).double().cpu().numpy()
    for (s, a, b), v in zip(where, values):
        if a == -1:
            start[s, b] = v
        else:
            succ[s, a, b] = v
    return start, succ


def predict_orders(models, data, lm_start, lm_succ, lm_weight):
    """Best printed order for every stanza: the network's log-probabilities plus lm_weight times the language
    model's log-likelihoods, maximised exactly. Stanzas are grouped by size only to vectorise the dynamic
    programme; each row of a group is solved separately, so one stanza's result never depends on another."""
    start, succ = log_probs(models, data)
    start = start + lm_weight * lm_start
    succ = succ + lm_weight * lm_succ
    orders = [None] * data["size"]
    for n in np.unique(data["n"]).tolist():
        rows = np.flatnonzero(data["n"] == n)
        best = decode(start[rows, :n], succ[rows], n)
        for r, o in zip(rows, best):
            orders[r] = o.tolist()
    return orders


def to_submission(test, orders, sample):
    """The submission table in exactly the sample's columns and id order."""
    by_id = {i: " ".join(f"L{k + 1}" for k in o) for i, o in zip(test["ids"], orders)}
    submission = sample.copy()
    submission["order"] = [by_id[i] for i in submission["example_id"]]
    return submission


def main():
    # Required input/output pattern. Keep the names public_dir / submission_out everywhere in the file.
    public_dir = (
        Path(sys.argv[1])
        if len(sys.argv) >= 3 and not sys.argv[1].startswith("-f")
        else Path("/kaggle/working/extracted_dataset")
    )
    submission_out = (
        Path(sys.argv[2])
        if len(sys.argv) >= 3 and not sys.argv[1].startswith("-f")
        else Path("/kaggle/working/submission.csv")
    )
    t_start = time.time()
    minutes = lambda: (time.time() - t_start) / 60
    cfg = CONFIG
    print(f"device {DEVICE}, config {cfg}", flush=True)

    sample = pd.read_csv(public_dir / "sample_submission.csv")
    # Safety net: the sample (each stanza's labels in label order) is a valid submission from the first seconds.
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(submission_out, index=False, encoding="utf-8")

    # Training uses train.csv only; the first-word vocabulary is counted on training lines.
    train_table = pd.read_csv(public_dir / "train.csv")
    vocab = build_vocab(train_table, cfg["first_vocab"])
    train = featurise(train_table, vocab)
    first, succ = targets(train_table)
    print(f"train featurised: {train['size']} stanzas at {minutes():.1f} min", flush=True)

    models = []
    for seed in cfg["seeds"]:
        models.append(fit(train, first, succ, cfg, seed, on_epoch=progress_logger()))
        print(f"seed {seed} trained at {minutes():.1f} min", flush=True)
    lm = fit_lm(train_table, cfg, cfg["seeds"][0], on_epoch=progress_logger())
    print(f"language model trained at {minutes():.1f} min", flush=True)

    # Final stage: test.csv is read only now; every stanza is featurised and decoded on its own.
    test_table = pd.read_csv(public_dir / "test.csv")
    test = featurise(test_table, vocab)
    lm_start, lm_succ = lm_scores(lm, test_table, cfg)
    orders = predict_orders(models, test, lm_start, lm_succ, cfg["lm_weight"])
    submission = to_submission(test, orders, sample)
    assert list(submission.columns) == list(sample.columns)
    assert submission.iloc[:, 0].tolist() == sample.iloc[:, 0].tolist()
    for o, n in zip(orders, test["n"]):
        assert sorted(o) == list(range(n))
    submission.to_csv(submission_out, index=False, encoding="utf-8")
    print(f"total {minutes():.1f} min -> {submission_out}", flush=True)


if __name__ == "__main__":
    main()
