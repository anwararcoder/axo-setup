"""
StanzaOrder: restoring the printed line order of scrambled German stanzas (1600-1900) for unseen poets

Task
  Each row of test.csv is one stanza of 4-10 lines in scrambled order (labels L1..Ln). Output the labels in
  printed order. Score per stanza: the share of its n-1 consecutive printed pairs that are also consecutive
  (same direction) in the prediction; the final score is the mean over stanzas.

Approach
  1. Hand-made line features, computed from the line's own text: the punctuation it ends with (period prints
     close a sense unit with a full stop or colon and continue it with a comma or virgule), its first word (a
     vocabulary of the 200 most frequent line-initial words of train.csv, counted in code), the syllable
     count, masculine/feminine ending, and how many other lines of the same stanza it rhymes with.
     Rhyme is detected with hand-written rules on word endings (no dictionary): the spelling variants of the
     prints are normalised (long s, e-above umlauts, qu, ey/ei, th/t, doubled letters, final devoicing) and two
     lines rhyme when their last words agree from the last stressed vowel on.
  2. Pair features for every ordered pair of lines (a, b) of the same stanza: rhyme, shared final letters,
     same ending type, syllable difference.
  3. Character encoders let the network read the text itself: the first 12 characters of the line (how it
     joins the line before), the last 10 characters of its last word (the rhyme and the ending), and the
     whole line through two width-5 convolutions with max pooling (written as sliding windows + linear layers).
     The character table is the inventory of train.csv.
  4. A set transformer (2 layers, no positional input) lets every line's representation see the other lines
     of its stanza, so the scores can follow the stanza's rhyme scheme, ending pattern and punctuation.
  5. Heads, all trained jointly with cross-entropy against the printed orders of train.csv: which line is
     printed first; for every line which line follows it or whether the stanza ends after it; the printed
     position of every line; and for every pair whether line a is printed before line b.
     The network is created with random initialisation and trained in this file on train.csv only.
  6. Decoding: the order with the largest total score (first line + each successor + end, plus the position
     and before/after log-probabilities with fixed weights) is found exactly by dynamic programming over
     subsets of lines (Held-Karp; at most 2^10 x 10 states per stanza). Every term depends only on the set of
     lines already placed, so the search is exact.
  Each test stanza is featurised and decoded on its own: no information passes between test rows.

Deterministic runtime plan (keep this section; the platform checks for it)
  * One fixed device (cuda:0). No device selection, no fallback, no multi-GPU path, no data-loader workers,
    no mixed precision, no retry logic.
  * Fixed seeds for Python, NumPy and torch, set again before the model is created; a fixed number of epochs
    and a fixed one-cycle schedule; no early stopping; the clock is read only to print progress.
  * Deterministic backends: use_deterministic_algorithms(True), cuDNN deterministic, benchmarking off, TF32 off,
    fixed cuBLAS workspace.
  * No result depends on the iteration order of a set or on Python's hash seed: vocabularies are sorted by
    count and then alphabetically, and every loop runs over sorted or positional indices.

Rules followed
  * From scratch: no pretrained weights or embeddings, no external text, no rhyme or pronunciation
    dictionary. Every learned parameter is created from RANDOM INITIALISATION in this file and trained here on
    the provided train.csv. The rhyme rules are hand-written string rules on word endings.
  * test.csv is read only after training and is used only for the single prediction pass; each test stanza
    is predicted from its own lines alone.
"""
import json
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
    epochs=40,
    batch=128,          # stanzas per optimisation step
    lr=2e-3,
    weight_decay=0.05,
    hidden=128,
    first_vocab=200,    # most frequent line-initial words of train.csv used as one-hot features
    chars=1,            # 1: character encoders (line head, last word, whole line) are part of the line encoder
    context_layers=2,   # transformer layers over the lines of one stanza (no positional input: a set encoder)
    dropout=0.1,
    aux_pos=1.0,        # weight of the auxiliary "absolute position of each line" loss
    aux_before=1.0,     # weight of the auxiliary "line a is printed before line b" loss
    w_pos=1.0,          # decoding weight of the position log-probabilities
    w_before=0.25,      # decoding weight of the before/after log-probabilities
)
DEVICE = "cuda:0"
MAX_LINES = 10          # the description: stanzas have 4-10 lines
HEAD_CHARS = 12         # first characters of a line (left-aligned)
TAIL_CHARS = 10         # last characters of the line's last word (right-aligned): the rhyme
LINE_CHARS = 80         # whole line, truncated, for the convolutional encoder


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
SPELLING_RULES = [("qu", "kw"), ("eu", "2"), ("ei", "1"), ("ey", "1"), ("ai", "1"), ("ay", "1"), ("au", "3"), ("ie", "i"),
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
    """Vocabularies counted on the training stanzas only: the most frequent line-initial words (count, then
    alphabetical) and the character inventory (count, then code point; 0 = padding, 1 = unseen character)."""
    words, chars = Counter(), Counter()
    for lines_json in train_table["lines"]:
        for line in json.loads(lines_json):
            text = clean(line)
            words[first_word(text)] += 1
            chars.update(text.lower())
    ranked = sorted(words.items(), key=lambda kv: (-kv[1], kv[0]))
    char_ranked = sorted(chars.items(), key=lambda kv: (-kv[1], kv[0]))
    return dict(first={w: i for i, (w, _) in enumerate(ranked[:size])},
                chars={c: i + 2 for i, (c, _) in enumerate(char_ranked)})


def char_ids(text, chars, length, align_right):
    ids = [chars.get(c, 1) for c in text]
    ids = ids[-length:] if align_right else ids[:length]
    pad = [0] * (length - len(ids))
    return pad + ids if align_right else ids + pad


def stanza_features(lines, vocab):
    first_vocab, chars = vocab["first"], vocab["chars"]
    n = len(lines)
    lines = [clean(x) for x in lines]
    last_words = [(words_of(x) or [""])[-1] for x in lines]
    parts = [rhyme_parts(w) for w in last_words]
    keys = [p[0] for p in parts]
    fem = [p[1] for p in parts]
    syl = [syllables(x) for x in lines]
    med_syl = float(np.median(syl))
    v = len(first_vocab)
    n_line = len(END_PUNCT) + 2 + 4 + 7 + 9 + (v + 1)
    L = np.zeros((MAX_LINES, n_line), np.float32)
    C_head = np.zeros((MAX_LINES, HEAD_CHARS), np.int64)
    C_tail = np.zeros((MAX_LINES, TAIL_CHARS), np.int64)
    C_line = np.zeros((MAX_LINES, LINE_CHARS), np.int64)
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
        f[o + first_vocab.get(first_word(text), v)] = 1.0
        L[i] = f
        low = text.lower()
        C_head[i] = char_ids(low.lstrip(), chars, HEAD_CHARS, align_right=False)
        C_tail[i] = char_ids(last_words[i].lower(), chars, TAIL_CHARS, align_right=True)
        C_line[i] = char_ids(low, chars, LINE_CHARS, align_right=False)
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
    return L, P, C_head, C_tail, C_line


def featurise(table, vocab):
    """Row-aligned arrays for a table of stanzas; each stanza is featurised from its own lines only."""
    out = {k: [] for k in ("L", "P", "head", "tail", "line")}
    for lines_json in table["lines"]:
        for k, arr in zip(("L", "P", "head", "tail", "line"), stanza_features(json.loads(lines_json), vocab)):
            out[k].append(arr)
    n = table["n_lines"].to_numpy().astype(np.int64)
    data = {k: np.stack(v) for k, v in out.items()}
    data.update(mask=np.arange(MAX_LINES)[None, :] < n[:, None], n=n, ids=table["example_id"].tolist(),
                size=len(table), n_chars=len(vocab["chars"]) + 2)
    return data


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


def positions(first, succ):
    """pos[s, i] = printed position of line i (from the first line and the successor links), -100 for padding."""
    S = len(first)
    pos = np.full((S, MAX_LINES), -100, np.int64)
    for s in range(S):
        cur, k = int(first[s]), 0
        while cur != MAX_LINES:
            pos[s, cur] = k
            cur, k = int(succ[s, cur]), k + 1
    return pos


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class CharTable(nn.Module):
    """Character embedding as a one-hot matrix product (a plain matmul, deterministic in both directions)."""

    def __init__(self, n_chars, dim):
        super().__init__()
        self.n_chars = n_chars
        self.weight = nn.Parameter(torch.randn(n_chars, dim) * 0.1)

    def forward(self, ids):
        return F.one_hot(ids, self.n_chars).float() @ self.weight


def windows(x, width=5):
    """[M, T, D] -> [M, T, width * D]: the zero-padded window of `width` positions centred on each position."""
    pad = width // 2
    xp = F.pad(x, (0, 0, pad, pad))
    return torch.cat([xp[:, k:k + x.shape[1], :] for k in range(width)], dim=-1)


class StanzaOrderNet(nn.Module):
    """Line encoder (hand-made features + character encoders) -> set transformer over the stanza's lines ->
    pair scorer ("b follows a", "a is printed before b") and per-line heads (first, last, position)."""

    def __init__(self, n_line, n_pair, n_chars, cfg):
        super().__init__()
        H = cfg["hidden"]
        self.use_chars = bool(cfg["chars"])
        d_in = n_line
        if self.use_chars:
            self.emb = CharTable(n_chars, 24)
            self.head_enc = nn.Linear(HEAD_CHARS * 24, 64)
            self.tail_enc = nn.Linear(TAIL_CHARS * 24, 64)
            # Two width-5 convolutions written as sliding windows + linear layers: plain matrix products, so no
            # convolution algorithm is chosen at run time.
            self.conv1 = nn.Linear(5 * 24, 96)
            self.conv2 = nn.Linear(5 * 96, 96)
            d_in += 64 + 64 + 96
        self.line = nn.Sequential(nn.Linear(d_in, H), nn.ReLU(), nn.Dropout(cfg["dropout"]), nn.Linear(H, H))
        self.context = None
        if cfg["context_layers"]:
            layer = nn.TransformerEncoderLayer(H, 4, 2 * H, cfg["dropout"], batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, cfg["context_layers"], enable_nested_tensor=False)
        self.pair = nn.Sequential(nn.Linear(3 * H + n_pair, H), nn.ReLU(), nn.Dropout(cfg["dropout"]),
                                  nn.Linear(H, 2))
        self.end = nn.Linear(H, 1)
        self.start = nn.Linear(H, 1)
        self.pos = nn.Linear(H, MAX_LINES)

    def encode(self, b):
        x = b["L"]
        if self.use_chars:
            B, N = x.shape[:2]
            head = self.emb(b["head"]).reshape(B, N, -1)
            tail = self.emb(b["tail"]).reshape(B, N, -1)
            line = self.emb(b["line"]).reshape(B * N, LINE_CHARS, -1)
            conv = F.relu(self.conv1(windows(line)))
            conv = F.relu(self.conv2(windows(conv)))                       # [B*N, LINE_CHARS, 96]
            valid = (b["line"].reshape(B * N, LINE_CHARS, 1) > 0)
            conv = conv.masked_fill(~valid, 0.0).amax(dim=1).reshape(B, N, -1)
            x = torch.cat([x, F.relu(self.head_enc(head)), F.relu(self.tail_enc(tail)), conv], dim=-1)
        h = self.line(x)
        if self.context is not None:
            h = self.context(h, src_key_padding_mask=~b["mask"])
        return h

    def forward(self, b):
        h = self.encode(b)                                                 # [B, N, H]
        mask = b["mask"]
        B, N, H = h.shape
        hi = h[:, :, None, :].expand(B, N, N, H)
        hj = h[:, None, :, :].expand(B, N, N, H)
        out = self.pair(torch.cat([hi, hj, hi * hj, b["P"]], dim=-1))     # [B, N, N, 2]
        eye = torch.eye(N, dtype=torch.bool, device=h.device)[None]
        valid = mask[:, :, None] & mask[:, None, :] & ~eye
        z = out[..., 0].masked_fill(~valid, -1e9)
        succ = torch.cat([z, self.end(h)], dim=-1)                         # column N = "stanza ends here"
        start = self.start(h).squeeze(-1).masked_fill(~mask, -1e9)
        n = mask.sum(1)
        pos_valid = torch.arange(N, device=h.device)[None, None, :] < n[:, None, None]
        pos = self.pos(h).masked_fill(~pos_valid, -1e9)
        return dict(succ=F.log_softmax(succ, dim=-1), start=F.log_softmax(start, dim=-1),
                    pos=F.log_softmax(pos, dim=-1), before=out[..., 1], valid=valid)


TENSOR_KEYS = ("L", "P", "head", "tail", "line", "mask")


def to_device(data, rows):
    return {k: torch.tensor(data[k][rows], device=DEVICE) for k in TENSOR_KEYS}


def fit(train, first, succ, cfg, seed, on_epoch=None):
    """Train one network on the given stanzas for exactly cfg["epochs"] passes."""
    make_deterministic(seed)
    pos = positions(first, succ)
    T_first = torch.tensor(first, device=DEVICE)
    T_succ = torch.tensor(succ, device=DEVICE)
    T_pos = torch.tensor(pos, device=DEVICE)
    model = StanzaOrderNet(train["L"].shape[-1], train["P"].shape[-1], train["n_chars"], cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    S = train["size"]
    steps_per_epoch = (S + cfg["batch"] - 1) // cfg["batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch,
                                                pct_start=0.1)
    gen = torch.Generator().manual_seed(seed)                         # CPU generator: shuffling is reproducible
    model.train()
    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(S, generator=gen).numpy()
        for k in range(steps_per_epoch):
            rows = np.sort(perm[k * cfg["batch"]:(k + 1) * cfg["batch"]])
            b = to_device(train, rows)
            r = torch.tensor(rows, device=DEVICE)
            out = model(b)
            loss = F.nll_loss(out["succ"].reshape(-1, MAX_LINES + 1), T_succ[r].reshape(-1), ignore_index=-100) \
                + F.nll_loss(out["start"], T_first[r])
            p = T_pos[r]
            loss = loss + cfg["aux_pos"] * F.nll_loss(out["pos"].reshape(-1, MAX_LINES), p.reshape(-1),
                                                      ignore_index=-100)
            before_target = (p[:, :, None] < p[:, None, :]).float()
            bce = F.binary_cross_entropy_with_logits(out["before"], before_target, reduction="none")
            loss = loss + cfg["aux_before"] * (bce * out["valid"]).sum() / out["valid"].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if on_epoch:
            on_epoch((epoch + 1) / cfg["epochs"])
    model.eval()
    return model


@torch.no_grad()
def log_probs(models, data, batch=256):
    """Average over models of the start, successor, position and before/after log-probabilities (float64)."""
    S = data["size"]
    acc = dict(start=np.zeros((S, MAX_LINES)), succ=np.zeros((S, MAX_LINES, MAX_LINES + 1)),
               pos=np.zeros((S, MAX_LINES, MAX_LINES)), before=np.zeros((S, MAX_LINES, MAX_LINES)))
    for model in models:
        for k in range(0, S, batch):
            rows = np.arange(k, min(k + batch, S))
            out = model(to_device(data, rows))
            out["before"] = F.logsigmoid(out["before"])
            for key in acc:
                acc[key][rows] += out[key].double().cpu().numpy() / len(models)
    return acc


def decode(lp, n, w_pos, w_before):
    """Exact best order of one group of stanzas with the same size n (Held-Karp over subsets of lines).
    Score of an order o: start[o0] + sum succ[o_k, o_k+1] + end[o_last] + w_pos * sum pos[o_k, k]
    + w_before * sum_{k<l} before[o_k, o_l]. Every term only needs the set of lines already placed, so the
    dynamic programme is exact. Returns [B, n] line indices."""
    start, succ, pos, before = lp["start"][:, :n], lp["succ"], lp["pos"][:, :n, :n], lp["before"][:, :n, :n]
    B = start.shape[0]
    trans = succ[:, :n, :n].copy()
    trans[:, np.arange(n), np.arange(n)] = -np.inf
    end = succ[:, :n, MAX_LINES]
    full = 1 << n
    bits = ((np.arange(full)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)   # [full, n]
    count = bits.sum(1).astype(np.int64)
    dp = np.full((B, full, n), -np.inf)
    back = np.zeros((B, full, n), np.int64)
    rows = np.arange(B)
    for j in range(n):
        dp[:, 1 << j, j] = start[:, j] + w_pos * pos[:, j, 0]
    for mask in range(1, full):                       # increasing masks: every subset is finished before use
        for j in range(n):
            if not (mask >> j) & 1 or mask == (1 << j):
                continue
            prev = mask ^ (1 << j)
            gain = w_pos * pos[:, j, count[prev]] + w_before * (before[:, :, j] @ bits[prev])
            cand = dp[:, prev, :] + trans[:, :, j]
            k = cand.argmax(axis=1)
            dp[:, mask, j] = cand[rows, k] + gain
            back[:, mask, j] = k
    last = (dp[:, full - 1, :] + end).argmax(axis=1)
    orders = np.zeros((B, n), np.int64)
    mask = np.full(B, full - 1)
    cur = last.copy()
    for p in range(n - 1, -1, -1):
        orders[:, p] = cur
        prev_cur = back[rows, mask, cur]
        mask = mask ^ (1 << cur)
        cur = prev_cur
    return orders


def predict_orders(models, data, cfg=CONFIG, lp=None):
    """Best printed order for every stanza. Stanzas are grouped by size only to vectorise the dynamic programme;
    each row of a group is solved separately, so one stanza's result never depends on another."""
    if lp is None:
        lp = log_probs(models, data)
    orders = [None] * data["size"]
    for n in np.unique(data["n"]).tolist():
        rows = np.flatnonzero(data["n"] == n)
        best = decode({k: v[rows] for k, v in lp.items()}, n, cfg["w_pos"], cfg["w_before"])
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

    # Final stage: test.csv is read only now; every stanza is featurised and decoded on its own.
    test_table = pd.read_csv(public_dir / "test.csv")
    test = featurise(test_table, vocab)
    orders = predict_orders(models, test)
    submission = to_submission(test, orders, sample)
    assert list(submission.columns) == list(sample.columns)
    assert submission.iloc[:, 0].tolist() == sample.iloc[:, 0].tolist()
    for o, n in zip(orders, test["n"]):
        assert sorted(o) == list(range(n))
    submission.to_csv(submission_out, index=False, encoding="utf-8")
    print(f"total {minutes():.1f} min -> {submission_out}", flush=True)


if __name__ == "__main__":
    main()
