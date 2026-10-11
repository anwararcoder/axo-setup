"""Development: whole-order re-scoring. For every stanza, candidate orders (all permutations for n <= 8, beam
top-K by pair score for n = 9, 10) are re-scored with pair log-probs + weighted log count priors of the canonical
rhyme pattern, ending (masculine/feminine) pattern and end-punctuation pattern, counted on the training stanzas."""
import itertools
import json
from collections import Counter

import numpy as np

PERMS = {n: np.array(list(itertools.permutations(range(n))), dtype=np.int64) for n in range(4, 9)}


def canon_codes(labels_by_pos):
    """labels_by_pos [m, n] ints -> canonical pattern code per row (first-occurrence relabelling, base n+1)."""
    m, n = labels_by_pos.shape
    eq = labels_by_pos[:, :, None] == labels_by_pos[:, None, :]
    firstpos = eq.argmax(axis=2)                      # index of first position with the same label
    # relabel first positions to 0..k-1 in order of appearance: rank among distinct first positions
    is_new = firstpos == np.arange(n)[None, :]
    newid = np.cumsum(is_new, axis=1) - 1
    lab = np.take_along_axis(newid, firstpos, axis=1)
    return (lab * (n + 1) ** np.arange(n)[None, :]).sum(1) * 16 + n


def seq_codes(labels_by_pos, base):
    m, n = labels_by_pos.shape
    return (labels_by_pos * base ** np.arange(n)[None, :]).sum(1) * 16 + n


class PatternPrior:
    """log(count + alpha) of pattern codes counted on gold orders of training stanzas."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.counts = {"rhyme": Counter(), "gender": Counter(), "punct": Counter()}

    def add(self, info, order):
        o = np.array(order)[None, :]
        for name, codes in self.codes(info, o).items():
            self.counts[name][int(codes[0])] += 1

    @staticmethod
    def codes(info, orders):
        rhyme, gender, punct = info
        return {"rhyme": canon_codes(rhyme[orders]), "gender": seq_codes(gender[orders], 2),
                "punct": seq_codes(punct[orders], 10)}

    def logp(self, name, codes):
        uniq, inv = np.unique(codes, return_inverse=True)
        c = self.counts[name]
        vals = np.log(np.array([c.get(int(u), 0) for u in uniq], dtype=np.float64) + self.alpha)
        return vals[inv]


def beam_candidates(start, trans, end, n, width):
    """Top `width` complete orders by pair score (start + transitions + end)."""
    beams = [((j,), start[j], 1 << j) for j in range(n)]
    for _ in range(n - 1):
        ext = []
        for seq, sc, mask in beams:
            last = seq[-1]
            for j in range(n):
                if not (mask >> j) & 1:
                    ext.append((seq + (j,), sc + trans[last, j], mask | (1 << j)))
        ext.sort(key=lambda t: -t[1])
        beams = ext[:width]
    return np.array([b[0] for b in beams], dtype=np.int64)


def candidates(start, succ, n, max_lines, width=3000):
    trans = succ[:n, :n].copy()
    trans[np.arange(n), np.arange(n)] = -np.inf
    end = succ[:n, max_lines]
    if n <= 8:
        P = PERMS[n]
    else:
        P = beam_candidates(start[:n], trans, end, n, width)
    tot = start[P[:, 0]] + trans[P[:, :-1], P[:, 1:]].sum(1) + end[P[:, -1]]
    return P, tot


def best_order(info, start, succ, n, max_lines, prior, weights, width=3000):
    P, tot = candidates(start, succ, n, max_lines, width)
    score = tot.copy()
    codes = PatternPrior.codes(info, P)
    for name, w in weights.items():
        if w:
            score += w * prior.logp(name, codes[name])
    return P[score.argmax()]
