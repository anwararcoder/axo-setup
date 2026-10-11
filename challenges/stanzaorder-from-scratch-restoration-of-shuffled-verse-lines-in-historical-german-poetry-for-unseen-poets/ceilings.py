"""Ceilings on fold 0: v1 pair model scores; (a) v1 exact decode, (b) best pair-score order among those whose
canonical rhyme / gender / punctuation pattern equals the gold one (oracle pattern), (c) count prior on patterns."""
import importlib.util, itertools, json, sys, time
from collections import Counter, defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from dev_eval import author_folds, row_scores, summary
spec = importlib.util.spec_from_file_location("s", "solution_v1.py"); s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)

table = pd.read_csv("data/train.csv"); folds = author_folds(table)
tr, ho = table[folds != 0].reset_index(drop=True), table[folds == 0].reset_index(drop=True)
vocab = s.build_vocab(tr, 100); dtr, dho = s.featurise(tr, vocab), s.featurise(ho, vocab)
first, succ = s.targets(tr)
model = s.fit(dtr, first, succ, s.CONFIG, 0)
start, sc = s.log_probs([model], dho)

def line_info(lines):
    lines = [s.clean(x) for x in lines]
    lw = [(s.words_of(x) or [""])[-1] for x in lines]
    parts = [s.rhyme_parts(w) for w in lw]
    return [p[0] for p in parts], [int(p[1]) for p in parts], [s.end_punct(x)[0] for x in lines]

def canon(seq):
    m = {}; return tuple(m.setdefault(x, len(m)) for x in seq)

# pattern counts on the training fold (gold order)
cnt = {"rhyme": Counter(), "gender": Counter(), "punct": Counter()}
for lines, order in zip(tr["lines"], tr["order"]):
    k, g, p = line_info(json.loads(lines)); o = [int(x[1:]) - 1 for x in order.split()]
    cnt["rhyme"][canon([k[i] for i in o])] += 1; cnt["gender"][tuple(g[i] for i in o)] += 1
    cnt["punct"][tuple(p[i] for i in o)] += 1

res = defaultdict(list)
t0 = time.time()
for r in range(len(ho)):
    n = int(ho.n_lines[r])
    if n > 8: continue
    k, g, p = line_info(json.loads(ho.lines[r])); gold = [int(x[1:]) - 1 for x in ho.order[r].split()]
    perms = np.array(list(itertools.permutations(range(n))))
    tot = start[r, perms[:, 0]] + sc[r, perms[:, :-1], perms[:, 1:]].sum(1) + sc[r, perms[:, -1], s.MAX_LINES]
    def score(pr):
        nxt = {a: b for a, b in zip(pr[:-1], pr[1:])}; return sum(nxt.get(a) == b for a, b in zip(gold[:-1], gold[1:])) / (n - 1)
    res["v1"].append(score(perms[tot.argmax()]))
    for name, seqf in [("rhyme", lambda pr: canon([k[i] for i in pr])), ("gender", lambda pr: tuple(g[i] for i in pr)),
                       ("punct", lambda pr: tuple(p[i] for i in pr))]:
        pats = [seqf(pr) for pr in perms]; goldpat = seqf(gold)
        ok = np.array([q == goldpat for q in pats])
        res["oracle_" + name].append(score(perms[np.where(ok, tot, -np.inf).argmax()]))
        prior = np.log(np.array([cnt[name][q] for q in pats]) + 0.5)
        res["prior_" + name].append(score(perms[(tot + prior).argmax()]))
    pats3 = [(canon([k[i] for i in pr]), tuple(g[i] for i in pr), tuple(p[i] for i in pr)) for pr in perms]
    prior3 = sum(np.log(np.array([cnt[nm][q[j]] for q in pats3]) + 0.5) for j, nm in enumerate(["rhyme", "gender", "punct"]))
    res["prior_all"].append(score(perms[(tot + prior3).argmax()]))
    res["n"].append(n)
nn_ = np.array(res.pop("n"))
print(f"{len(nn_)} hold-out stanzas with n<=8, {time.time()-t0:.0f}s")
w = {4: 1103, 5: 201, 6: 683, 7: 108, 8: 626}
for name, v in res.items():
    v = np.array(v); byn = {m: v[nn_ == m].mean() for m in w}
    print(f"{name:15s} mean {v.mean():.4f}  test-mix(n<=8) {sum(w[m]*byn[m] for m in w)/sum(w.values()):.4f}  " + " ".join(f"{m}:{byn[m]:.3f}" for m in w))
print("top rhyme patterns n=4:", [(''.join('abcdefghij'[i] for i in q), c) for q, c in cnt['rhyme'].most_common(40) if len(q) == 4][:10])
