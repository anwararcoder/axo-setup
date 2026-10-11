"""
Who Cited Whom? Restoring the lost directions of a citation graph

Task
  For every link of a test graph whose direction was removed, say whether paper_a cites paper_b (1) or the
  reverse (0). Scored by the Matthews correlation coefficient over all answered links of all test graphs.

Approach (version 1)
  A citation runs from the newer paper to the older one, so the direction of a link is a comparison of the
  "age" of its two papers. Age is not given, but it leaves traces in the shape of the graph: old, foundational
  papers collect many links because later papers cite them, new papers link mostly to well-connected older ones,
  and the roughly one-in-ten surviving directions of the same graph say directly who cites whom.
  1. For every graph, structural numbers per paper are computed from that graph alone: degree (raw, relative to
     the graph's largest, rank inside the graph), triangles and clustering, neighbours' degrees, core number,
     two-step reach, eigenvector centrality, and the counts of known outgoing and incoming citations among the
     graph's surviving directions, plus the neighbours' average of the known balance.
  2. A link is described by the numbers of both papers, their differences, the shared neighbours, and the known
     citation chains between the two papers (a cites x and x cites b says that a is the newer paper).
  3. Training imitates the test exactly: in every training graph each direction is kept with probability 0.1 (a
     fixed random draw), the kept directions become that graph's surviving inputs and the other links are the rows
     to learn. Every row is also added with the two papers swapped and the label flipped, because which paper is
     listed first is a coin flip and must not carry information.
  4. One gradient-boosted tree model (scikit-learn HistGradientBoostingClassifier) is trained on these rows.
  5. Each test graph is then handled on its own: its features use only its own links and its own surviving
     directions, both orders of each link are scored, and the two probabilities are averaged,
     p = (P(a cites b) + 1 - P(b cites a)) / 2; the answer is 1 when p > 0.5.

Deterministic runtime plan (keep this section; the platform checks for it)
  * One fixed device: the CPU. No GPU, no device selection, no fallback, no retry logic.
  * A fixed thread count (CPU_THREADS = 4), assigned to the OpenMP and BLAS settings before NumPy is imported
    and enforced with threadpoolctl around training and prediction; it is a constant, not read from the hardware.
  * Fixed seeds for every random draw: the simulated surviving directions use one numpy RandomState per training
    graph, seeded from the seed and the graph's position in sorted graph_id order. The model has a fixed
    random_state, a fixed number of boosting iterations and no early stopping. The clock is read only to print
    progress.
  * No result depends on the iteration order of a set or on Python's hash seed: graphs are processed in sorted
    graph_id order and everything else is index arithmetic on arrays.

Rules followed
  * Only the five released files are read; no pretrained weights and no other data.
  * Everything learned (the tree model) and every setting comes from the training graphs alone.
  * The answer for a test graph is computed from its own links and its own surviving directions only, one graph
    at a time, after training has finished; no statistic is computed across test graphs.
  * Paper numbers, link and graph ids, row order and the listed order of a link's two papers are never used as
    signals.
"""
import os
import random
import sys
import time
from pathlib import Path

CPU_THREADS = 4
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)  # assigned (never setdefault), before NumPy is imported
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

# One fixed plan: the amount of work, the device and the paths never depend on the clock, a score, the CPU count
# or files found on disk.
CONFIG = dict(
    seed=0,
    known_share=0.1,      # the share of directions that survive in a test graph, as stated in the description
    mask_draws=1,         # simulated surviving-direction draws per training graph
    max_iter=200,         # boosting iterations (fixed; no early stopping)
    learning_rate=0.1,
    max_leaf_nodes=31,
    min_samples_leaf=40,
)
DEVICE = "cpu"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_train_graphs(public_dir):
    """Training graphs in sorted graph_id order, papers renumbered from 0. y[i] = 1 when a[i] cites b[i]."""
    links = pd.read_csv(public_dir / "train.csv").merge(
        pd.read_csv(public_dir / "train_labels.csv"), on="link_id", how="left", validate="one_to_one")
    graphs = []
    for graph_id, df in links.groupby("graph_id", sort=True):
        graphs.append(dict(graph_id=graph_id,
                           a=df["paper_a"].to_numpy(np.int64) - 1,
                           b=df["paper_b"].to_numpy(np.int64) - 1,
                           y=df["a_cites_b"].to_numpy(np.int64)))
    return graphs


def load_test_graphs(public_dir):
    """Test graphs in sorted graph_id order. Each holds its links to answer (a, b, link_id) and its own surviving
    directions (known_src cites known_dst); `all_a`/`all_b` is the graph's full wiring (answered + known links)."""
    test = pd.read_csv(public_dir / "test.csv")
    known = pd.read_csv(public_dir / "test_known.csv")
    known_by_graph = {graph_id: df for graph_id, df in known.groupby("graph_id", sort=True)}
    graphs = []
    for graph_id, df in test.groupby("graph_id", sort=True):
        kd = known_by_graph.get(graph_id, known.iloc[0:0])
        ka = kd["paper_a"].to_numpy(np.int64) - 1
        kb = kd["paper_b"].to_numpy(np.int64) - 1
        ky = kd["a_cites_b"].to_numpy(np.int64)
        a = df["paper_a"].to_numpy(np.int64) - 1
        b = df["paper_b"].to_numpy(np.int64) - 1
        graphs.append(dict(graph_id=graph_id, link_id=df["link_id"].to_numpy(), a=a, b=b,
                           all_a=np.concatenate([a, ka]), all_b=np.concatenate([b, kb]),
                           known_src=np.where(ky == 1, ka, kb), known_dst=np.where(ky == 1, kb, ka)))
    return graphs


# ----------------------------------------------------------------------------
# Features: everything is computed from one graph's own wiring and its own surviving directions
# ----------------------------------------------------------------------------
def core_numbers(U):
    """k-core number of every node (repeatedly remove the node of smallest remaining degree)."""
    n = U.shape[0]
    d = U.sum(1).astype(np.int64)
    alive = np.ones(n, bool)
    core = np.zeros(n)
    k = 0
    for _ in range(n):
        candidates = np.nonzero(alive)[0]
        v = candidates[np.argmin(d[candidates])]  # argmin takes the first minimum: a fixed tie rule
        k = max(k, int(d[v]))
        core[v] = k
        alive[v] = False
        d[np.nonzero(U[v] > 0)[0]] -= 1
    return core


def graph_tables(n, a, b, known_src, known_dst):
    """Per-node feature matrix and the pair tables of one graph."""
    U = np.zeros((n, n))
    U[a, b] = 1.0
    U[b, a] = 1.0
    K = np.zeros((n, n))  # K[s, d] = 1: s is known to cite d
    K[known_src, known_dst] = 1.0

    deg = U.sum(1)
    UU = U @ U
    tri = np.einsum("ij,ji->i", UU, U) / 2.0
    pairs = deg * (deg - 1.0) / 2.0
    clust = np.divide(tri, pairs, out=np.zeros(n), where=pairs > 0)
    nbr_deg_mean = (U @ deg) / np.maximum(deg, 1.0)
    nbr_deg_max = (U * deg[None, :]).max(1)
    two_hop = ((UU + U) > 0).sum(1) - 1.0  # the diagonal of UU is the node itself
    less = (deg[None, :] < deg[:, None]).sum(1)
    ties = (deg[None, :] == deg[:, None]).sum(1) - 1
    deg_rank = (less + 0.5 * ties) / (n - 1.0)
    core = core_numbers(U)
    eig = np.ones(n)
    M = U + np.eye(n)  # the self loop keeps the power iteration from oscillating on bipartite parts
    for _ in range(100):
        eig = M @ eig
        eig = eig / eig.max()

    kout, kin = K.sum(1), K.sum(0)
    knet = kout - kin  # positive: known to cite more than to be cited, i.e. a newer paper
    nbr_knet = (U @ knet) / np.maximum(deg, 1.0)

    F = np.column_stack([
        deg, deg / deg.max(), deg_rank, tri, clust, nbr_deg_mean, nbr_deg_max, deg / np.maximum(nbr_deg_mean, 1.0),
        core, core / max(core.max(), 1.0), two_hop / (n - 1.0), eig,
        kout, kin, knet, nbr_knet, (kout + kin) / np.maximum(deg, 1.0),
    ])

    # Reachability along known citations only (a known chain a -> ... -> b says a is newer than b).
    R = K > 0
    for k in range(n):
        R = R | (R[:, k:k + 1] & R[k:k + 1, :])

    m = len(a)
    graph_row = np.array([n, m, m / (n * (n - 1) / 2.0), len(known_src) / max(m, 1)])
    return dict(F=F, deg=deg, UU=UU, KK=K @ K, R=R.astype(np.float64), graph=graph_row)


def pair_features(t, p, q):
    """Feature rows for the ordered pairs (p, q) of one graph: 'does p cite q?'."""
    Fp, Fq = t["F"][p], t["F"][q]
    cn = t["UU"][p, q]
    union = t["deg"][p] + t["deg"][q] - 2.0 - cn
    jaccard = np.divide(cn, union, out=np.zeros(len(p)), where=union > 0)
    k2_pq, k2_qp = t["KK"][p, q], t["KK"][q, p]
    r_pq, r_qp = t["R"][p, q], t["R"][q, p]
    pair = np.column_stack([cn, jaccard, k2_pq, k2_qp, k2_pq - k2_qp, r_pq, r_qp, r_pq - r_qp])
    return np.hstack([Fp, Fq, Fp - Fq, pair, np.repeat(t["graph"][None, :], len(p), axis=0)])


def training_rows(graphs, cfg):
    """Rows imitating the test: per training graph a fixed draw keeps each direction with probability
    known_share; the kept directions are inputs, the other links are rows to learn, in both orders."""
    xs, ys = [], []
    for gi, g in enumerate(graphs):
        n = int(max(g["a"].max(), g["b"].max())) + 1
        src = np.where(g["y"] == 1, g["a"], g["b"])
        dst = np.where(g["y"] == 1, g["b"], g["a"])
        for draw in range(cfg["mask_draws"]):
            rng = np.random.RandomState(cfg["seed"] * 1_000_003 + draw * 10_007 + gi)
            keep = rng.random_sample(len(g["a"])) < cfg["known_share"]
            t = graph_tables(n, g["a"], g["b"], src[keep], dst[keep])
            p, q, y = g["a"][~keep], g["b"][~keep], g["y"][~keep]
            xs += [pair_features(t, p, q), pair_features(t, q, p)]
            ys += [y, 1 - y]
    return np.vstack(xs), np.concatenate(ys)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def fit(X, y, cfg):
    """One gradient-boosted tree model with a fixed number of iterations and a fixed seed."""
    model = HistGradientBoostingClassifier(
        max_iter=cfg["max_iter"], learning_rate=cfg["learning_rate"], max_leaf_nodes=cfg["max_leaf_nodes"],
        min_samples_leaf=cfg["min_samples_leaf"], early_stopping=False, random_state=cfg["seed"])
    with threadpool_limits(limits=CPU_THREADS):
        model.fit(X, y)
    return model


def predict_pairs(model, t, p, q):
    """P(p cites q) made symmetric: the average of scoring (p, q) and the flipped (q, p)."""
    with threadpool_limits(limits=CPU_THREADS):
        forward = model.predict_proba(pair_features(t, p, q))[:, 1]
        backward = model.predict_proba(pair_features(t, q, p))[:, 1]
    return 0.5 * (forward + 1.0 - backward)


def main():
    # Required input/output pattern. Keep the names public_dir / submission_out everywhere in the file.
    # Arguments are tolerated, never required: the file also runs when pasted into one notebook cell.
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
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    print(f"numpy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}, "
          f"device {DEVICE}, threads {CPU_THREADS}", flush=True)

    # Safety net: a valid submission (all 0, which scores 0) exists from the first seconds; it is replaced by the
    # real predictions at the end and is never a prediction method.
    sample = pd.read_csv(public_dir / "sample_submission.csv")
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(submission_out, index=False, encoding="utf-8")

    # Training: training graphs only.
    train_graphs = load_train_graphs(public_dir)
    X, y = training_rows(train_graphs, cfg)
    print(f"{len(train_graphs)} training graphs, {len(y)} training rows x {X.shape[1]} features "
          f"at {minutes():.1f} min", flush=True)
    model = fit(X, y, cfg)
    print(f"model trained at {minutes():.1f} min", flush=True)

    # Prediction: the test files are read only now, and each test graph is answered on its own from its own links
    # and its own surviving directions.
    test_graphs = load_test_graphs(public_dir)
    answers = {}
    for i, g in enumerate(test_graphs):
        n = int(max(g["all_a"].max(), g["all_b"].max())) + 1
        t = graph_tables(n, g["all_a"], g["all_b"], g["known_src"], g["known_dst"])
        prob = predict_pairs(model, t, g["a"], g["b"])
        for link_id, value in zip(g["link_id"], (prob > 0.5).astype(int)):
            answers[link_id] = int(value)
        if (i + 1) % 50 == 0 or i + 1 == len(test_graphs):
            print(f"{i + 1}/{len(test_graphs)} test graphs answered at {minutes():.1f} min", flush=True)

    # Final stage: the CSV in the sample's id order, every value 0 or 1.
    submission = sample[["link_id"]].copy()
    submission["a_cites_b"] = [answers.get(link_id, 0) for link_id in submission["link_id"]]
    print(f"{sum(link_id not in answers for link_id in submission['link_id'])} sample ids without an answer",
          flush=True)
    submission.to_csv(submission_out, index=False, encoding="utf-8")
    print(f"share of 1: {submission['a_cites_b'].mean():.4f}; total {minutes():.1f} min -> {submission_out}",
          flush=True)


if __name__ == "__main__":
    main()
