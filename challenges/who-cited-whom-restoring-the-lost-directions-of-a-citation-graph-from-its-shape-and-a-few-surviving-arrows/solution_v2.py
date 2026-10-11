"""
Who Cited Whom? Restoring the lost directions of a citation graph

Task
  For every link of a test graph whose direction was removed, say whether paper_a cites paper_b (1) or the
  reverse (0). Scored by the Matthews correlation coefficient over all answered links of all test graphs.

Approach (version 2)
  A citation runs from the newer paper to the older one, so a link's direction compares the "age" of its two
  papers. Age is not given; it leaves traces in the shape of the graph (old papers collect links because later
  papers cite them) and in the surviving directions. If the share of each paper's links that point to it were known,
  comparing the two papers' shares would orient almost every link; the model is built to estimate exactly that kind
  of per-paper quantity from the whole graph.
  1. Per paper, structural numbers are computed from its own graph: degree (raw, relative to the graph's largest,
     rank in the graph), triangles and clustering, neighbours' degrees, core number, two-step reach, eigenvector
     centrality, and the graph's size and density. They are standardised with means and spreads of the training
     papers.
  2. A message-passing network, trained from random initialisation in this file, builds a vector per paper. Each
     layer combines the paper's own vector with the mean over all its linked papers and, separately, the sums over
     the papers it is known to cite and the papers known to cite it (the surviving directions of its own graph).
  3. A link is scored from the two papers' vectors by an antisymmetric head: logit(a cites b) = f(a, b) - f(b, a), so
     swapping the two papers flips the answer exactly and the listed order (a coin flip) carries no information.
  4. Two rounds: after the first round the predicted directions of the answered links are written next to the
     surviving ones in a soft "who cites whom" matrix, and a second set of layers passes messages along it, so each
     paper's estimate can use what was inferred about its neighbours. Both rounds are trained (summed losses); the
     answer comes from the second round.
  5. Training imitates the test: in every pass each training graph gets a new fixed draw that keeps each direction
     with probability 0.1 as the surviving input; the loss is on the other links.
  6. Three networks with seeds 0, 1 and 2 (fixed in advance) are trained; their probabilities are averaged and the
     answer is 1 when the average is above 0.5.
  7. Each test graph is answered on its own, in a batch of one graph, from its own links and its own surviving
     directions, after training has finished.

Deterministic runtime plan (keep this section; the platform checks for it)
  * One fixed device: DEVICE = "cpu". No GPU code, no device selection, no fallback, no data-loader workers, no mixed
    precision, no retry logic.
  * A fixed thread count (CPU_THREADS = 4): assigned to the OpenMP and BLAS settings before NumPy and torch are
    imported, passed to torch.set_num_threads and enforced with threadpoolctl; a constant, not read from the
    hardware.
  * Fixed seeds: make_deterministic(seed) seeds Python, NumPy and torch and sets use_deterministic_algorithms(True),
    cuDNN deterministic, benchmarking off and TF32 off; it is called before every network is created. Every random
    draw (batch order, simulated surviving directions) comes from its own numpy RandomState with a fixed seed.
  * A fixed number of passes (30) with a fixed one-cycle schedule; no early stopping; the clock is read only to
    print progress.
  * No result depends on the iteration order of a set or on Python's hash seed: graphs are processed in sorted
    graph_id order and everything else is index arithmetic on arrays.

Rules followed
  * Only the five released files are read; no pretrained weights and no other data. Every learned parameter is
    created from random initialisation in this file and trained here on the training graphs.
  * Everything learned (network weights, standardisation constants) and every setting comes from the training
    graphs alone.
  * The answer for a test graph is computed from its own links and its own surviving directions only, one graph at
    a time, after training has finished; no statistic is computed across test graphs.
  * Paper numbers, link and graph ids, row order and the listed order of a link's two papers are never used as
    signals.
"""
import os
import random
import sys
import time
from pathlib import Path

CPU_THREADS = 4
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # assigned (never setdefault), before torch is imported
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from threadpoolctl import threadpool_limits

# One fixed plan: the amount of work, the device and the paths never depend on the clock, a score, the CPU count or
# files found on disk.
CONFIG = dict(
    seeds=[0, 1, 2],      # networks averaged; fixed in advance
    epochs=30,            # passes over the training graphs (fixed; no early stopping)
    width=64,             # vector size per paper
    layers=4,             # message-passing layers per round
    rounds=2,             # rounds of message passing (round 2 uses round 1's predicted directions)
    lr=2e-3,
    weight_decay=0.01,
    batch_graphs=32,
    known_share=0.1,      # the share of directions that survive in a test graph, as stated in the description
)
DEVICE = "cpu"
LOG_COLUMNS = [0, 3, 5, 6, 8]  # count-like structural columns, used on a log scale


def make_deterministic(seed):
    """Fixed seeds and deterministic kernels. Called before each network is built."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


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
    """Test graphs in sorted graph_id order: the links to answer (a, b, link_id), the full wiring (all_a, all_b =
    answered + surviving links) and the graph's own surviving directions (known_src cites known_dst)."""
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
# Structural numbers per paper: computed from one graph's own wiring
# ----------------------------------------------------------------------------
def core_numbers(U):
    """k-core number of every paper (repeatedly remove the paper of smallest remaining degree)."""
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


def structure_features(n, a, b):
    """Per-paper structural numbers of one graph (no direction information)."""
    U = np.zeros((n, n))
    U[a, b] = 1.0
    U[b, a] = 1.0
    deg = U.sum(1)
    UU = U @ U
    tri = np.einsum("ij,ji->i", UU, U) / 2.0
    pairs = deg * (deg - 1.0) / 2.0
    clust = np.divide(tri, pairs, out=np.zeros(n), where=pairs > 0)
    nbr_deg_mean = (U @ deg) / np.maximum(deg, 1.0)
    nbr_deg_max = (U * deg[None, :]).max(1)
    two_hop = ((UU + U) > 0).sum(1) - 1.0  # the diagonal of UU is the paper itself
    less = (deg[None, :] < deg[:, None]).sum(1)
    ties = (deg[None, :] == deg[:, None]).sum(1) - 1
    deg_rank = (less + 0.5 * ties) / (n - 1.0)
    core = core_numbers(U)
    eig = np.ones(n)
    M = U + np.eye(n)  # the self loop keeps the power iteration from oscillating on bipartite parts
    for _ in range(100):
        eig = M @ eig
        eig = eig / eig.max()
    m = len(a)
    graph_row = np.array([np.log(n), np.log(m), m / (n * (n - 1) / 2.0)])
    X = np.column_stack([
        deg, deg / deg.max(), deg_rank, tri, clust, nbr_deg_mean, nbr_deg_max, deg / np.maximum(nbr_deg_mean, 1.0),
        core, core / max(core.max(), 1.0), two_hop / (n - 1.0), eig,
    ])
    X[:, LOG_COLUMNS] = np.log1p(X[:, LOG_COLUMNS])
    return np.hstack([X, np.repeat(graph_row[None, :], n, axis=0)])


def prepare(n, a, b, known_src, known_dst):
    """One graph ready for the network: wiring, structural numbers, surviving directions."""
    return dict(n=n, a=a, b=b, X=structure_features(n, a, b), known_src=known_src, known_dst=known_dst)


def to_tensors(items, mean, std, ask_a, ask_b):
    """Padded tensors for a list of prepared graphs and, per graph, the pairs (ask_a cites ask_b?) to score."""
    N = max(it["n"] for it in items)
    B = len(items)
    X = np.zeros((B, N, items[0]["X"].shape[1]), np.float32)
    U = np.zeros((B, N, N), np.float32)
    K = np.zeros((B, N, N), np.float32)
    node_mask = np.zeros((B, N, 1), np.float32)
    bi, p, q = [], [], []
    for i, it in enumerate(items):
        n = it["n"]
        X[i, :n] = (it["X"] - mean) / std
        U[i, it["a"], it["b"]] = 1.0
        U[i, it["b"], it["a"]] = 1.0
        K[i, it["known_src"], it["known_dst"]] = 1.0
        node_mask[i, :n] = 1.0
        bi.append(np.full(len(ask_a[i]), i))
        p.append(ask_a[i])
        q.append(ask_b[i])
    t = torch.from_numpy
    return t(X), t(U), t(K), t(node_mask), t(np.concatenate(bi)), t(np.concatenate(p)), t(np.concatenate(q))


# ----------------------------------------------------------------------------
# Network (from random initialisation)
# ----------------------------------------------------------------------------
class Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_, self.und, self.out, self.inn = (nn.Linear(d, d) for _ in range(4))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, U, K, deg, node_mask):
        m_und = torch.bmm(U, h) / deg            # mean over all linked papers
        m_out = torch.bmm(K, h)                  # papers this one cites (surviving or predicted)
        m_in = torch.bmm(K.transpose(1, 2), h)   # papers that cite this one (surviving or predicted)
        z = self.self_(h) + self.und(m_und) + self.out(m_out) + self.inn(m_in)
        return (h + torch.relu(self.norm(z))) * node_mask


def edge_logit(head, h, bi, p, q, cn):
    hp, hq = h[bi, p], h[bi, q]
    forward = head(torch.cat([hp, hq, hp - hq, hp * hq, cn], 1))
    backward = head(torch.cat([hq, hp, hq - hp, hp * hq, cn], 1))
    return (forward - backward).squeeze(1)  # antisymmetric: logit of "p cites q"


class Net(nn.Module):
    def __init__(self, d_in, d, layers, rounds):
        super().__init__()
        self.inp = nn.Linear(d_in + 2, d)
        self.rounds = nn.ModuleList(nn.ModuleList(Layer(d) for _ in range(layers)) for _ in range(rounds))
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(4 * d + 1, d), nn.ReLU(), nn.Linear(d, 1)) for _ in range(rounds))

    def forward(self, X, U, K, node_mask, bi, p, q):
        """Logits of 'p cites q' after every round. K holds the surviving directions (K[s, d] = 1: s cites d)."""
        kout, kin = K.sum(2, keepdim=True), K.sum(1).unsqueeze(2)
        h = torch.relu(self.inp(torch.cat([X, torch.log1p(kout), torch.log1p(kin)], 2))) * node_mask
        deg = U.sum(2, keepdim=True).clamp(min=1.0)
        cn = torch.bmm(U, U)[bi, p, q].unsqueeze(1)  # shared neighbours of the two papers
        directed, outputs = K, []
        for layers, head in zip(self.rounds, self.heads):
            for layer in layers:
                h = layer(h, U, directed, deg, node_mask)
            logit = edge_logit(head, h, bi, p, q, cn)
            outputs.append(logit)
            prob = torch.sigmoid(logit)
            # the next round sees the predicted directions of the answered links next to the surviving ones
            directed = K.index_put((bi, p, q), prob).index_put((bi, q, p), 1.0 - prob)
        return outputs


def train_net(graphs, prepared, mean, std, cfg, seed, progress):
    """Train one network on the training graphs with a fresh simulated surviving-direction draw per pass."""
    make_deterministic(seed)
    net = Net(prepared[0]["X"].shape[1], cfg["width"], cfg["layers"], cfg["rounds"]).to(DEVICE)
    optim = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    per_epoch = (len(graphs) + cfg["batch_graphs"] - 1) // cfg["batch_graphs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=cfg["lr"], total_steps=cfg["epochs"] * per_epoch,
                                                pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    net.train()
    for epoch in range(cfg["epochs"]):
        order = np.random.RandomState(seed * 7919 + epoch).permutation(len(graphs))
        for s in range(0, len(order), cfg["batch_graphs"]):
            items, ask_a, ask_b, ys = [], [], [], []
            for gi in order[s:s + cfg["batch_graphs"]]:
                g, pr = graphs[gi], prepared[gi]
                # surviving directions for this pass: fixed draw per (pass, graph), probability known_share
                keep = np.random.RandomState((1000 + epoch) * 1_000_003 + int(gi)).random_sample(len(g["a"])) \
                    < cfg["known_share"]
                src = np.where(g["y"] == 1, g["a"], g["b"])
                dst = np.where(g["y"] == 1, g["b"], g["a"])
                items.append(dict(pr, known_src=src[keep], known_dst=dst[keep]))
                ask_a.append(g["a"][~keep])
                ask_b.append(g["b"][~keep])
                ys.append(g["y"][~keep])
            X, U, K, nm, bi, p, q = to_tensors(items, mean, std, ask_a, ask_b)
            y = torch.from_numpy(np.concatenate(ys).astype(np.float32))
            loss = sum(loss_fn(logit, y) for logit in net(X, U, K, nm, bi, p, q))
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
        progress((epoch + 1) / cfg["epochs"])
    net.eval()
    return net


def predict_graph(nets, item, mean, std):
    """P(a cites b) for the links to answer of ONE graph, averaged over the networks."""
    X, U, K, nm, bi, p, q = to_tensors([item], mean, std, [item["ask_a"]], [item["ask_b"]])
    with torch.no_grad():
        return np.mean([torch.sigmoid(net(X, U, K, nm, bi, p, q)[-1]).numpy() for net in nets], axis=0)


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
    torch.set_num_threads(CPU_THREADS)
    make_deterministic(0)
    print(f"numpy {np.__version__}, pandas {pd.__version__}, torch {torch.__version__}, device {DEVICE}, "
          f"threads {CPU_THREADS}", flush=True)

    # Safety net: a valid submission (all 0, which scores 0) exists from the first seconds; it is replaced by the
    # real predictions at the end and is never a prediction method.
    sample = pd.read_csv(public_dir / "sample_submission.csv")
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(submission_out, index=False, encoding="utf-8")

    with threadpool_limits(limits=CPU_THREADS):
        # Training: training graphs only. Standardisation constants come from the training papers.
        train_graphs = load_train_graphs(public_dir)
        prepared = [prepare(int(max(g["a"].max(), g["b"].max())) + 1, g["a"], g["b"], None, None)
                    for g in train_graphs]
        stacked = np.vstack([pr["X"] for pr in prepared])
        mean, std = stacked.mean(0), stacked.std(0) + 1e-6
        print(f"{len(train_graphs)} training graphs prepared at {minutes():.1f} min", flush=True)

        nets = []
        total = len(cfg["seeds"])
        for k, seed in enumerate(cfg["seeds"]):
            def progress(fraction, k=k):
                done = (k + fraction) / total
                if fraction == 1.0 or int(fraction * cfg["epochs"]) % 10 == 0:
                    print(f"network {k + 1}/{total}: {fraction:6.1%} of its passes, {minutes():.1f} min elapsed, "
                          f"about {minutes() / done * (1 - done):.1f} min of training left", flush=True)
            nets.append(train_net(train_graphs, prepared, mean, std, cfg, seed, progress))

        # Prediction: the test files are read only now, and each test graph is answered on its own, in a batch of
        # one graph, from its own links and its own surviving directions.
        test_graphs = load_test_graphs(public_dir)
        answers = {}
        for i, g in enumerate(test_graphs):
            n = int(max(g["all_a"].max(), g["all_b"].max())) + 1
            item = prepare(n, g["all_a"], g["all_b"], g["known_src"], g["known_dst"])
            item["ask_a"], item["ask_b"] = g["a"], g["b"]
            prob = predict_graph(nets, item, mean, std)
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
