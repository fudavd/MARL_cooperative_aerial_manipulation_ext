"""Are the hidden variables of the two task families decodable from the actor observation, and with how much history?

hook (switch_hook.pt):            attached flag (the hook-blind actor never sees it)        -> balanced accuracy
transport (variant_transport.pt): payload mass, the drone's own attachment point (body)     -> R^2

Inputs: the stored frame stack (3 consecutive control steps = 153), its current frame (51), and a longer window of
10 stored frames (stored every 2nd control step -> 20 control steps, 510 dims); each also without / only the 6-dim
INDI residual. Held-out TASK GROUPS (unseen attachment geometry for transport). Linear and MLP-128 probes.
"""
import json
import sys
import time

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NO_RES = [i for i in range(51) if i not in range(45, 51)]


def windows(obs, first, k_stored):
    """[G,T,L,153] -> features [G,T,L,k*51] of the last k stored current frames (oldest first); valid = the window
    lies inside one episode (episode id = cumulative count of `first` flags along T)."""
    cur = obs[..., -51:].float()
    G, T, L, _ = cur.shape
    epid = first.long().cumsum(1)
    feats = []
    for j in range(k_stored - 1, -1, -1):           # oldest first
        sh = torch.zeros_like(cur)
        sh[:, j:] = cur[:, :T - j]
        feats.append(sh)
    valid = torch.zeros(G, T, L, dtype=torch.bool)
    valid[:, k_stored - 1:] = epid[:, k_stored - 1:] == epid[:, :T - k_stored + 1]
    return torch.cat(feats, -1), valid


def fit_probe(xtr, ytr, xte, yte, task, hidden=0, epochs=400, lr=3e-3):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    d, o = xtr.shape[1], ytr.shape[1]
    net = (torch.nn.Linear(d, o) if hidden == 0 else
           torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ELU(), torch.nn.Linear(hidden, o))).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    if task == "cls":
        pos = ytr.mean().clamp(1e-3, 1 - 1e-3)
        pw = ((1 - pos) / pos).to(DEV)
    bs = 8192
    for ep in range(epochs):
        perm = torch.randperm(len(xtr), device=DEV)
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            out = net(xtr[idx])
            if task == "cls":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out, ytr[idx], pos_weight=pw)
            else:
                loss = torch.nn.functional.mse_loss(out, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = net(xte)
        if task == "cls":
            pred = (out > 0).float()
            tp = ((pred == 1) & (yte == 1)).float().sum(); fn = ((pred == 0) & (yte == 1)).float().sum()
            tn = ((pred == 0) & (yte == 0)).float().sum(); fp = ((pred == 1) & (yte == 0)).float().sum()
            return {"bal_acc": float(0.5 * (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9))),
                    "acc": float((pred == yte).float().mean())}
        ss_res = ((out - yte) ** 2).sum(0); ss_tot = ((yte - yte.mean(0)) ** 2).sum(0)
        return {"r2": [round(float(v), 3) for v in (1 - ss_res / ss_tot)], "r2_mean": float((1 - ss_res / ss_tot).mean())}


def run(path, targets, task, n_max=150_000, seed=0):
    d = torch.load(path, map_location="cpu", weights_only=False)
    obs, first, aux = d["obs"], d["first"], d["aux"]
    G = obs.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(G, generator=g)
    tr_g, te_g = perm[: int(0.7 * G)], perm[int(0.7 * G):]
    print(f"== {path}: obs {tuple(obs.shape)}, train groups {len(tr_g)}, test groups {len(te_g)}", flush=True)
    y = torch.stack([aux[t].float() for t in targets], -1)          # [G,T,L,n]
    if task == "cls":
        print(f"   positive rate {float(y.mean()):.3f}", flush=True)
    else:
        print(f"   target std {[round(float(v), 4) for v in y.reshape(-1, y.shape[-1]).std(0)]}", flush=True)
    results = {}
    variants = {}
    stack = obs.float()
    variants["1 frame (51)"] = (stack[..., -51:], None)
    variants["3 frames (153)"] = (stack, None)
    variants["3 frames, no INDI residual"] = (stack[..., [f * 51 + i for f in range(3) for i in NO_RES]], None)
    variants["3 frames, INDI residual only"] = (stack[..., [f * 51 + i for f in range(3) for i in range(45, 51)]], None)
    x10, v10 = windows(obs, first, 10)
    variants["10 stored frames = 20 steps (510)"] = (x10, v10)
    variants["20 steps, INDI residual only (60)"] = (x10[..., [f * 51 + i for f in range(10) for i in range(45, 51)]], v10)
    for name, (x, valid) in variants.items():
        t0 = time.time()
        if valid is None:
            valid = torch.ones(x.shape[:3], dtype=torch.bool)

        def sel(groups):
            xs, ys = x[groups][valid[groups]], y[groups][valid[groups]]
            if len(xs) > n_max:
                idx = torch.randperm(len(xs), generator=g)[:n_max]
                xs, ys = xs[idx], ys[idx]
            return xs.to(DEV), ys.to(DEV)
        xtr, ytr = sel(tr_g); xte, yte = sel(te_g)
        finite = torch.isfinite(xtr).all(1); xtr, ytr = xtr[finite], ytr[finite]
        finite = torch.isfinite(xte).all(1); xte, yte = xte[finite], yte[finite]
        lin = fit_probe(xtr, ytr, xte, yte, task, 0)
        mlp = fit_probe(xtr, ytr, xte, yte, task, 128)
        results[name] = {"linear": lin, "mlp128": mlp, "n_train": int(len(xtr)), "n_test": int(len(xte))}
        print(f"   {name:40s} linear {lin}  mlp {mlp}  ({time.time() - t0:.0f} s)", flush=True)
    return results


if __name__ == "__main__":
    root = "/home/fuda/PycharmProjects/MARL_cooperative_aerial_manipulation_ext/projects/flycrane-meta/logs/demos/"
    out = {}
    out["hook_attached"] = run(root + "switch_hook.pt", ["attached"], "cls")
    out["transport_mass"] = run(root + "variant_transport.pt", ["payload_mass"], "reg")
    out["transport_attach_xyz"] = run(root + "variant_transport.pt", ["attach_x", "attach_y", "attach_z"], "reg")
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print("[done]", sys.argv[1])
