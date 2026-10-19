import sys, torch
sys.path.insert(0, sys.argv[1])
from probe_hidden import windows, fit_probe, DEV
root = "/home/fuda/PycharmProjects/MARL_cooperative_aerial_manipulation_ext/projects/flycrane-meta/logs/demos/"
d = torch.load(root + "variant_transport.pt", map_location="cpu", weights_only=False)
obs, first, y = d["obs"], d["first"], d["aux"]["payload_mass"].float().unsqueeze(-1)
G = obs.shape[0]; g = torch.Generator().manual_seed(0); perm = torch.randperm(G, generator=g)
tr_g, te_g = perm[:44], perm[44:]
for k in (10, 25, 50, 100):
    x, valid = windows(obs, first, k)          # k stored frames = 2k control steps
    for name, cols in (("residual only", [f * 51 + i for f in range(k) for i in range(45, 51)]),
                       ("residual + own state", [f * 51 + i for f in range(k) for i in list(range(15, 33)) + list(range(45, 51))])):
        xs = x[..., cols]
        def sel(groups):
            a, b = xs[groups][valid[groups]], y[groups][valid[groups]]
            idx = torch.randperm(len(a), generator=g)[:120000]
            return a[idx].to(DEV), b[idx].to(DEV)
        xtr, ytr = sel(tr_g); xte, yte = sel(te_g)
        m = torch.isfinite(xtr).all(1); xtr, ytr = xtr[m], ytr[m]
        m = torch.isfinite(xte).all(1); xte, yte = xte[m], yte[m]
        lin = fit_probe(xtr, ytr, xte, yte, "reg", 0, epochs=200)
        print(f"mass | {2*k:4d} control steps | {name:22s} | linear R2 {lin['r2'][0]:.3f}  (n_test {len(xte)})", flush=True)
