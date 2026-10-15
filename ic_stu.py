#!/usr/bin/env python3
"""
IC-STU: Identifiability-Certified Spatio-Temporal Unfolding for EEG artifact removal.
Reference implementation of the manuscript's method (single-channel EEGdenoiseNet configuration).

Run:
    python ic_stu.py --data_dir data --artifact eog --epochs 100

Needs EEG_all_epochs.npy and EOG_all_epochs.npy (or EMG_all_epochs.npy) from
https://github.com/ncclabsustech/EEGdenoiseNet  (folder "data").

Outputs (in --out_dir): history.csv, history.png, best.pt, results.json
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------
# Data: EEGdenoiseNet protocol (Zhang et al. 2021)
# ----------------------------------------------------------------------------------------
SNR_LEVELS = np.arange(-7, 3, 1.0)  # ten levels, -7..2 dB


def rms(x, axis=-1):
    return np.sqrt(np.mean(x ** 2, axis=axis))


def mix(eeg, art, snr_db):
    """y = x + lam * n with SNR = 10 log10(RMS(x)/RMS(lam n)); then divide by std(y)."""
    lam = rms(eeg) / (rms(art) * 10 ** (snr_db / 20.0))
    a = art * lam[:, None]
    y = eeg + a
    s = y.std(axis=1, keepdims=True)
    return (y / s).astype(np.float32), (eeg / s).astype(np.float32), (a / s).astype(np.float32)


class EEGDenoiseData:
    def __init__(self, data_dir, artifact="eog", seed=0):
        eeg = np.load(os.path.join(data_dir, "EEG_all_epochs.npy"))
        art = np.load(os.path.join(data_dir, f"{artifact.upper()}_all_epochs.npy"))
        # EEGdenoiseNet normalises every epoch to unit std before mixing
        eeg = eeg / eeg.std(axis=1, keepdims=True)
        art = art / art.std(axis=1, keepdims=True)
        rng = np.random.RandomState(seed)
        pe, pa = rng.permutation(len(eeg)), rng.permutation(len(art))
        ne, na = len(eeg), len(art)
        e_tr, e_va = int(0.8 * ne), int(0.9 * ne)
        a_tr, a_va = int(0.8 * na), int(0.9 * na)
        self.eeg = {"train": eeg[pe[:e_tr]], "val": eeg[pe[e_tr:e_va]], "test": eeg[pe[e_va:]]}
        self.art = {"train": art[pa[:a_tr]], "val": art[pa[a_tr:a_va]], "test": art[pa[a_va:]]}
        self.rng = rng
        # val / test mixtures are fixed once (as in the benchmark)
        self.fixed = {k: self._draw(k, np.random.RandomState(seed + 1 + i), balanced=True)
                      for i, k in enumerate(["val", "test"])}

    def _draw(self, split, rng, balanced=False):
        eeg = self.eeg[split]
        art = self.art[split][rng.randint(0, len(self.art[split]), len(eeg))]
        if balanced:
            snr = SNR_LEVELS[np.arange(len(eeg)) % len(SNR_LEVELS)]
            snr = snr[rng.permutation(len(eeg))]
        else:
            snr = rng.choice(SNR_LEVELS, len(eeg))
        y, x, a = mix(eeg, art, snr)
        return tuple(torch.from_numpy(v) for v in (y, x, a)) + (torch.from_numpy(snr.astype(np.float32)),)

    def train_epoch(self):
        """New random EEG/artifact pairing and SNR every epoch."""
        return self._draw("train", self.rng)


# ----------------------------------------------------------------------------------------
# Fixed orthonormal initial dictionaries
# ----------------------------------------------------------------------------------------
def dct_basis(T):
    t = np.arange(T)[:, None]
    k = np.arange(T)[None, :]
    D = np.cos(np.pi * (t + 0.5) * k / T)
    D[:, 0] *= 1 / math.sqrt(2)
    return D * math.sqrt(2.0 / T)  # orthonormal columns


def haar_basis(T):
    """Orthonormal Haar wavelet basis (T = power of two); columns are unit-norm atoms."""
    assert T & (T - 1) == 0
    H = np.array([[1.0]])
    while H.shape[0] < T:
        n = H.shape[0]
        H = np.vstack([np.kron(H, [1, 1]), np.kron(np.eye(n), [1, -1])])
    H = H / np.linalg.norm(H, axis=1, keepdims=True)
    return H.T  # (T, T)


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------
def group_soft(g, thr):
    """Column-wise group soft thresholding (Eq. 16). g: (B, r, K); thr: scalar tensor."""
    n = torch.sqrt((g ** 2).sum(1, keepdim=True) + 1e-12)
    return torch.clamp(1 - thr / n, min=0) * g


def spectral_norm_sq(D):
    with torch.no_grad():
        return torch.linalg.matrix_norm(D, ord=2) ** 2


class ICSTU(nn.Module):
    def __init__(self, C=1, T=512, Kx=512, Ka=512, rx=1, ra=1, K=7, sx=28, sa=6, q=2,
                 Ux_init=None, Ua_init=None, learn_Ua=True, delta=1e-3):
        super().__init__()
        self.C, self.T, self.K, self.sx, self.sa, self.q, self.delta = C, T, K, sx, sa, q, delta
        self.rx, self.ra = rx, ra
        self.Dx = nn.Parameter(torch.tensor(dct_basis(T)[:, :Kx], dtype=torch.float32))
        self.Da = nn.Parameter(torch.tensor(haar_basis(T)[:, :Ka], dtype=torch.float32))
        # spatial bases kept orthonormal through QR re-parameterisation
        ux = torch.ones(C, rx) if Ux_init is None else torch.as_tensor(Ux_init, dtype=torch.float32)
        ua = torch.ones(C, ra) if Ua_init is None else torch.as_tensor(Ua_init, dtype=torch.float32)
        self.Ux_raw = nn.Parameter(ux.clone())
        self.Ua_raw = nn.Parameter(ua.clone(), requires_grad=learn_Ua)
        self.omega = nn.Parameter(torch.zeros(K))                      # eta_k = sigmoid(omega_k)/L
        self.th_x = nn.Parameter(torch.full((K,), math.log(math.exp(0.05) - 1)))  # softplus -> 0.05
        self.th_a = nn.Parameter(torch.full((K,), math.log(math.exp(0.05) - 1)))

    # ---- helpers
    @staticmethod
    def orth(M):
        Q, R = torch.linalg.qr(M)
        sgn = torch.sign(torch.diagonal(R)).clamp(min=0) * 2 - 1  # fix sign ambiguity
        return Q * sgn

    def U(self):
        return self.orth(self.Ux_raw), self.orth(self.Ua_raw)

    @torch.no_grad()
    def renorm(self):
        for D in (self.Dx, self.Da):
            D.div_(D.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def L_hat(self):
        return spectral_norm_sq(self.Dx) + spectral_norm_sq(self.Da)  # Eq. (15)

    def mu_s(self):
        Ux, Ua = self.U()
        return torch.linalg.matrix_norm(Ux.T @ Ua, ord=2)

    # ---- unfolded stages (Eq. 22)
    def forward(self, Y):
        """Y: (B, C, T). Returns stage-output coefficients Gx (B,rx,Kx), Ga (B,ra,Ka)."""
        Ux, Ua = self.U()
        B = Y.shape[0]
        Gx = Y.new_zeros(B, self.rx, self.Dx.shape[1])
        Ga = Y.new_zeros(B, self.ra, self.Da.shape[1])
        L = self.L_hat()
        for k in range(self.K):
            R = Y - Ux @ Gx @ self.Dx.T - Ua @ Ga @ self.Da.T                 # residual
            eta = torch.sigmoid(self.omega[k]) / L                             # eta_k <= 1/L
            Gx = group_soft(Gx + eta * (Ux.T @ R @ self.Dx), eta * F.softplus(self.th_x[k]))
            Ga = group_soft(Ga + eta * (Ua.T @ R @ self.Da), eta * F.softplus(self.th_a[k]))
        return Gx, Ga

    def reconstruct(self, Gx, Ga):
        Ux, Ua = self.U()
        return Ux @ Gx @ self.Dx.T, Ua @ Ga @ self.Da.T

    # ---- support selection, refit (Eq. 17), certificate (Eq. 18); inference only
    @torch.no_grad()
    def certify(self, Y, Gx, Ga):
        dt = torch.float64
        Ux, Ua = [u.to(dt) for u in self.U()]
        Dx, Da = self.Dx.to(dt), self.Da.to(dt)
        Y, Gx, Ga = Y.to(dt), Gx.to(dt), Ga.to(dt)
        B = Y.shape[0]
        sx, sa, q = self.sx, self.sa, self.q
        nx, na = Gx.norm(dim=1), Ga.norm(dim=1)                        # (B,K) column norms
        ox = nx.topk(sx, dim=1).indices
        oa = na.topk(sa, dim=1).indices
        DxO = Dx.T[ox].transpose(1, 2)                                   # (B,T,sx)
        DaL = Da.T[oa].transpose(1, 2)                                   # (B,T,sa)
        # normal equations of Eq. (17) for the Kronecker-structured least squares
        Gxx = DxO.transpose(1, 2) @ DxO
        Gaa = DaL.transpose(1, 2) @ DaL
        Gxa = DxO.transpose(1, 2) @ DaL                                  # (B,sx,sa)
        Sxa = Ux.T @ Ua                                                  # (rx,ra)
        Ix, Ia = torch.eye(self.rx, dtype=dt), torch.eye(self.ra, dtype=dt)
        M11 = torch.einsum("ij,bkl->bikjl", Ix, Gxx).reshape(B, self.rx * sx, self.rx * sx)
        M22 = torch.einsum("ij,bkl->bikjl", Ia, Gaa).reshape(B, self.ra * sa, self.ra * sa)
        M12 = torch.einsum("ij,bkl->bikjl", Sxa, Gxa).reshape(B, self.rx * sx, self.ra * sa)
        M = torch.cat([torch.cat([M11, M12], 2), torch.cat([M12.transpose(1, 2), M22], 2)], 1)
        M = M + self.delta * torch.eye(M.shape[-1], dtype=dt)
        rx_ = (Ux.T @ Y @ DxO).reshape(B, -1)
        ra_ = (Ua.T @ Y @ DaL).reshape(B, -1)
        sol = torch.linalg.solve(M, torch.cat([rx_, ra_], 1).unsqueeze(-1)).squeeze(-1)
        Cx = sol[:, : self.rx * sx].reshape(B, self.rx, sx)
        Ca = sol[:, self.rx * sx:].reshape(B, self.ra, sa)
        Xh = Ux @ Cx @ DxO.transpose(1, 2)
        Ah = Ua @ Ca @ DaL.transpose(1, 2)
        # certificate on the supports enlarged by a margin of q atoms (conservative)
        ox2 = nx.topk(min(sx + q, nx.shape[1]), dim=1).indices
        oa2 = na.topk(min(sa + q, na.shape[1]), dim=1).indices
        Qx = torch.linalg.qr(Dx.T[ox2].transpose(1, 2)).Q
        Qa = torch.linalg.qr(Da.T[oa2].transpose(1, 2)).Q
        mu_t = torch.linalg.matrix_norm(Qx.transpose(1, 2) @ Qa, ord=2)
        mu_s = torch.linalg.matrix_norm(Ux.T @ Ua, ord=2)
        kappa = torch.sqrt((1 - mu_s * mu_t).clamp_min(0))
        return Xh.float(), Ah.float(), kappa.float(), mu_t.float()


# ----------------------------------------------------------------------------------------
# Loss (Eq. 24) and metrics (Eq. 28)
# ----------------------------------------------------------------------------------------
def icstu_loss(model, Xh, Ah, X, A, beta=0.5, gc=0.05, gg=0.01):
    """Reconstruction terms are per-element means; the dictionary terms are divided by K so
    that gamma_c and gamma_g keep a comparable scale across T."""
    rec = ((Xh - X) ** 2).mean() + beta * ((Ah - A) ** 2).mean()
    Kx, Ka = model.Dx.shape[1], model.Da.shape[1]
    coh = (model.Dx.T @ model.Da).pow(2).sum() / min(Kx, Ka)
    gram = ((model.Dx.T @ model.Dx - torch.eye(Kx)).pow(2).sum() / Kx +
            (model.Da.T @ model.Da - torch.eye(Ka)).pow(2).sum() / Ka)
    return rec + gc * coh + gg * gram, rec.detach(), coh.detach()


def rrmse_t(Xh, X):
    return (Xh - X).pow(2).mean(dim=(1, 2)).sqrt() / X.pow(2).mean(dim=(1, 2)).sqrt()


def rrmse_s(Xh, X):
    P = lambda z: torch.fft.rfft(z, dim=-1).abs().pow(2)
    ph, p = P(Xh), P(X)
    return (ph - p).pow(2).mean(dim=(1, 2)).sqrt() / p.pow(2).mean(dim=(1, 2)).sqrt()


def corr(Xh, X):
    a = Xh - Xh.mean(-1, keepdim=True)
    b = X - X.mean(-1, keepdim=True)
    return ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)).mean(-1)


@torch.no_grad()
def evaluate(model, data, kmin, beta, gc, gg, bs=256):
    model.eval()
    Y, X, A, snr = data
    Y, X, A = Y.unsqueeze(1), X.unsqueeze(1), A.unsqueeze(1)
    out = {k: [] for k in ["loss", "rt_stage", "rt_refit", "rs_refit", "cc_refit", "kappa", "mu_t"]}
    n = len(Y)
    for i in range(0, n, bs):
        y, x, a = Y[i:i + bs], X[i:i + bs], A[i:i + bs]
        Gx, Ga = model(y)
        xs, as_ = model.reconstruct(Gx, Ga)
        loss, _, _ = icstu_loss(model, xs, as_, x, a, beta, gc, gg)
        xr, ar, kap, mut = model.certify(y, Gx, Ga)
        out["loss"].append(loss.expand(len(y)))
        out["rt_stage"].append(rrmse_t(xs, x))
        out["rt_refit"].append(rrmse_t(xr, x))
        out["rs_refit"].append(rrmse_s(xr, x))
        out["cc_refit"].append(corr(xr, x))
        out["kappa"].append(kap)
        out["mu_t"].append(mut)
    out = {k: torch.cat(v) for k, v in out.items()}
    flagged = out["kappa"] < kmin
    res = {k: out[k].mean().item() for k in out}
    res["flag_rate"] = flagged.float().mean().item()
    res["rt_flagged"] = out["rt_refit"][flagged].mean().item() if flagged.any() else float("nan")
    res["rt_unflagged"] = out["rt_refit"][~flagged].mean().item() if (~flagged).any() else float("nan")
    res["_per_seg"] = out
    return res


# ----------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--artifact", default="eog", choices=["eog", "emg"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--stages", type=int, default=7)
    p.add_argument("--sx", type=int, default=28)
    p.add_argument("--sa", type=int, default=6)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma_c", type=float, default=0.05)
    p.add_argument("--gamma_g", type=float, default=0.01)
    p.add_argument("--kappa_min", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="runs/eog")
    p.add_argument("--max_train", type=int, default=0, help="debug: limit training segments")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(args.out_dir, exist_ok=True)

    ds = EEGDenoiseData(args.data_dir, args.artifact, args.seed)
    T = ds.eeg["train"].shape[1]
    model = ICSTU(C=1, T=T, Kx=T, Ka=T, rx=1, ra=1, K=args.stages, sx=args.sx, sa=args.sa)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"T={T}  stages={args.stages}  trainable parameters={nparam:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    hdr = ("epoch,train_loss,val_loss,train_rrmse_t,val_rrmse_t_stage,val_rrmse_t_refit,"
           "val_rrmse_s_refit,val_cc_refit,val_kappa,val_flag_rate,lr,seconds")
    print(hdr.replace(",", "  "), flush=True)
    rows, best, t0 = [], float("inf"), time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        Y, X, A, _ = ds.train_epoch()
        if args.max_train:
            Y, X, A = Y[:args.max_train], X[:args.max_train], A[:args.max_train]
        perm = torch.randperm(len(Y))
        tl, tr, nb = 0.0, 0.0, 0
        for i in range(0, len(Y), args.batch):
            idx = perm[i:i + args.batch]
            y, x, a = Y[idx].unsqueeze(1), X[idx].unsqueeze(1), A[idx].unsqueeze(1)
            Gx, Ga = model(y)
            xh, ah = model.reconstruct(Gx, Ga)
            loss, _, _ = icstu_loss(model, xh, ah, x, a, args.beta, args.gamma_c, args.gamma_g)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            model.renorm()
            tl += loss.item() * len(idx)
            tr += rrmse_t(xh.detach(), x).sum().item()
            nb += len(idx)
        sched.step()
        v = evaluate(model, ds.fixed["val"], args.kappa_min, args.beta, args.gamma_c, args.gamma_g)
        row = dict(epoch=ep, train_loss=tl / nb, val_loss=v["loss"], train_rrmse_t=tr / nb,
                   val_rrmse_t_stage=v["rt_stage"], val_rrmse_t_refit=v["rt_refit"],
                   val_rrmse_s_refit=v["rs_refit"], val_cc_refit=v["cc_refit"],
                   val_kappa=v["kappa"], val_flag_rate=v["flag_rate"],
                   lr=sched.get_last_lr()[0], seconds=time.time() - t0)
        rows.append(row)
        print(f"{ep:3d}  {row['train_loss']:.5f}  {row['val_loss']:.5f}  {row['train_rrmse_t']:.4f}  "
              f"{row['val_rrmse_t_stage']:.4f}  {row['val_rrmse_t_refit']:.4f}  {row['val_rrmse_s_refit']:.4f}  "
              f"{row['val_cc_refit']:.4f}  {row['val_kappa']:.3f}  {row['val_flag_rate']:.3f}  "
              f"{row['lr']:.2e}  {row['seconds']:.0f}s", flush=True)
        if v["rt_refit"] < best:
            best = v["rt_refit"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

    # ---- history files
    with open(os.path.join(args.out_dir, "history.csv"), "w") as f:
        f.write(hdr + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in hdr.split(",")) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [r["epoch"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(ep, [r["train_loss"] for r in rows], label="train")
        ax[0].plot(ep, [r["val_loss"] for r in rows], label="validation")
        ax[0].set(xlabel="epoch", ylabel="loss (Eq. 24)", yscale="log"); ax[0].legend()
        ax[1].plot(ep, [r["train_rrmse_t"] for r in rows], label="train")
        ax[1].plot(ep, [r["val_rrmse_t_stage"] for r in rows], label="validation (stage output)")
        ax[1].plot(ep, [r["val_rrmse_t_refit"] for r in rows], label="validation (refit)")
        ax[1].set(xlabel="epoch", ylabel="RRMSEt"); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "history.png"), dpi=150)
    except Exception as e:
        print("plot skipped:", e)

    # ---- test with the best-validation checkpoint
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best.pt")))
    t = evaluate(model, ds.fixed["test"], args.kappa_min, args.beta, args.gamma_c, args.gamma_g)
    seg = t.pop("_per_seg")
    from scipy.stats import spearmanr
    rho = spearmanr(1 / seg["kappa"].numpy(), seg["rt_refit"].numpy())[0]
    t["spearman_err_vs_1_over_kappa"] = float(rho)
    t["parameters"] = nparam
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(dict(args=vars(args), test=t, best_val_rrmse_t_refit=best), f, indent=2)
    print("\nTEST (best-validation checkpoint):")
    for k, val in t.items():
        print(f"  {k:32s} {val:.4f}" if isinstance(val, float) else f"  {k:32s} {val}")


if __name__ == "__main__":
    main()
