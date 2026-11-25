"""
LUMO healthy-vs-damaged anomaly detection (leave-one-zip-out).

Expected folder layout under --root (one folder per extracted zip):
  <root>/exemplary_datasets_dam3_111/05_Healthy/*.mat
  <root>/exemplary_datasets_dam3_111/06_DAM3_111/*.mat
  <root>/exemplary_datasets_dam4_111/...   <root>/exemplary_datasets_dam6_111/...
  <root>/extra/Healthy/*.mat               (optional: other healthy files, used only for training)

Label  = parent folder name  ('healthy' -> 0, contains 'dam' -> 1)
Group  = 'damN_xxx' found in the path (e.g. dam3_111); files with no such tag -> 'extra' (train only)

Usage:  python lumo_pipeline.py --root D:\\lumo --out results --jobs 4 [--mpu] [--win 60]
"""
import argparse, os, re, glob, zlib, time
import numpy as np, pandas as pd, scipy.io as sio
from scipy.signal import resample_poly, welch
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

FS_OUT = 100
BANDS = [(1, 3), (3, 6), (6, 10), (10, 13), (13, 15), (15, 17), (17, 20), (20, 30), (30, 45)]
PEAKS = [(2, 4), (12, 15), (15, 18)]          # tower modes seen near 2.8, 13.4, 16.0/16.3 Hz
CHANNELS = ("accel01x", "accel01y")           # top of tower, closest to one MPU6050
MPU_NOISE = 400e-6                            # g/sqrt(Hz), datasheet typical -- verify in your datasheet
MPU_LSB = 1 / 16384                           # +-2 g range


def win_shape(w):
    f, P = welch(w, fs=FS_OUT, nperseg=2048, noverlap=1024)
    df = f[1] - f[0]
    tot = P[(f >= 1) & (f < 45)].sum() * df
    bp = [np.log10(P[(f >= a) & (f < b)].sum() * df / tot + 1e-12) for a, b in BANDS]   # amplitude-free spectral shape
    pk = []
    for a, b in PEAKS:
        m = np.where((f >= a) & (f <= b))[0]
        i = m[P[m].argmax()]
        delta = 0.0
        if 0 < i < len(P) - 1:
            y0, y1, y2 = np.log(P[i - 1] + 1e-30), np.log(P[i] + 1e-30), np.log(P[i + 1] + 1e-30)
            den = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        pk.append(f[i] + delta * df)
    return bp + pk


def extract(path, label, group, win_s, mpu):
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["Dat"]
    names = [str(n) for n in d.ChannelNames]
    T = float(d.Data[:, names.index("temp01")].mean())
    rng = np.random.default_rng(zlib.crc32(path.encode()))
    n = FS_OUT * win_s
    per_ch, rms = [], None
    for c in CHANNELS:
        x = d.Data[:, names.index(c)].astype(np.float64)
        x = resample_poly(x - x.mean(), 1000, int(round(d.Fs / FS_OUT * 1000)))
        if mpu:
            x = x + rng.normal(0, MPU_NOISE * np.sqrt(FS_OUT / 2), x.size)
            x = np.round(x / MPU_LSB) * MPU_LSB
        nw = len(x) // n
        per_ch.append(np.array([win_shape(x[i * n:(i + 1) * n]) for i in range(nw)]))
        if rms is None:
            rms = np.array([x[i * n:(i + 1) * n].std() for i in range(nw)])
    X = np.hstack(per_ch)
    return dict(path=path, label=label, group=group, T=T, X=X, rms=rms,
                dur=d.Data.shape[0] / d.Fs)


def discover(root):
    items = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)):
        if "meta" in os.path.basename(p).lower():
            continue
        rel = os.path.relpath(p, root).lower()
        parent = os.path.basename(os.path.dirname(p)).lower()
        if "healthy" in parent:
            label = 0
        elif "dam" in parent:
            label = 1
        else:
            print("skipping (cannot infer label from folder):", p)
            continue
        m = re.search(r"dam\d_\d{3}", rel) or re.search(r"dam\d", rel)
        items.append((p, label, m.group(0) if m else "extra"))
    return items


def stack(files):
    X = np.vstack([f["X"] for f in files])
    rms = np.concatenate([f["rms"] for f in files])
    fid = np.concatenate([[i] * len(f["X"]) for i, f in enumerate(files)])
    y = np.concatenate([[f["label"]] * len(f["X"]) for f in files])
    return X, rms, fid, y


def file_flag(score, fid, thr):
    return np.array([(score[fid == i] > thr).mean() >= 0.5 for i in np.unique(fid)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--win", type=int, default=60, help="window length in seconds")
    ap.add_argument("--mpu", action="store_true", help="add MPU6050-like noise + quantization")
    ap.add_argument("--pct", type=float, default=99, help="threshold percentile of healthy scores")
    ap.add_argument("--min-train", type=int, default=5, help="min healthy training files per fold")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    items = discover(a.root)
    print("found %d files" % len(items))
    t0 = time.time()
    files = Parallel(n_jobs=a.jobs)(delayed(extract)(p, l, g, a.win, a.mpu) for p, l, g in items)
    print("features extracted in %.1f s (mpu noise: %s, window %d s)" % (time.time() - t0, a.mpu, a.win))
    short = [f for f in files if f["dur"] < 590]
    if short:
        print("WARNING: files shorter than ~10 min:", [(os.path.basename(f["path"]), round(f["dur"])) for f in short])

    df = pd.DataFrame([dict(file=os.path.basename(f["path"]), group=f["group"], label=f["label"],
                            T=f["T"], windows=len(f["X"]), rms_mg=1e3 * f["rms"].mean()) for f in files])
    print("\nfiles per group / label:\n", df.groupby(["group", "label"]).size().unstack(fill_value=0))
    print("\nmean temperature (C) and RMS (mg) per group/label:\n",
          df.groupby(["group", "label"])[["T", "rms_mg"]].mean().round(2))
    df.to_csv(os.path.join(a.out, "file_summary.csv"), index=False)

    # save every window's features so results can be re-analysed without re-running extraction
    cols = ["f%02d" % i for i in range(files[0]["X"].shape[1])]
    wrows = []
    for f in files:
        for i in range(len(f["X"])):
            wrows.append([os.path.basename(f["path"]), f["group"], f["label"], f["T"], i, f["rms"][i], *f["X"][i]])
    pd.DataFrame(wrows, columns=["file", "group", "label", "T", "win", "rms"] + cols).to_csv(
        os.path.join(a.out, "features_mpu%s_win%d.csv" % ("1" if a.mpu else "0", a.win)), index=False)

    test_groups = sorted({f["group"] for f in files if f["group"] != "extra" and f["label"] == 1})
    rows = []
    for g in test_groups:
        te = [f for f in files if f["group"] == g]
        tr = [f for f in files if f["group"] != g and f["label"] == 0]
        if len(tr) < a.min_train or not any(f["label"] == 0 for f in te):
            print("skip fold %s (train files=%d)" % (g, len(tr)))
            continue
        Xt, rt, fidt, yt = stack(te)

        def fit_models(files):
            X, _, _, _ = stack(files)
            mu, sd = X.mean(0), X.std(0) + 1e-9
            lw = LedoitWolf().fit((X - mu) / sd)
            iso = IsolationForest(n_estimators=200, random_state=0).fit((X - mu) / sd)
            return {
                "RMS threshold (fixed, upper)": (lambda X, r: r),
                "Mahalanobis (spectral shape)": (lambda X, r: np.sqrt(lw.mahalanobis((X - mu) / sd))),
                "IsolationForest (spectral shape)": (lambda X, r: -iso.score_samples((X - mu) / sd)),
            }

        # threshold from ALL training files via file-level cross-fitting (not just 30% of them)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(tr))
        K = min(5, len(tr))
        pooled = {}
        for kf in range(K):
            held = [tr[i] for j, i in enumerate(order) if j % K == kf]
            rest = [tr[i] for j, i in enumerate(order) if j % K != kf]
            Xh, rh, _, _ = stack(held)
            for name, fn in fit_models(rest).items():
                pooled.setdefault(name, []).append(fn(Xh, rh))
        thresholds = {n: np.percentile(np.concatenate(v), a.pct) for n, v in pooled.items()}
        scorers = fit_models(tr)
        for name, fn in scorers.items():
            thr = thresholds[name]
            s = fn(Xt, rt)
            fa = (s[yt == 0] > thr).mean(); dr = (s[yt == 1] > thr).mean()
            auc = roc_auc_score(yt, s) if len(set(yt)) == 2 else np.nan
            fl = file_flag(s, fidt, thr)
            fl_lab = np.array([f["label"] for f in te])
            rows.append(dict(heldout=g, method=name, train_files=len(tr), test_healthy_files=int((fl_lab == 0).sum()),
                             test_damaged_files=int((fl_lab == 1).sum()), window_FA=fa, window_detect=dr, window_AUC=auc,
                             files_false_alarm="%d/%d" % (fl[fl_lab == 0].sum(), (fl_lab == 0).sum()),
                             files_detected="%d/%d" % (fl[fl_lab == 1].sum(), (fl_lab == 1).sum())))
    if rows:
        res = pd.DataFrame(rows)
        pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
        print("\nRESULTS (threshold = percentile %g of cross-validated healthy scores)" % a.pct + ":")
        print(res.round(3).to_string(index=False))
        res.to_csv(os.path.join(a.out, "results_mpu%s_win%d.csv" % ("1" if a.mpu else "0", a.win)), index=False)
    else:
        print("\nNo evaluable folds (need damaged folders plus enough healthy training files).")


if __name__ == "__main__":
    main()
