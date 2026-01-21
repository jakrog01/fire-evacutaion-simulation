import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--start_thr", type=float, default=0.05)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    t = d["t"]
    curves = d["curves"]
    roi_names = d["roi_names"]

    max_vals = curves.max(axis=0)

    print("ROI max burn_ratio:")
    order = np.argsort(-max_vals)
    for i in order:
        print(f"{roi_names[i]}  max={max_vals[i]:.3f}")

    print("")
    print(f"First time burn_ratio >= {args.start_thr} (START proxy):")
    for i in range(curves.shape[1]):
        idx = np.argmax(curves[:, i] >= args.start_thr)
        if curves[idx, i] >= args.start_thr:
            print(f"{roi_names[i]}  t={t[idx]:.2f}s  val={curves[idx, i]:.3f}")
        else:
            print(f"{roi_names[i]}  never")


if __name__ == "__main__":
    main()
