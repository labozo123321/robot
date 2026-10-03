"""Step 4: scan many moments of the recorded seed-0 game to find where plans split.

For each step we sample a few plans, split their endpoints into 2 groups with
KMeans, and measure how far apart the two groups are. One route -> the groups
are a few pixels apart. Two real routes -> they are far apart.

Usage:  python 04_scan_for_splits.py --first 65 --last 95 --n 16
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

from pusht_common import DATA_DIR, load_policy, sample_plans

parser = argparse.ArgumentParser()
parser.add_argument("--first", type=int, default=65)
parser.add_argument("--last", type=int, default=95)
parser.add_argument("--n", type=int, default=16, help="plans per step")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

episode = np.load(os.path.join(DATA_DIR, "seed0_episode_steps000-100.npz"))


def obs_at(step):
    return {"pixels": episode["pixels"][step], "agent_pos": episode["agent_pos"][step]}


policy, preprocess, postprocess = load_policy()
csv_path = os.path.join(DATA_DIR, f"seed0_scan_steps{args.first:03d}-{args.last:03d}_n{args.n}.csv")
plans_path = csv_path.replace(".csv", "_plans.npz")
all_plans = {}

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "gap_px", "smaller_group_pct", "spread_px"])
    print(f"{'step':>4}  {'gap(px)':>7}  {'smaller group':>13}  {'spread(px)':>10}")
    for step in range(args.first, args.last + 1):
        t0 = time.perf_counter()
        torch.manual_seed(args.seed)
        plans = sample_plans(policy, preprocess, postprocess, obs_at(step - 1), obs_at(step), args.n)
        ends = plans[:, -1, :]
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(ends)
        gap = np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1])  # distance between group centers
        smaller = 100 * min(km.labels_.mean(), 1 - km.labels_.mean())
        spread = ends.std(axis=0).mean()  # how scattered all endpoints are
        writer.writerow([step, f"{gap:.1f}", f"{smaller:.0f}", f"{spread:.1f}"])
        f.flush()
        all_plans[f"step{step:03d}"] = plans
        np.savez_compressed(plans_path, **all_plans)  # save as we go, so a crash loses nothing
        flag = "  <-- possible split" if gap > 30 and smaller >= 15 else ""
        print(f"{step:>4}  {gap:7.1f}  {smaller:12.0f}%  {spread:10.1f}   ({time.perf_counter() - t0:.0f}s){flag}",
              flush=True)

print(f"Saved {csv_path} and {plans_path}")
