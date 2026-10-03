"""Step 3: sample many plans from step 80 and sort them into two routes.

Usage:  python 03_sample_plans_step80.py --n 32
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pusht_common import DATA_DIR, FIGURES_DIR, load_policy, sample_plans

STEP = 80
ROUTE_COLORS = ["#2a78d6", "#eb6834"]  # blue = route A (bigger), orange = route B
GAME_TO_PICTURE = 384 / 512  # game board is 512x512, the saved picture is 384x384

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=32, help="number of plans to sample")
parser.add_argument("--seed", type=int, default=0, help="random seed for the denoising noise")
args = parser.parse_args()


def load_obs(step):
    d = np.load(os.path.join(DATA_DIR, f"seed0_step{step:03d}_obs.npz"))
    return {"pixels": d["pixels"], "agent_pos": d["agent_pos"]}, d["render"]


prev_obs, _ = load_obs(STEP - 1)
cur_obs, picture = load_obs(STEP)

policy, preprocess, postprocess = load_policy()
torch.manual_seed(args.seed)
plans = sample_plans(policy, preprocess, postprocess, prev_obs, cur_obs, args.n)  # (n, 8, 2)

# --- Sort plans into 2 routes by where each plan ends ---
endpoints = plans[:, -1, :]
kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(endpoints)
labels = kmeans.labels_
if (labels == 0).sum() < (labels == 1).sum():  # make route A (label 0) the bigger one
    labels = 1 - labels
sil = silhouette_score(endpoints, labels)

print(f"\nStep {STEP}, {args.n} plans (seed {args.seed})")
for r, name in enumerate(["A", "B"]):
    members = labels == r
    cx, cy = endpoints[members].mean(axis=0)
    print(f"  Route {name}: {members.sum():3d} plans = {100 * members.mean():5.1f}%   "
          f"average endpoint (x={cx:.0f}, y={cy:.0f})")
print(f"  Silhouette score: {sil:.2f}  (near 1 = two clearly separate routes, near 0 = one blob)")
print(f"  Robot is at (x={cur_obs['agent_pos'][0]:.0f}, y={cur_obs['agent_pos'][1]:.0f})")

# --- Save the numbers ---
os.makedirs(DATA_DIR, exist_ok=True)
out = os.path.join(DATA_DIR, f"seed0_step{STEP:03d}_plans_n{args.n}_seed{args.seed}.npz")
np.savez_compressed(out, plans=plans, route_labels=labels, silhouette=sil, step=STEP, sample_seed=args.seed)
print(f"Saved {out}")

# --- Draw the picture ---
fig, ax = plt.subplots(figsize=(6, 6.4))
ax.imshow(picture, alpha=0.55)
start = cur_obs["agent_pos"] * GAME_TO_PICTURE
for plan, label in zip(plans, labels):
    path = np.vstack([cur_obs["agent_pos"], plan]) * GAME_TO_PICTURE
    ax.plot(path[:, 0], path[:, 1], color=ROUTE_COLORS[label], lw=1.2, alpha=0.5)
    ax.plot(path[-1, 0], path[-1, 1], "o", color=ROUTE_COLORS[label], ms=4, alpha=0.8)
ax.plot(*start, "o", ms=9, mfc="white", mec="#222222", mew=2, label="robot now")
for r, name in enumerate(["A", "B"]):
    pct = 100 * (labels == r).mean()
    ax.plot([], [], color=ROUTE_COLORS[r], lw=2, label=f"Route {name}: {pct:.0f}%  ({(labels == r).sum()} plans)")
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
ax.set_title(f"Push-T, env seed 0, step {STEP}: {args.n} sampled plans\n"
             f"colored by KMeans route on plan endpoints (silhouette {sil:.2f})", fontsize=10)
ax.set_axis_off()
os.makedirs(FIGURES_DIR, exist_ok=True)
fig_path = os.path.join(FIGURES_DIR, f"seed0_step{STEP:03d}_routes_n{args.n}.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved {fig_path}")
