"""Step 1: check the policy runs and time how long plans take on this computer.

Plays the game for a few steps to get two real consecutive moments, then times
generating 1 plan and 8 plans from that moment.
"""

import time

import torch

from pusht_common import load_policy, make_env, sample_plans

torch.manual_seed(0)

t0 = time.perf_counter()
policy, preprocess, postprocess = load_policy()
print(f"Loading the model took {time.perf_counter() - t0:.1f} s")
print(f"Denoising rounds per plan: {policy.diffusion.num_inference_steps}")

# Get two consecutive real observations (steps 0 and 1).
env = make_env()
obs0, _ = env.reset(seed=0)
obs1, *_ = env.step(env.action_space.sample())

sample_plans(policy, preprocess, postprocess, obs0, obs1, 1)  # warm-up run (first run is always slower)

for n in [1, 8, 32]:
    times = []
    for _ in range(2):
        t0 = time.perf_counter()
        plans = sample_plans(policy, preprocess, postprocess, obs0, obs1, n)
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"{n:3d} plan(s): {best:.2f} s  ({best / n:.3f} s per plan)  output shape {plans.shape}")

print(f"torch is using {torch.get_num_threads()} CPU threads")
