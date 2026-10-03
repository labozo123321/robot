"""Step 2: play one game (env seed 0) and save the observations at steps 79 and 80.

"Step t" = the observation the game shows after the robot has made t moves
(step 0 = the starting picture right after reset).
"""

import os

import numpy as np
import torch

from pusht_common import DATA_DIR, load_policy, make_env, prepare_obs

ENV_SEED = 0
TORCH_SEED = 0  # the policy's random noise also decides how the game goes
LAST_STEP = 100  # we only need the game up to a bit past step 80
SAVE_STEPS = [79, 80]

torch.manual_seed(TORCH_SEED)
policy, preprocess, postprocess = load_policy()
env = make_env()

policy.reset()
obs, _ = env.reset(seed=ENV_SEED)
pixels, agent_pos, actions, renders = [], [], [], []

for step in range(LAST_STEP + 1):
    pixels.append(obs["pixels"])
    agent_pos.append(obs["agent_pos"])
    renders.append(env.render())  # 384x384 picture of the game at this moment
    if step == LAST_STEP:
        break
    with torch.no_grad():
        action = postprocess(policy.select_action(prepare_obs(obs, preprocess)))
    action = action.squeeze(0).cpu().numpy()
    actions.append(action)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        print(f"Game ended early at step {step + 1}")
        break

os.makedirs(DATA_DIR, exist_ok=True)
for step in SAVE_STEPS:
    path = os.path.join(DATA_DIR, f"seed{ENV_SEED}_step{step:03d}_obs.npz")
    np.savez_compressed(path, pixels=pixels[step], agent_pos=agent_pos[step], render=renders[step], step=step)
    print(f"Saved {path}   agent_pos={agent_pos[step]}")

path = os.path.join(DATA_DIR, f"seed{ENV_SEED}_episode_steps000-{LAST_STEP:03d}.npz")
np.savez_compressed(path, pixels=np.array(pixels), agent_pos=np.array(agent_pos), actions=np.array(actions),
                    renders=np.array(renders),
                    env_seed=ENV_SEED, torch_seed=TORCH_SEED)
print(f"Saved {path}")
