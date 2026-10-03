"""Shared helpers for the Push-T "when does the robot decide?" project.

Every script imports from here, so the model loading, game setup and
"sample N plans from one moment" logic is written only once.
"""

import os
import subprocess
import sys

import gymnasium as gym
import gym_pusht  # noqa: F401  (importing it registers the "gym_pusht/PushT-v0" game)
import numpy as np
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import OBS_IMAGES

DEVICE = "cpu"
HUB_MODEL = "lerobot/diffusion_pusht"
MIGRATED_DIR = "pusht_migrated"
DATA_DIR = "data"
FIGURES_DIR = "figures"


def migrate_if_needed():
    """Convert the old-format model settings once. Skips if already converted.

    Known quirk (lerobot 0.6.1): the converter saves the pre/postprocessor files
    and then crashes while saving the model itself. We only need the processor
    files (weights come from the Hub), so the crash is fine as long as they exist.
    """
    marker = os.path.join(MIGRATED_DIR, "policy_preprocessor.json")
    if os.path.exists(marker):
        return
    print(f"Converting {HUB_MODEL} settings into ./{MIGRATED_DIR} (one time only)...")
    subprocess.run(
        [
            sys.executable, "-m", "lerobot.processor.migrate_policy_normalization",
            "--pretrained-path", HUB_MODEL,
            "--output-dir", MIGRATED_DIR,
        ],
        capture_output=True,
    )
    if not os.path.exists(marker):
        raise RuntimeError(f"Conversion failed: {marker} was not created")


def load_policy():
    """Return (policy, preprocess, postprocess), ready to run on the CPU."""
    migrate_if_needed()
    policy = DiffusionPolicy.from_pretrained(HUB_MODEL)
    policy.to(DEVICE)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=MIGRATED_DIR,
        preprocessor_overrides={"device_processor": {"device": DEVICE}},
    )
    return policy, preprocess, postprocess


def make_env():
    return gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        visualization_width=384,
        visualization_height=384,
        max_episode_steps=300,
    )


def prepare_obs(raw_obs, preprocess):
    """Raw game observation (numpy) -> normalized tensors the policy understands."""
    batch = preprocess_observation(raw_obs)
    batch["task"] = [""]
    return preprocess(batch)


def sample_plans(policy, preprocess, postprocess, prev_raw_obs, cur_raw_obs, n):
    """Ask the policy for n independent plans from the same moment.

    Returns a numpy array of shape (n, 8, 2): n plans, 8 steps each, (x, y)
    target positions in game coordinates.
    """
    def repeated(raw_obs):
        b = prepare_obs(raw_obs, preprocess)
        b = {k: v.repeat(n, *[1] * (v.ndim - 1)) for k, v in b.items() if isinstance(v, torch.Tensor)}
        b[OBS_IMAGES] = torch.stack([b[k] for k in policy.config.image_features], dim=-4)
        return b

    prev_b, cur_b = repeated(prev_raw_obs), repeated(cur_raw_obs)
    policy.reset()
    populate_queues(policy._queues, prev_b)  # fills the 2-frame memory with the previous moment...
    populate_queues(policy._queues, cur_b)   # ...then pushes the current moment in: [prev, cur]
    with torch.no_grad():
        actions = policy.predict_action_chunk(cur_b)  # (n, 8, 2), still normalized
    actions = postprocess(actions)  # back to game coordinates
    return actions.cpu().numpy()
