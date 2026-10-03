# When does a diffusion policy "decide"? (Push-T)

Studying at which denoising round LeRobot's pretrained Diffusion Policy
(`lerobot/diffusion_pusht`) commits to one of several possible routes.

## Setup (CPU)

```bash
python3.12 -m venv .venv
.venv/bin/pip install "lerobot[pusht,diffusion]==0.6.1" scikit-learn matplotlib
```

Network needs `pypi.org`, `files.pythonhosted.org` and `huggingface.co`.

## Scripts (run in order)

| Script | What it does |
|---|---|
| `pusht_common.py` | Shared helpers: load model + processors (auto-converts old settings into `pusht_migrated/`), make the game, sample N plans from one moment. |
| `01_timing_test.py` | Checks the policy runs and times 1 / 8 / 32 plans. |
| `02_play_game_seed0.py` | Plays one game (env seed 0, torch seed 0), saves observations at steps 79 and 80, plus every step 0-100 (with pictures) to `data/`. |
| `03_sample_plans.py --step 80 --n 128` | Samples plans at one step, splits them into 2 routes with KMeans on plan endpoints, saves plans to `data/` and a picture to `figures/`. |
| `04_scan_for_splits.py --first 65 --last 95 --n 16` | For each step, measures the gap between the two KMeans groups' endpoints. Big gap + decent minority share = possible split. |

## Results log

- **Timing (4 CPU threads, 100 denoising rounds):** 1 plan 6.0 s, 8 plans 26.6 s, 32 plans 53 s, 128 plans ~2.5 min.
- **Step 80, env seed 0, torch seed 0:** only ONE route (all 128 plans go up the right
  side of the T; endpoints within a ~17x19 px box). KMeans still splits it 69% / 31%,
  but that is cutting one blob in half, not two routes. The earlier two-route finding
  at step 80 was probably from a game that took a different path (different torch seed).
- **Scan of steps 65-95 (16 plans each):** no split anywhere. Gap between groups is
  1-3 px at most steps, rising to 14-17 px at steps 86-87 (one wider fan around the
  right side of the T, not two routes). See `data/seed0_scan_steps065-095_n16.csv`
  and `figures/seed0_step087_routes_n32.png`.
