# Patches applied to run on Python 3.12 / gym 0.26 / numpy 2.x (Sept 2026)

1. marlgrid/base.py, seed(): `gym.utils.seeding.np_random(seed)` now returns a
   numpy Generator (no `.randint`). Replaced with `np.random.RandomState(seed)`.
2. Replaced removed numpy aliases `np.int`, `np.bool`, `np.float` with builtins
   in marlgrid/*.py and marlgrid/envs/*.py.
3. Headless machines: run with `PYGLET_HEADLESS=1` (rendering imports pyglet.gl
   at module import time).

Install:  pip install -e .   then  pip install pyglet
Run:      PYGLET_HEADLESS=1 python examples/toy_random_agents.py

NOTE: this repo contains ONLY the environment (Goal Cycle, Cluttered, DoorKey,
Empty). The paper's training code (PPO + LSTM + auxiliary next-state loss,
expert training curriculum, imitation baseline, Four Rooms and Maze transfer
envs) is NOT released here.
