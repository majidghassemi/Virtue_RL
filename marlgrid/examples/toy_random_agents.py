import numpy as np
from marlgrid.envs import env_from_config

env_config = {
    "env_class": "ClutteredGoalCycleEnv",
    "grid_size": 13, "max_steps": 250, "clutter_density": 0.15,
    "respawn": True, "ghost_mode": True, "reward_decay": False,
    "n_bonus_tiles": 3, "initial_reward": True, "penalty": -1.5,
}
agent_cfg = {"view_size": 7, "view_offset": 1, "view_tile_size": 3,
             "observation_style": "image", "see_through_walls": False, "color": "prestige"}
env_config["agents"] = [dict(agent_cfg) for _ in range(3)]  # 3 agents like the paper
env = env_from_config(env_config)

rng = np.random.default_rng(0)
for ep in range(3):
    obs = env.reset()
    tot = np.zeros(len(env.agents)); done = False; t = 0
    while not done:
        acts = [rng.integers(0, 3) for _ in env.agents]  # left/right/forward
        obs, rew, done, _ = env.step(acts)
        tot += np.array(rew); t += 1
    print(f"ep {ep}: steps={t} obs_shape={np.asarray(obs[0]).shape} returns={tot}")
