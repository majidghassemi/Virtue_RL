"""Environment construction and scripted experts for the SociAPL replication.

The learner is always agent index 0. Experts (if any) are agents 1..n and are
driven by ScriptedExpert, which has privileged access to the goal order.
"""
import os
os.environ.setdefault("PYGLET_HEADLESS", "1")
import warnings
warnings.filterwarnings("ignore")

from collections import deque
import numpy as np
from marlgrid.envs import env_from_config
from marlgrid.objects import BonusTile

AGENT_CFG = dict(view_size=7, view_offset=1, view_tile_size=3,
                 observation_style="image", see_through_walls=False, color="prestige")
# 7 tiles * 3 px = 21x21x3 observation -> conv stack flattens to 576, matching the paper's FC(576,192)

N_ACTIONS = 7  # marlgrid action space; only left/right/forward matter in Goal Cycle


def make_env(n_experts=2, n_goals=3, grid_size=13, max_steps=250, penalty=-1.5,
             clutter_density=0.15, seed=0):
    cfg = {
        "env_class": "ClutteredGoalCycleEnv", "grid_size": grid_size, "max_steps": max_steps,
        "clutter_density": clutter_density, "respawn": True, "ghost_mode": True,
        "reward_decay": False, "n_bonus_tiles": n_goals, "initial_reward": True,
        "penalty": penalty, "agents": [dict(AGENT_CFG) for _ in range(1 + n_experts)],
    }
    env = env_from_config(cfg, randomize_seed=False)
    env.seed(seed)
    return env


# direction index -> (dx, dy), matching GridAgentInterface.dir_vec
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class ScriptedExpert:
    """Privileged expert: knows the bonus-tile order, BFS-paths to the next correct tile.

    epsilon > 0 gives an imperfect expert (random action with prob epsilon),
    matching the paper's 'sub-optimal experts' condition qualitatively.
    """

    def __init__(self, env, agent_idx, epsilon=0.0, rng=None):
        self.env = env
        self.agent = env.agents[agent_idx]
        self.epsilon = epsilon
        self.rng = rng or np.random.default_rng()

    def _bonus_tiles(self):
        g = self.env.grid
        out = {}
        for i in range(g.width):
            for j in range(g.height):
                o = g.get(i, j)
                if isinstance(o, BonusTile):
                    out[o.bonus_id] = (i, j)
        return out

    def _passable(self, i, j, target_id, tiles):
        g = self.env.grid
        if i < 0 or j < 0 or i >= g.width or j >= g.height:
            return False
        o = g.get(i, j)
        if isinstance(o, BonusTile):
            return o.bonus_id == target_id  # stepping on any other tile is a penalty
        return o is None or o.can_overlap()

    def _bfs_first_step(self, start, goal, target_id, tiles):
        """Return the direction index of the first move on a shortest path, or None."""
        start = tuple(start); goal = tuple(goal)
        if start == goal:
            return None
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for d, (dx, dy) in enumerate(DIRS):
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in prev or not self._passable(*nxt, target_id, tiles):
                    continue
                prev[nxt] = (cur, d)
                q.append(nxt)
        if goal not in prev:
            return None
        node = goal
        while prev[node][0] != start:
            node = prev[node][0]
        return prev[node][1]

    def act(self):
        a = self.agent
        if not a.active or a.pos is None:
            return 6  # 'done'/no-op
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 3))
        tiles = self._bonus_tiles()
        n = self.env.n_bonus_tiles
        if a.bonus_state is None:
            # first tile always pays: go to the nearest one
            cands = list(tiles.keys())
        else:
            cands = [(a.bonus_state + 1) % n]
        best = None
        for tid in cands:
            d = self._bfs_first_step(a.pos, tiles[tid], tid, tiles)
            if d is not None:
                # crude nearest choice: manhattan distance tiebreak
                dist = abs(tiles[tid][0] - a.pos[0]) + abs(tiles[tid][1] - a.pos[1])
                if best is None or dist < best[0]:
                    best = (dist, d)
        if best is None:
            return int(self.rng.integers(0, 3))
        want = best[1]
        cur = a.dir
        if want == cur:
            return 2  # forward
        if (cur + 1) % 4 == want:
            return 1  # right
        return 0      # left (also for the 180-degree case: two lefts)


class Worker:
    """Wraps one env. The caller supplies only the learner's action; experts act internally.

    mode: 'solo' (learner alone), 'social' (learner + n_experts), or 'mixed'
    (each episode is social with probability p_social, else solo).
    """

    def __init__(self, mode="social", n_experts=2, n_goals=3, expert_eps=0.0, p_social=0.25, seed=0, **env_kw):
        self.mode = mode
        self.p_social = p_social
        self.rng = np.random.default_rng(seed)
        self.envs = {}
        if mode in ("solo", "mixed"):
            self.envs["solo"] = make_env(0, n_goals, seed=seed, **env_kw)
        if mode in ("social", "mixed"):
            self.envs["social"] = make_env(n_experts, n_goals, seed=seed + 10_000, **env_kw)
        self.expert_eps = expert_eps
        self.env = None
        self.experts = []
        self.ep_return = 0.0
        self.ep_expert_return = 0.0

    def reset(self):
        if self.mode == "mixed":
            key = "social" if self.rng.random() < self.p_social else "solo"
        else:
            key = self.mode
        self.env = self.envs[key]
        obs = self.env.reset()
        self.experts = [ScriptedExpert(self.env, i, self.expert_eps, self.rng) for i in range(1, len(self.env.agents))]
        self.ep_return = 0.0
        self.ep_expert_return = 0.0
        self.is_social = key == "social"
        return np.asarray(obs[0], dtype=np.uint8)

    def step(self, learner_action):
        actions = [int(learner_action)] + [e.act() for e in self.experts]
        obs, rew, done, _ = self.env.step(actions)
        self.ep_return += rew[0]
        if self.experts:
            self.ep_expert_return += float(np.mean(rew[1:]))
        info = {}
        if done:
            info = {"ep_return": self.ep_return, "ep_expert_return": self.ep_expert_return,
                    "social": self.is_social}
        return np.asarray(obs[0], dtype=np.uint8), float(rew[0]), bool(done), info

    def expert_obs_and_actions(self):
        """For behaviour cloning: returns (obs, action) of expert 0 for the *current* state."""
        if not self.experts:
            raise RuntimeError("no experts in this env")
        obs = self.env.gen_obs()
        return np.asarray(obs[1], dtype=np.uint8), self.experts[0].act()
