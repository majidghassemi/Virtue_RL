"""Break collect() down: rendering vs expert BFS vs env logic vs torch.

  cd sociapl && python profile_collect.py
  python profile_collect.py --mode solo
  python profile_collect.py --cprofile      # full cProfile table

Phase 0 of the speedup work: says how much each optimisation is actually worth
before any of them are written. Diagnostic only -- changes nothing.
"""
import argparse, time, collections
import numpy as np, torch
from ethics import EthicsWorker
from model import SociAPLNet
from ppo import collect, HP
import marlgrid.base as mbase
import envs as envs_mod

COUNT = collections.Counter()
TIME = collections.Counter()


def instrument():
    """Wrap the hot functions with counters. Restores nothing -- one-shot process."""
    orig_gao = mbase.MultiGridEnv.gen_agent_obs
    def gen_agent_obs(self, agent):
        t = time.perf_counter()
        r = orig_gao(self, agent)
        TIME["render"] += time.perf_counter() - t; COUNT["render"] += 1
        return r
    mbase.MultiGridEnv.gen_agent_obs = gen_agent_obs

    orig_step = mbase.MultiGridEnv.step
    def step(self, actions):
        t = time.perf_counter()
        r = orig_step(self, actions)
        TIME["env_step_total"] += time.perf_counter() - t; COUNT["env_step"] += 1
        return r
    mbase.MultiGridEnv.step = step

    orig_act = envs_mod.ScriptedExpert.act
    def act(self):
        t = time.perf_counter()
        r = orig_act(self)
        TIME["expert_bfs"] += time.perf_counter() - t; COUNT["expert_bfs"] += 1
        return r
    envs_mod.ScriptedExpert.act = act

    orig_net_act = SociAPLNet.act
    def net_act(self, *a, **k):
        t = time.perf_counter()
        r = orig_net_act(self, *a, **k)
        TIME["net_act"] += time.perf_counter() - t; COUNT["net_act"] += 1
        return r
    SociAPLNet.act = net_act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="social", choices=["solo", "social"])
    p.add_argument("--virtuous", type=int, default=1)
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cprofile", action="store_true")
    a = p.parse_args()

    torch.set_num_threads(a.threads); torch.manual_seed(0); np.random.seed(0)
    instrument()
    workers = [EthicsWorker(a.mode, 2, 3, bool(a.virtuous), 0.0, 0.25, seed=i,
                            harm_delivery="none", harm_lambda=1.0, n_harm_tiles=6)
               for i in range(a.n_envs)]
    net = SociAPLNet(aux="pred")
    obs = np.stack([w.reset() for w in workers]); h, c = net.init_state(a.n_envs)

    stats = []
    if a.cprofile:
        import cProfile, pstats
        pr = cProfile.Profile(); pr.enable()
    t0 = time.perf_counter()
    collect(net, workers, a.episodes, obs, h, c, stats)
    total = time.perf_counter() - t0
    if a.cprofile:
        pr.disable(); pstats.Stats(pr).sort_stats("cumulative").print_stats(25)

    # env_step_total includes the renders step() does internally; separate them.
    render_in_step = TIME["render"] * (COUNT["env_step"] * len(workers[0].env.agents)
                                       / max(1, COUNT["render"]))
    print(f"\nmode={a.mode} n_envs={a.n_envs} episodes={len(stats)} threads={a.threads}")
    print(f"collect total: {total:.1f}s\n")
    print(f"  {'component':<22} {'time':>8} {'%':>6} {'calls':>10} {'us/call':>9}")
    for k, label in [("render", "observation render"), ("expert_bfs", "expert BFS"),
                     ("net_act", "net.act (torch)"), ("env_step_total", "env.step (incl render)")]:
        t, n = TIME[k], COUNT[k.replace("_total", "")] or COUNT[k]
        print(f"  {label:<22} {t:>7.1f}s {100*t/total:>5.1f}% {n:>10,} "
              f"{1e6*t/max(1,n):>8.1f}")
    other = total - TIME["env_step_total"] - TIME["net_act"]
    print(f"  {'other (collect glue)':<22} {other:>7.1f}s {100*other/total:>5.1f}%")
    print()
    n_ag = len(workers[0].env.agents)
    needed = COUNT["env_step"]           # one render per env-step is all the learner needs
    print(f"  renders: {COUNT['render']:,} performed, ~{needed:,} needed "
          f"({COUNT['render']/max(1,needed):.1f}x) -- {n_ag} agents, rendered twice per step")
    print(f"  if renders drop to 1x: collect ~= "
          f"{total - TIME['render'] * (1 - needed/max(1,COUNT['render'])):.1f}s "
          f"({total / max(0.01, total - TIME['render'] * (1 - needed/max(1,COUNT['render']))):.2f}x)")


if __name__ == "__main__":
    main()
