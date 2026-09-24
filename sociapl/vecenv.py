"""Parallel env stepping: split the workers (envs.Worker / ethics.EthicsWorker) across
subprocesses so rollout collection uses all CPU cores of the job.

Each subprocess owns a fixed slice of workers and steps them serially, so every worker
sees exactly the same RNG stream and action sequence as in-process stepping: results
are identical to SerialVec, only faster.

Create the vector env BEFORE initialising CUDA (fork after CUDA init is unsafe); the
training scripts do this. Children never import or use torch.
"""
import multiprocessing as mp
import os
import sys
import numpy as np


def _step_slice(workers, actions):
    """Step each worker; auto-reset finished ones.
    Returns true_next (pre-reset frame, aux target), next_obs (post-reset), rew, done, infos."""
    true_next, next_obs, rew, done, infos = [], [], [], [], []
    for w, a in zip(workers, actions):
        o, r, d, info = w.step(a)
        true_next.append(o)
        if d:
            o = w.reset()
        next_obs.append(o); rew.append(r); done.append(d); infos.append(info if d else None)
    return (np.stack(true_next), np.stack(next_obs), np.asarray(rew, np.float32),
            np.asarray(done, bool), infos)


class SerialVec:
    """In-process reference implementation."""

    def __init__(self, workers):
        self.workers = list(workers)
        self.n = len(self.workers)

    def reset(self):
        return np.stack([w.reset() for w in self.workers])

    def step(self, actions):
        return _step_slice(self.workers, actions)

    def close(self):
        pass


def _proc_main(conn, factory, args_list):
    workers = [factory(*a, **kw) for a, kw in args_list]
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                conn.send(_step_slice(workers, data))
            elif cmd == "reset":
                conn.send(np.stack([w.reset() for w in workers]))
            elif cmd == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        conn.close()


class SubprocVec:
    """factory(*args, **kwargs) builds one worker; specs = [(args, kwargs), ...] per env."""

    def __init__(self, factory, specs, n_procs):
        self.n = len(specs)
        n_procs = max(1, min(n_procs, self.n))
        bounds = np.linspace(0, self.n, n_procs + 1).astype(int)
        self.slices = [(bounds[i], bounds[i + 1]) for i in range(n_procs)]
        ctx = mp.get_context("fork" if sys.platform == "linux" else "spawn")
        self.conns, self.procs = [], []
        for lo, hi in self.slices:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_proc_main, args=(child, factory, specs[lo:hi]), daemon=True)
            p.start(); child.close()
            self.conns.append(parent); self.procs.append(p)

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, actions):
        actions = np.asarray(actions)
        for c, (lo, hi) in zip(self.conns, self.slices):
            c.send(("step", actions[lo:hi]))
        parts = [c.recv() for c in self.conns]
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]),
                [i for p in parts for i in p[4]])

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def make_vec(factory, specs, n_procs):
    """n_procs <= 1: in-process; otherwise that many subprocesses."""
    if n_procs <= 1:
        return SerialVec([factory(*a, **kw) for a, kw in specs])
    return SubprocVec(factory, specs, n_procs)


def auto_procs(n_envs, requested=-1):
    """requested < 0: one process per env, capped at (usable CPUs - 1) so the learner keeps a core."""
    if requested >= 0:
        return requested
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    return max(1, min(n_envs, cpus - 1))
