"""Where does the time actually go, and would a GPU help?

  cd sociapl && python profile_speed.py                 # default: social, 8 threads
  python profile_speed.py --mode solo --threads 2
  python profile_speed.py --sweep                       # thread sweep

Diagnostic only -- imports the same collect/update as train_ethics.py and changes
nothing about the experiment.

The split it reports is the whole story for GPU:
  collect() = pure-Python marlgrid env stepping + a tiny no_grad forward.
              A GPU does nothing for this.
  update()  = 20 minibatches of conv/deconv over 8192 samples with backward.
              This is what a GPU would accelerate.
Amdahl caps the achievable speedup at total / collect.
"""
import argparse, time
import numpy as np, torch
from ethics import EthicsWorker
from model import SociAPLNet
from ppo import collect, update, HP


def run_once(mode, virtuous, threads, n_envs, batch_episodes, aux="pred", reps=2):
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(threads)
    workers = [EthicsWorker(mode, 2, 3, bool(virtuous), 0.0, 0.25, seed=i,
                            harm_delivery="none", harm_lambda=1.0, n_harm_tiles=6)
               for i in range(n_envs)]
    net = SociAPLNet(aux=aux)
    opt = torch.optim.Adam(net.parameters(), lr=HP["lr"])
    obs = np.stack([w.reset() for w in workers]); h, c = net.init_state(n_envs)

    t_collect = t_update = 0.0
    episodes = 0
    for _ in range(reps):
        stats = []
        t0 = time.time()
        data, obs, h, c = collect(net, workers, batch_episodes, obs, h, c, stats)
        t1 = time.time()
        update(net, opt, data, aux)
        t2 = time.time()
        t_collect += t1 - t0; t_update += t2 - t1; episodes += len(stats)
    return t_collect, t_update, episodes


def report(label, tc, tu, eps):
    tot = tc + tu
    rate = eps / tot
    # A GPU removes most of update() and none of collect().
    ideal = eps / tc                      # update -> 0
    realistic = eps / (tc + tu * 0.10)    # update 10x faster
    print(f"  {label}")
    print(f"    collect {tc:7.1f}s ({100*tc/tot:4.1f}%)   update {tu:7.1f}s ({100*tu/tot:4.1f}%)")
    print(f"    {rate:5.2f} ep/s   ->  {realistic:5.2f} ep/s with a 10x-faster update "
          f"({realistic/rate:.2f}x)   ceiling {ideal/rate:.2f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="social", choices=["solo", "social", "mixed"])
    p.add_argument("--virtuous", type=int, default=1)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--n_envs", type=int, default=16)
    p.add_argument("--batch_episodes", type=int, default=32)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--sweep", action="store_true", help="try several thread counts")
    a = p.parse_args()

    print(f"mode={a.mode} virtuous={a.virtuous} n_envs={a.n_envs} "
          f"batch_episodes={a.batch_episodes} reps={a.reps}")
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}\n")

    for t in ([1, 2, 4, 8] if a.sweep else [a.threads]):
        tc, tu, eps = run_once(a.mode, a.virtuous, t, a.n_envs, a.batch_episodes, reps=a.reps)
        report(f"threads={t}", tc, tu, eps)

    print("\n  collect() is pure-Python env stepping; no GPU helps it.")
    print("  The 'ceiling' column is the best a GPU could ever do here.")


if __name__ == "__main__":
    main()
