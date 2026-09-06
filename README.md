# Virtue_RL

Does an RL agent learn to *avoid harming a bystander* purely by watching a
teacher who avoids it — with no penalty of its own for causing the harm?

Built on a replication of the learner from Ndousse, Eck, Levine & Jaques,
*Emergent Social Learning via Multi-agent RL* (ICML 2021), extended with an
**Ethical Goal Cycle** environment: harm tiles are shortcuts through clutter, and
each traversal costs a bystander −1. Teachers are either virtuous (route around
them) or shortcut-taking. Under R0 the harm never enters the learner's own
reward at all, so anything it learns has to come from the teacher.

The test statistic is `harm_per_100_moves`, normalised by movement because harm
tiles sit on efficient paths — a busier agent encounters more of them regardless
of any ethical behaviour.

## Where things are

| | |
|---|---|
| **[experiment.md](experiment.md)** | how to run the whole thing on Rorqual (DRAC) |
| **[METRICS.md](METRICS.md)** | what every logged column means, and how to read a run that looks wrong |
| **[sociapl/README.md](sociapl/README.md)** | the science: replication targets, success criteria, deviations from the paper |
| [sociapl/](sociapl/) | environment, model, PPO, training and evaluation |
| [marlgrid/](marlgrid/) | vendored + patched environment package ([PATCHES.md](marlgrid/PATCHES.md)) |
| [slurm/](slurm/) | cluster configuration and job scripts |
| [example.env](example.env) | template for credentials — copy to `.env` (gitignored) |

## Quick start

Locally:

```bash
pip install -r requirements.txt
pip install -e ./marlgrid --no-deps
cd sociapl && python train_ethics.py --episodes 2000 --out runs/smoke
```

On the cluster — see [experiment.md](experiment.md) for the real version:

```bash
cp example.env .env && chmod 600 .env   # wandb key, DRAC user, account
bash slurm/deploy.sh                    # rsync the code up
ssh <you>@rorqual.alliancecan.ca
cd ~/Virtue_RL
bash slurm/setup_env.sh                 # once, on a login node
GRID=pilot bash run_all.sh              # 3 conditions x 1 seed @ 200k
bash slurm/sync_wandb.sh                # push to wandb.ai
```

Deployment is rsync, not git — no remote to configure, and `.env` reaches the
cluster without ever being committed.

CPU-only throughout; there is no GPU code path.
