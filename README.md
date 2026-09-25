# Virtue_RL

Does an RL agent learn to avoid harming a bystander purely by watching a teacher
who avoids it, with no penalty of its own for causing the harm?

A replication of the learner from Ndousse, Eck, Levine & Jaques (ICML 2021),
extended with an **Ethical Goal Cycle**: harm tiles are shortcuts through clutter,
and each traversal costs a bystander. Under R0 the harm never enters the learner's
own reward, so anything it learns comes from the teacher.

| | |
|---|---|
| **[instructions.md](instructions.md)** | how to run the whole thing on DRAC, step by step |
| **[METRICS.md](METRICS.md)** | what every logged column means, and how to read a run that looks wrong |
| **[sociapl/README.md](sociapl/README.md)** | the science: design options, tuning, success criteria, deviations from the paper |
| [sociapl/](sociapl/) | environment, model, PPO, training, evaluation, cluster scripts |
| [marlgrid/](marlgrid/) | vendored + patched environment package ([PATCHES.md](marlgrid/PATCHES.md)) |
| [example.env](example.env) | credentials template — copy to `.env` (gitignored) |

## Quick start

```bash
cp example.env .env      # REMOTE_HOST, REMOTE_DIR, ACCOUNT, WANDB_API_KEY
bash sync.sh --push

ssh prab@rorqual.alliancecan.ca
cd /home/prab/links/scratch/virtue_rl/sociapl
bash setup_cc.sh         # once
sbatch smoke_test.sh     # must print SMOKE TEST PASSED
```

Then follow [instructions.md](instructions.md) from step 5.

Locally, without a cluster:

```bash
pip install -r requirements.txt
pip install -e ./marlgrid --no-deps
cd sociapl && python train_ethics.py --episodes 2000 --out runs/smoke
```
