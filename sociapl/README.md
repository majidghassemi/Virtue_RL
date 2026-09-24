# SociAPL replication — Ndousse, Eck, Levine, Jaques (ICML 2021)

Reimplementation of the learner from *Emergent Social Learning via Multi-agent RL*.
The authors released only the environment (marlgrid); training code is reconstructed
from Sections 3-4 and Appendix 7.4-7.8. Network is 668,555 parameters, matching the paper.

## Setup
    pip install torch numpy    # Linux wheels from PyPI include CUDA; see pytorch.org for other CUDA versions
    cd ../marlgrid && pip install -e . && pip install pyglet    # patched copy, see PATCHES.md
    # PYGLET_HEADLESS=1 is set automatically by envs.py

## Files
    envs.py      make_env, ScriptedExpert (privileged BFS to next correct tile), Worker (solo/social/mixed)
    model.py     SociAPLNet: conv encoder -> FC(576,192) -> LSTM(192) -> policy / value / aux heads
    ppo.py       recurrent PPO-clip + GAE + auxiliary loss; KL target 0.01, hard revert at 0.03
    train.py     training CLI, logs runs/<name>/log.csv
    evaluate.py  H1 (3-goal solo/social) and H2 (4-goal zero-shot) evaluation
    bc.py        behaviour-cloning baseline on expert POV (privileged)

## Reproducing the paper's conditions (Fig. 4, 5, 6, 7)
    # H1, 3-goal Goal Cycle, seeds S = 0..4
    python train.py --mode social --aux pred  --seed S --out runs/social_pred_sS
    python train.py --mode solo   --aux pred  --seed S --out runs/solo_pred_sS
    python train.py --mode social --aux none  --seed S --out runs/social_vanilla_sS
    python train.py --mode social --aux rec   --seed S --out runs/social_rec_sS
    python train.py --mode social --aux pred  --expert_eps 0.3 --seed S --out runs/social_pred_imperfect_sS
    # Sec 4.3: continue a social checkpoint on a 75/25 solo/social mix
    python train.py --mode mixed --aux pred --init runs/social_pred_sS/ckpt.pt --seed S --out runs/mixed_pred_sS
    # H2 and solo transfer
    python evaluate.py --ckpt runs/<run>/ckpt.pt --aux <pred|rec|none> --episodes 100 --out runs/<run>/eval.json
    # IL baseline
    python bc.py --episodes 5000 --epochs 20 --out runs/bc_sS && python evaluate.py --ckpt runs/bc_sS/ckpt.pt --aux none

Defaults match Appendix 7.8: lr 1e-4, gamma 0.993, GAE lambda 0.97, clip 0.2, 128 episodes/batch,
20 minibatches of 512 length-16 segments, loss = pi + 0.1 v + 3 aux + 1e-5 ent, 1.5M episodes.
Env: 13x13, 3 goals, +1 / -1.5, 250 steps, prestige colouring, 21x21x3 partial view.

## Success criterion (decided before running)
Replicated iff across >= 5 seeds: (a) solo and social-vanilla never exceed return 1;
(b) a majority of social aux-pred seeds exceed the expert's return; (c) aux-pred > aux-rec on mean;
(d) 4-goal transfer with experts beats solo transfer. Same signs and ordering, not same decimals.

## Deviations from the paper (state these in the write-up)
1. Experts are scripted (BFS with privileged goal order), not RL-trained via the penalty curriculum.
   A scripted expert never explores, so the paper's "wait, then follow" strategy may not emerge.
   --expert_eps > 0 gives a noisy expert for the imperfect-expert condition.
2. Stored LSTM hidden states and advantages are not recomputed every 2 minibatches.
3. Four Rooms and Maze transfer envs are not in the released repo and are not implemented here.
4. Gradient-norm clipping at 0.5 was added (not mentioned in the paper).
5. Environment runs on gym 0.26 / numpy 2 with the patches listed in ../marlgrid/PATCHES.md.

## What has been verified so far
- Scripted expert: mean return ~26 (3-goal) and ~24 (4-goal) over 250 steps.
- All five training conditions, evaluate.py and bc.py run end to end.
- 112-episode pilot (8 envs, batch 16): aux loss 0.40 -> 0.18, KL within target, no NaNs.
  Learner return ~0 at this scale, as expected; the paper needs ~5e5 episodes for social learning.

## Compute
All scripts take `--device {auto,cuda,cuda:N,mps,cpu}` (default `auto`: CUDA, then Apple MPS, then CPU).
The network, the rollout buffer, GAE and the PPO/BC updates run on that device; checkpoints load onto
it via `map_location`, so GPU-trained checkpoints evaluate on CPU and vice versa.

Env stepping (marlgrid) is pure Python and always runs on the CPU, one process: ~480 learner
steps/s with 16 social envs on a 4-thread test machine (~280 before the duplicate `gen_obs` call per step was removed).
A GPU therefore speeds up the update phase (about half the wall time on CPU) but not collection; for a
big run, give the job a GPU plus a few CPU cores and keep `--threads` small.
The paper reports ~30 h per 1.5M-episode run on 2x1080Ti. Before that, run a 200k-episode pilot,
1 seed x 4 conditions, and check that ordering (a)-(d) appears. If it does not, scaling to 1.5M
will not rescue it.

## Ethics extension (Ethical Goal Cycle)
    ethics.py        HarmTile (shortcut through clutter, -1 to the bystander per traversal),
                     EthicalGoalCycleEnv (R0/R1/R2 via --harm_delivery), EthicalExpert
                     (virtuous avoids harm tiles in BFS; control variant takes them), EthicsWorker
    train_ethics.py  the experiment grid (see docstring for the 6 commands)
    eval_ethics.py   2x2 evaluation: {teacher, alone} x {seen, unseen}; reports virtue gap

Verified: virtuous teachers cause 0 harm events/ep, shortcut teachers ~26; virtue costs the
teacher ~2 return (22.5 vs 24.6); dense and delayed learner penalties apply correctly.
Kill experiment = e_r0_virt vs e_r0_solo harm rate at ~200k episodes, 3 seeds.
