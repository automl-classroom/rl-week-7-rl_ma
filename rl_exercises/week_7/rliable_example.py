import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

n_seeds = 4
seeds = np.arange(n_seeds)
eval_interval = 100
max_steps = 10000
aligned_steps = np.arange(0, max_steps, eval_interval)

def load_and_interp(mode: str):
    interpolated = []
    for seed in seeds:
        path = f"rl_exercises/week_7/rnd_results/training_dataseed{seed}_mode{mode}.csv"
        df = pd.read_csv(path).sort_values("steps")
        rewards_interp = np.interp(aligned_steps, df["steps"], df["rewards"])
        interpolated.append(rewards_interp)
    return np.stack(interpolated)  # shape: [n_seeds, len(aligned_steps)]

# Load both versions
vanilla_scores = load_and_interp("False")  # vanilla DQN
rnd_scores = load_and_interp("True")       # RND-enhanced DQN

# Prepare for rliable
train_scores = {
    "vanilla_dqn": vanilla_scores,
    "rnd_dqn": rnd_scores
}

# IQM aggregation
iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[:, i]) for i in range(scores.shape[1])])
iqm_scores, iqm_cis = get_interval_estimates(train_scores, iqm, reps=2000)

# Plot
plot_sample_efficiency_curve(
    aligned_steps,
    iqm_scores,
    iqm_cis,
    algorithms=["vanilla_dqn", "rnd_dqn"],
    xlabel="Environment Steps",
    ylabel="Episode Reward (IQM)",
)
plt.gcf().canvas.manager.set_window_title(
    "IQM Episode Reward - RND vs Vanilla DQN"
)
plt.legend()
plt.tight_layout()
plt.show()