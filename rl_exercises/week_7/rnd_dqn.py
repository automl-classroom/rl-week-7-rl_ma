"""
Deep Q-Learning with RND implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import os
import minigrid
from minigrid.wrappers import FlatObsWrapper
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rl_exercises.week_7.RNDNetwork import RNDNetwork


class RNDDQNAgent(DQNAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        rnd_hidden_size: int = 128,
        rnd_lr: float = 1e-3,
        rnd_update_freq: int = 1000,
        rnd_n_layers: int = 2,
        rnd_reward_weight: float = 0.1,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.seed = seed
        # TODO: initialize the RND networks

        self.rnd_hidden_size = rnd_hidden_size
        self.rnd_lr = rnd_lr
        self.rnd_update_freq = rnd_update_freq
        self.rnd_n_layers = rnd_n_layers
        self.rnd_reward_weight = rnd_reward_weight
        
        self.rnd_targetNetwork = RNDNetwork(
            input_dimension=self.env.observation_space.shape[0],
            output_dimension=128,  # Rule of thumb output dimension, can be adjusted
            hidden_dimension=self.rnd_hidden_size,
            n_layers=self.rnd_n_layers,
        )
        self.rnd_predictorNetwork = RNDNetwork(
            input_dimension=self.env.observation_space.shape[0],
            output_dimension=128,  # Rule of thumb output dimension, can be adjusted
            hidden_dimension=self.rnd_hidden_size,
            n_layers=self.rnd_n_layers,
        )
        self.rnd_optimizer = optim.Adam(
            self.rnd_predictorNetwork.parameters(), lr=self.rnd_lr
        ) # Initialize the optimizer for the RND predictor network, target network is not optimized because it is frozen
        

    def update_rnd(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on the RND network on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).
        """
        # TODO: get states and next_states from the batch
        states = np.array([transition[0] for transition in training_batch]) # why though? I don't think we need it
        np_next_states = np.array([transition[3] for transition in training_batch])
        next_states = torch.tensor(np_next_states, dtype=torch.float32)

        # TODO: compute the MSE
        rnd_predicted_state_vector = self.rnd_predictorNetwork(next_states)
        with torch.no_grad():
            # Freeze the target network during training
            rnd_target_state_vector = self.rnd_targetNetwork(next_states)
        rnd_error = torch.mean((rnd_predicted_state_vector - rnd_target_state_vector) ** 2)

        # TODO: update the RND network
        self.rnd_optimizer.zero_grad()
        rnd_error.backward()
        self.rnd_optimizer.step()

        return rnd_error.item()

    def get_rnd_bonus(self, state: np.ndarray) -> float:
        """Compute the RND bonus for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        float
            The RND bonus for the state.
        """
        # TODO: predict embeddings
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            predicted = self.rnd_predictorNetwork(state_tensor)
            fixed_target = self.rnd_targetNetwork(state_tensor)
        # TODO: get error
        error = torch.mean((predicted - fixed_target) ** 2)
        return error.item() * self.rnd_reward_weight

    def train(self, num_frames: int, eval_interval: int = 1000, rnd_Mode: bool = True) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        episode_rewards = []
        steps = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # TODO: apply RND bonus
            if rnd_Mode: reward += self.get_rnd_bonus(next_state)

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

                if self.total_steps % self.rnd_update_freq == 0:
                    self.update_rnd(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                steps.append(frame)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards)
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        # Saving to .csv for simplicity
        # Could also be e.g. npz
        print("Training complete.")
        training_data = pd.DataFrame({"steps": steps, "rewards": episode_rewards})
        # training_data.to_csv(f"training_data_seed_{self.seed}.csv", index=False)

        output_dir = os.path.join(hydra.utils.get_original_cwd(), "rl_exercises/week_7/rnd_results")
        os.makedirs(output_dir, exist_ok=True)
        training_data.to_csv(os.path.join(output_dir, f"training_dataseed{self.seed}_mode{rnd_Mode}.csv"), index=False)

        model_dir = os.path.join(hydra.utils.get_original_cwd(), "rl_exercises/week_7/models")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(
        {
        "parameters": self.q.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "seed": self.seed,
        "RND_mode": rnd_Mode,
        },
        os.path.join(model_dir, f"dqn_modelseed{self.seed}mode{rnd_Mode}.pth")
        )


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env

    for seed in range(0,5):
        for rnd_Mode in [True, False]:
            env = gym.make("MiniGrid-Empty-8x8-v0")
            env = FlatObsWrapper(env)
            set_seed(env, seed)

            # 3) TODO: instantiate & train the agent
            agent = RNDDQNAgent(env=env,
                                buffer_capacity=cfg.agent.buffer_capacity,
                                batch_size=cfg.agent.batch_size,
                                lr=cfg.agent.learning_rate,
                                gamma=cfg.agent.gamma,
                                epsilon_start=cfg.agent.epsilon_start,
                                epsilon_final=cfg.agent.epsilon_final,
                                epsilon_decay=cfg.agent.epsilon_decay,
                                target_update_freq=cfg.agent.target_update_freq,
                                seed=seed,
                                )
            agent.train(num_frames=cfg.train.num_frames, eval_interval=cfg.train.eval_interval, rnd_Mode=rnd_Mode)


if __name__ == "__main__":
    main()
