import gymnasium as gym
import torch
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rl_exercises.week_7.rnd_dqn import RNDDQNAgent
import minigrid
from minigrid.wrappers import FlatObsWrapper


# 1. Umgebung mit Render-Modus erstellen
env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="human")  # Passe den Namen ggf. an dein Modell an
env = FlatObsWrapper(env)
set_seed(env, 10)

# 2. Agent initialisieren (Parameter wie beim Training!)
agent = RNDDQNAgent(
    env=env,
    buffer_capacity=5000,
    batch_size=32,
    lr=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_final=0.01,
    epsilon_decay=500,
    target_update_freq=1000,
    seed=0,
)
 
# 3. Modell laden (Pfad ggf. anpassen)
checkpoint = torch.load("rl_exercises/week_7/models/dqn_modelseed1modeFalse.pth", map_location=torch.device('cpu'))
agent.q.load_state_dict(checkpoint["parameters"])
agent.optimizer.load_state_dict(checkpoint["optimizer"])

# 4. Simulation (eine Episode)
state, _ = env.reset()
done = False
while not done:
    action = agent.predict_action(state, evaluate=True)  # keine Exploration
    state, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        break

env.close()