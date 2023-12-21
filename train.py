from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_pybullet_drones.envs.PlumeDroneBulletEnv import PlumeDroneBulletEnv
import argparse
import numpy as np

# Parse Args
parser = argparse.ArgumentParser()
# parser.add_argument('-v', dest='verbose', default=False, action='store_true')
parser.add_argument('--gui', dest='gui', action='store_true')
parser.add_argument('--no-gui', dest='gui', action='store_false')
parser.set_defaults(gui=True)
parser.add_argument('--record', dest='record', default=False, action='store_true')
parser.add_argument('--num_drones', dest='num_drones', default=1, type=int)
parser.add_argument('--initial_drone_positions', dest='initial_drone_positions', default=None, type=int)
parser.add_argument('--num_plume_sources', dest='num_plume_sources', default=1, type=int)
parser.add_argument('--initial_plume_positions', dest='initial_plume_positions', default=None, type=int)
parser.add_argument('--max_steps', dest='max_steps', default=10000, type=int)
parser.add_argument('--size', dest='size', default=100, type=int)
parser.add_argument('--background_concentration', dest='background_concentration', default=5, type=int)
args = parser.parse_args()

def callback(locals, globals):
    # Extract the relevant information from locals_
    # For example, you can access the total reward of the current episode
    current_episode_reward = locals['ep_rew_mean']

    # Log the information (you can also use a custom logger)
    print(f"Average Reward per Episode: {current_episode_reward}")

def train_model():
    # Instantiate the environment
    env = PlumeDroneBulletEnv(
        num_drones=args.num_drones,
        initial_xyzs=args.initial_drone_positions,
        initial_plume_positions=args.initial_plume_positions,
        num_plume_sources=args.num_plume_sources,
        max_steps=args.max_steps,
        size=args.size,
        background_concentration=args.background_concentration,
        gui=args.gui,
        record=args.record
    )

    # The algorithms require a vectorized environment to run
    vec_env = DummyVecEnv([lambda: env])

    # Instantiate the agent
    model = PPO('MultiInputPolicy', vec_env, learning_rate=1e-4)

    # Train the agent
    model.learn(total_timesteps=1_000_000, progress_bar=True)

    # Save the model
    model.save("ppo_drone")

if __name__ == "__main__":
    train_model()
