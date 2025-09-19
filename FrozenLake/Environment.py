import gymnasium as gym
import time as time
from collections import deque, namedtuple

env = gym.make("FrozenLake-v1", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder="videos",
    episode_trigger=lambda ep: ep % 500 == 0
)

Move_left = 0
Move_down = 1
Move_right = 2
Move_up = 3

memory_size = 200_000
gamma = 0.99
Alpha = 3e-4
numSteps_Update = 4

obs, info = env.reset()
State_size = env.observation_space.n
action_number = env.action_space.n

experience = namedtuple("Experience", field_names=["batch", "state", "action", "reward", "next_state", "done"])

print("Initial observation (state index):", obs)
print("Observation space:", State_size)  # Describes the possible states → it will be a 4*4 grid = 16
print("Action space:", action_number)  # Describes the possible actions, e.g., up, down, right, left
'''def enviroment_show(env):
    done = False
    while not done:
        action = env.action_space.sample()  # Random actions
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"action={action}  obs={obs}  reward={reward}  done={done}")
    env.close()'''
