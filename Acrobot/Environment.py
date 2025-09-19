import gymnasium as gym
import time as time
from collections import deque, namedtuple
env=gym.make("Acrobot-v1",render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder="videos",
    episode_trigger=lambda ep: ep % 500 == 0
)



memory_size=100_000
gamma=0.995
Alpha=1e-3
numSteps_Update=3

obs,info=env.reset()
State_size=env.observation_space.shape[0]
action_number=env.action_space.n

experience = namedtuple("Experience", field_names=["batch","state", "action", "reward", "next_state", "done"])

print("Initial observation (state index):", obs)
print("Observation space:", State_size)   # Describes the possible states → it will be a 4*4 grid = 16
print("Action space:", action_number) # Describes the possible actions, e.g., up, down, right, left
'''def enviroment_show(env):
    done = False
    while not done:
        action = env.action_space.sample()  # Random actions
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"action={action}  obs={obs}  reward={reward}  done={done}")
        #time.sleep(0.4)
    env.close()'''


