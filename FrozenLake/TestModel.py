from LearnModel import *

Test_episodes = 5

test_env = gym.make('FrozenLake-v1', render_mode='rgb_array')
test_env = gym.wrappers.RecordVideo(
    test_env,
    video_folder="test_videos",
    episode_trigger=lambda ep: True,
)

with open("FrozenLake_TestLog.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Step", "Reward", "Success"])

for episode in range(Test_episodes):
    state, _ = test_env.reset()
    done = False
    step = 0
    total_Point = 0

    for i in range(max_num_steps):
        stateonehot = ConvertState_to_One_HotEncoding(state, State_size)
        state_input = np.array([stateonehot], dtype=np.float32)

        q_values = Qnetowrk(state_input)
        action = np.argmax(q_values.numpy())

        next_state, reward, terminated, truncated, info = test_env.step(action)

        state = next_state
        step += 1
        total_Point += reward
        done = terminated or truncated
        if done:
            break
    Success = 1 if total_Point >= 1 else 0
    with open("FrozenLake_TestLog.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, step, total_Point, Success])

test_env.close()
