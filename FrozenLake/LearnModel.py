import csv

from DeepQLearning import *

num_episodes = 6000
max_num_steps = 200
totalpoinhistory = []
batch_size = 64
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
memory_buffer = deque(maxlen=memory_size)
Min_NumberSteps_forUpdate = 4
TargetNetwork.set_weights(Qnetowrk.get_weights())
with open("FrozenLake_TrainingLog.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Step", "Reward", "Epsilon", "NumberOf_RandomAction_PerEpisode",
                     "Numberof_UsingQnetwork_PerEpisode"])

for episode in range(num_episodes):

    state, info = env.reset()
    totalpoint = 0
    Using_Random_Action = 0
    Using_Qnetwork = 0
    step = 0
    for t in range(max_num_steps):
        state_one_hot = ConvertState_to_One_HotEncoding(state, State_size)
        state_input = np.array([state_one_hot], dtype=np.float32)

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
            Using_Random_Action += 1

        else:
            q_values = Qnetowrk(state_input)
            action = np.argmax(q_values)
            Using_Qnetwork += 1

        nextstate, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state_onehot = ConvertState_to_One_HotEncoding(nextstate, State_size)
        memory_buffer.append(experience(batch, state_one_hot, action, reward, next_state_onehot, done))
        state = nextstate
        totalpoint += reward
        step += 1

        if (t + 1) % Min_NumberSteps_forUpdate == 0 and len(memory_buffer) > batch_size:
            batch_indices = np.random.choice(len(memory_buffer), 64, replace=False)

            batch = [memory_buffer[i] for i in batch_indices]
            states = tf.convert_to_tensor(np.array([x.state for x in batch if x is not None]), dtype=tf.float32)
            action = tf.convert_to_tensor(np.array([x.action for x in batch if x is not None]), dtype=tf.float32)
            rewards = tf.convert_to_tensor(np.array([x.reward for x in batch if x is not None]), dtype=tf.float32)
            nextstates = tf.convert_to_tensor(np.array([x.next_state for x in batch if x is not None]),
                                              dtype=tf.float32)
            Done_value = tf.convert_to_tensor(np.array([x.done for x in batch if x is not None]), dtype=tf.float32)
            batch_value = (batch, states, action, rewards, nextstates, Done_value)
            learn_Agent(batch_value, gamma)
        if done:
            break

    if episode % 250 == 0:
        with open("FrozenLake_TrainingLog.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, step, totalpoint, epsilon, Using_Random_Action, Using_Qnetwork])

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    totalpoinhistory.append(totalpoint)

env.close()
print("Training completed!")
