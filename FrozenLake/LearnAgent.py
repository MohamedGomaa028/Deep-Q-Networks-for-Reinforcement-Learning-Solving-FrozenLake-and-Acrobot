from DeepQLearning import *
num_episodes = 10000
maxnum_steps=500
totalpoinhistory=[]
batch_size=64
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
memory_buffer = deque(maxlen=MEMORY_SIZE)
Min_NumberSteps_forUpdate=4
target_q_network.set_weights(q_network.get_weights())
for episode in range(num_episodes):
    state = env.reset()
    totalpoint=0
    for t in range(maxnum_steps):
        state = np.array(state).reshape(1, -1)
        if np.random.rand()<epsilon:
            action=env.action_space.sample()
        else:
            q_values=Qnetwork(state)
            action=np,argmax(q_values)
        nextstate, reward, done, info = env.step(action)
        memory_buffer.append(experience(state, action, reward, nextstate, done))
        state = nextstate
        totalpoint+=reward
        if (t+1)%Min_NumberSteps_forUpdate==0 and len(memory_buffer)>batch_size:
            batch=random.sample(memory_buffer,batch_size)
            LearnAgent(batch,gamma)
        if done:
            break
    if epsilon> epsilon_min:
        epsilon*=epsilon_decay
    totalpoinhistory.append(totalpoint)







