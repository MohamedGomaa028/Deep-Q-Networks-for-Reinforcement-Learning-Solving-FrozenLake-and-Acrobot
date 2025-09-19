# Deep Q-Networks for Reinforcement Learning: Solving FrozenLake and Acrobot

This project implements **Deep Q-Networks (DQN)** from scratch using TensorFlow/Keras to solve two reinforcement learning environments from Gymnasium.  
The goal is to demonstrate how the same DQN framework can be applied to environments of increasing complexity.


_________

### FrozenLake-v1  
- **Environment link:** https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- **Goal:** Navigate a 4x4 frozen grid world from the start to the goal without falling into holes.  
- **States:** Discrete positions on the grid (converted to one-hot vectors for the neural network).  
- **Actions:** 4 discrete moves → Left, Right, Up, Down.  
- **Challenge:** Stochastic transitions (slipping on ice) make planning uncertain.  


_________

### Acrobot-v1
- **Environment link:** https://gymnasium.farama.org/environments/classic_control/acrobot/
- **Goal:** The goal is to have the free end reach a designated target height in as few steps as possible, and as such all steps that do not reach the goal incur a reward of -1. Achieving the target height results in termination with a reward of 0. The reward threshold is -100.
- **States:** Continuous, consisting of:  
  - `cos(θ1), sin(θ1)` → for the first joint  
  - `cos(θ2), sin(θ2)` → for the second joint  
  - `θ1_dot, θ2_dot` → angular velocities of both joints  
- **Actions:** 3 discrete torques → -1, 0, +1 applied to the joint.  
- **Challenge:** Highly non-linear dynamics, requires coordinated swinging to succeed.  


_________

## DQN Framework
Implemented techniques used across both environments:
- **Replay Buffer** for stable training.  
- **Target Network** to reduce Q-value oscillations.  
- **Epsilon-Greedy Exploration** with decay from 1.0 → 0.1.  
- Neural network with hidden layers for Q-value approximation.  


_________

## Training & Evaluation
- Training and testing pipelines implemented for both environments.  
- Logging of rewards and performance in CSV files.  
- Video recording of evaluation episodes for visualization.  






