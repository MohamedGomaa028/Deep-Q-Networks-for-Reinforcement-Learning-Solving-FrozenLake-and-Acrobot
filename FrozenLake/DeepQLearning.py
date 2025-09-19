import numpy as np

from Environment import *
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import Loss, MSE
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings("ignore")
batch = 64


def ConvertState_to_One_HotEncoding(state, State_size):
    one_hot = np.zeros(State_size)
    one_hot[state] = 1.0
    return one_hot


# For discrete states we use one-hot encoding: each position is represented by a vector
# where its index is 1 and all others are 0. This prevents the neural network from
# interpreting state indices as having mathematical relationships (e.g., 10 > 5),
# and ensures each state is treated as a unique categorical input.

Qnetowrk = Sequential([
    Input(shape=(State_size,)),
    Dense(64, activation='relu', name="Layer1"),
    Dense(64, activation='relu', name="Layer2"),
    Dense(units=action_number, activation='linear', name="OutputLayer"),
])
TargetNetwork = Sequential([
    Input(shape=(State_size,)),
    Dense(64, activation='relu', name="Layer1"),
    Dense(64, activation='relu', name="Layer2"),
    Dense(units=action_number, activation='linear', name="OutputLayer"),
])
optimizer = Adam(learning_rate=Alpha)


###################################Compute loss##########################################
def compute_loss(experience, gamma, Qnetwork, TargetNetwork):
    batch, state, action, reward, next_state, done = experience
    Q_sa = tf.reduce_max(TargetNetwork(next_state), axis=1)
    y_target = reward + (gamma * Q_sa * (1 - done))
    q_value = Qnetwork(state)
    q_value = tf.gather_nd(q_value, tf.stack([tf.range(q_value.shape[0]),
                                              tf.cast(action, tf.int32), ], axis=1))
    loss = MSE(y_target, q_value)
    return loss


############################LearnAgent################################
#######For Graph Mode using GPU###########
'''def UpdateTargetNetwork(q_network,target_network,Tau=0.01):
    for Qw, Tw in zip(q_network.trainable_variables, target_network.trainable_variables):
        Tw.assign(Tau * Qw + (1.0 - Tau) * Tw)'''


#######For Edger Mode###########

def UpdateTargetNetwork(q_network, target_network, Tau=0.01):
    q_weight = q_network.get_weights()
    target_weight = target_network.get_weights()
    updated_weight = []
    for Qw, Tw in zip(q_weight, target_weight):
        updated_weight.append(
            Tau * Qw + (1 - Tau) * Tw)  # blend 1% of new Q-network weights with 99% of target network weights
    target_network.set_weights(updated_weight)


# @tf.function
def learn_Agent(experience, gamma):
    with tf.GradientTape() as tape:
        loss = compute_loss(experience, gamma, Qnetowrk, TargetNetwork)
    modelgradient = tape.gradient(loss,
                                  Qnetowrk.trainable_variables)  # trainable_variables--> neural network's learnable parameters (Weights,biases)
    optimizer.apply_gradients(zip(modelgradient, Qnetowrk.trainable_variables))
    UpdateTargetNetwork(Qnetowrk, TargetNetwork)
