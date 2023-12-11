import tensorflow as tf
import keras as keras
from keras.optimizers import Adam
import numpy as np
import board
import time
import gc
import keras.backend as K

class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, discrete = True):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.index = 0
        
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float16)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float16)
        dtype = np.int8 if self.discrete else np.float16
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float16)
        self.terminal_memory = np.zeros(self.mem_size)

    def store_transition(self, state, action, reward, state_, done):

        index = self.mem_cntr % self.mem_size                
        self.state_memory[index] = state
        self.new_state_memory[index] = state_

        #* store one hot encoding of actions, if appropriate
        if self.discrete:
            #* Create an zeros-array size of the number of actions
            actions = np.zeros(self.action_memory.shape[1])
            #* Make 1 the value of performed action
            actions[action] = 1.0
            #* Store in action memory
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action

        #* store reward and if it's terminal info 
        self.reward_memory[index] = reward
        #* we send inverse done info!!!
        self.terminal_memory[index] = 1 - done
        self.mem_cntr +=1
        self.index = self.mem_cntr

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.9995, eps_end=0.05, 
                 mem_size=100000, fc1_dims=256,
                 fc2_dims=256, replace=1000):
        
        array_size = 11  # Adjust the size of the array as needed
        self.discrete_action_space = np.linspace(-1, 1, array_size)
        self.n_actions = len(self.discrete_action_space)
        self.action_space = [i for i in range(self.n_actions)]

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec # type: ignore
        self.epsilon_end = eps_end
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims,n_actions=self.n_actions)
        self.q_eval = DuelingDeepQNetwork(self.n_actions, fc1_dims, fc2_dims)
        self.q_next = DuelingDeepQNetwork(self.n_actions, fc1_dims, fc2_dims)

        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        # just a formality, won't optimize network
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        
        self.tensorboard = board.ModifiedTensorBoard(log_dir=f"logs/{board.MODEL_NAME}-{int(time.time())}")

    def epsilon_decay(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_end \
        else self.epsilon_end


    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action_index = np.random.randint(0, self.n_actions)
            action = self.discrete_action_space[action_index]

        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action_index = tf.math.argmax(actions, axis=1).numpy()[0]
            action = self.discrete_action_space[action_index]

        return action, action_index

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return


        states, actions, rewards, states_, dones = \
                                    self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(actions, action_values)
        
        q_pred = self.q_eval(states)
        q_next = self.q_next(states_)
        # changing q_pred doesn't matter because we are passing states to the train function anyway
        # also, no obvious way to copy tensors in tf2?
        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
        
        # improve on my solution!
        for idx, terminal in enumerate(dones):
            #if terminal:
                #q_next[idx] = 0.0
            q_target[idx, action_indices[idx]] = rewards[idx] + \
                    self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx])) # type: ignore
        self.q_eval.train_on_batch(states, q_target)


        if self.memory.mem_cntr % self.replace ==0:
            self.q_next.set_weights(self.q_eval.get_weights())
            print("Target Updated")

        self.epsilon_decay()
        gc.collect()
        K.clear_session()

        self.learn_step_counter += 1