# PER
import numpy as np
import random as rndm

class ReplayBuffer(object):
    """
    * init the values
    * for DQN actions are discrete
    """
    def __init__(self, max_size, min_size, input_shape, n_actions, discrete=True):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.min_size = min_size
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
