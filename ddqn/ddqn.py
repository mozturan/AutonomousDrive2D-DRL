#DDQN agent

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
tf.random.set_seed(26)
import numpy as np
import board
import board_steps
from buffer import ReplayBuffer
import gc
import time


class DDQNAgent:

    def __init__(self, alpha, gamma, epsilon, obs_shape,
                 batch_size, epsilon_dec, epsilon_end, mem_size, 
                 min_mem_size, learning_rate, replace_target):

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.min_mem_size = min_mem_size
        self.replace_target = replace_target
        self.obs_shape = obs_shape
        self.learning_rate = learning_rate

        array_size = 21  # Adjust the size of the array as needed
        self.discrete_action_space = np.linspace(-1, 1, array_size)

        self.n_actions = len(self.discrete_action_space)
        self.action_space = [i for i in range(self.n_actions)]

        self.memory = ReplayBuffer(max_size=self.mem_size, min_size=self.min_mem_size,input_shape=self.obs_shape,
                             n_actions=self.n_actions,discrete=True)
                        
        self.q_eval = self._make_model()
        self.q_target = self._make_model()      #we keep a target model which we update every K timesteps
        self.q_eval.summary()
        print("summary")
        # plot_model(self.q_eval, to_file='./model_ddqn.png')

        self.tensorboard = board.ModifiedTensorBoard(log_dir=f"logs/{board.MODEL_NAME}-{int(time.time())}")
        self.tensorboard_steps = board_steps.ModifiedTensorBoard(log_dir=f"logs/{board_steps.MODEL_NAME}-{int(time.time())}")

    def _make_model(self):
        
        model = Sequential()
        model.add( Dense(256, activation='relu', input_dim = self.obs_shape[0]) )
        model.add( Dense(256, activation='relu') )
        model.add( Dense( self.n_actions))
        model.compile(loss='mse',optimizer= Adam(learning_rate = self.learning_rate),metrics=["accuracy"]) # type: ignore
 
        return model

    def epsilon_decay(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_end \
        else self.epsilon_end

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())
        
    def get_action(self, observation, deterministic=False):

        if np.random.random() > self.epsilon or deterministic: # type: ignore
    
            # observation = tf.convert_to_tensor(observation, dtype = tf.float16)
            observation = tf.expand_dims(observation, axis=0)

            qs_= self.q_eval.predict(observation, verbose=0) # type: ignore

            #*----------------------------------------------------------------
            #! Added for testing only
            # print("Predicted Q values: ", qs_)
            #*----------------------------------------------------------------
            action_index = np.argmax(qs_)
            action = self.discrete_action_space[action_index]
        else:
            action_index = np.random.randint(0, self.n_actions)
            action = self.discrete_action_space[action_index]
        
        return action, action_index

    def train(self):

        if (self.memory.mem_cntr) < self.min_mem_size:
            return
        #* and ELSE:
        #* sample minibatch and get states vs..
        state, action, reward, new_state, done, sample_indices = \
                            self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        # state = tf.convert_to_tensor(state, dtype = tf.float16)
        # new_state = tf.convert_to_tensor(new_state, dtype = tf.float16)
        # reward = tf.convert_to_tensor(reward, dtype = tf.float16)
        # done = tf.convert_to_tensor(done)
        # action_indices = tf.convert_to_tensor(action_indices, dtype=np.int8)
        
        #* get the q values of current states by main network
        q_pred = self.q_eval.predict(state,verbose=0) # type: ignore

        #! for abs error
        target_old = np.array(q_pred)

        #* get the q values of next states by target network
        q_next = self.q_target.predict(new_state, verbose=0) # type: ignore #! target_val

        #* get the q values of next states by main network
        q_eval = self.q_eval.predict(new_state, verbose=0) # type: ignore #! target_next

        #* get the actions with highest q values
        max_actions = np.argmax(q_eval, axis=1)

        #* we will update this dont worry
        q_target = q_pred

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        #* new_q = reward + DISCOUNT * max_future_q
        q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done

        #* error
        error = target_old[batch_index, action_indices]-q_target[batch_index, action_indices]
        self.memory.set_priorities(sample_indices, error)

        #* now we fit the main model (q_eval)
        _ = self.q_eval.fit(state, q_target, verbose = 0) # type: ignore

        #* If counter reaches set value, update target network with weights of main network
        #* it will update it at the very beginning also
        if self.memory.mem_cntr & self.replace_target == 0:
            self.update_network_parameters()
            print("Target Updated")

        gc.collect()
        K.clear_session()
        self.epsilon_decay()

    def save_model(self, episode):
        print("-----saving models------")
        self.q_eval.save_weights(f"weights/q_net-{episode}.h5")
        # self.q_target.save_weights(self.network.checkpoint_file)

    def load_model(self):
        print("-----loading models------")
        self.q_eval.load_weights("q_net.h5")
        self.update_network_parameters()