from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp
import keras.backend as K
import keras
import os
import tensorflow as tf
tf.random.set_seed(26)
from buffer import ReplayBuffer
import gc
import sys
import random as rndm
import numpy as np
import board
import time


class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=128, fc2_dims=128,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')


        self.conv_a =Conv2D(32, 3, activation="relu")
        self.conv_b =Conv2D(64, 3, activation="relu")
        self.conv_c =MaxPooling2D(pool_size=(2, 2))
        self.flatten =Flatten()

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):

        state_value = self.conv_a(state)

        state_value = self.conv_b(state_value)
        state_value = self.conv_c(state_value)
        state_value = self.flatten(state_value)

        action_value = self.fc1(tf.concat([state_value, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q

class ValueNetwork(keras.Model):
    def __init__(self, fc1_dims=128, fc2_dims=128,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.conv_a =Conv2D(32, 3, activation="relu")
        self.conv_b =Conv2D(64, 3, activation="relu")
        self.conv_c =MaxPooling2D(pool_size=(2, 2))
        self.flatten =Flatten()

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)

    def call(self, state):

        state_value = self.conv_a(state)
        state_value = self.conv_b(state_value)
        state_value = self.conv_c(state_value)
        state_value = self.flatten(state_value)

        state_value = self.fc1(state_value)
        state_value = self.fc2(state_value)

        v = self.v(state_value)

        return v

class ActorNetwork(keras.Model):

    def __init__(self, max_action, fc1_dims=128,
            fc2_dims=128, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.noise = 1e-6

        self.conv_a =Conv2D(32, 3, activation="relu")
        self.conv_b =Conv2D(64, 3, activation="relu")
        self.conv_c =MaxPooling2D(pool_size=(2, 2))
        self.flatten =Flatten()

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)

    def call(self, state):

        state_value = self.conv_a(state)
        state_value = self.conv_b(state_value)
        state_value = self.conv_c(state_value)
        state_value = self.flatten(state_value)

        prob = self.fc1(state_value)
        prob = self.fc2(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        # might want to come back and change this, perhaps tf plays more nicely with
        # a sigma of ~0
        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.sample() # + something else if you want to implement
        else:
            actions = probabilities.sample()

        action = tf.math.tanh(actions)*self.max_action
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1 - tf.math.pow(action,2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs
    
class Agent:
    def __init__(self, alpha=0.003, beta=0.003, input_dims=[8],
            env=None, gamma=0.9, n_actions=1, max_size=100000, tau=0.05,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.temperature = 0.03
        self.actor = ActorNetwork(n_actions=n_actions, name='actor',
                                    max_action= 1.0 )
        self.critic_1 = CriticNetwork(n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(name='value')
        self.target_value = ValueNetwork(name='target_value')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        #* Custom tensorboard
        self.tensorboard = board.ModifiedTensorBoard(log_dir=f"logs/{board.MODEL_NAME}-{int(time.time())}")


    def choose_action(self, observation):
        # state = tf.convert_to_tensor([observation], dtype=tf.int32)
        
        state = tf.expand_dims(observation, axis=0)

        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)

    def save_models(self, episode):
        print('... saving models ...')
        os.mkdir(f"weights/{episode}") 
        self.actor.save_weights(f"weights/{episode}/self.actor.checkpoint_file.h5")
        self.critic_1.save_weights(f"weights/{episode}/self.critic_1.checkpoint_file.h5")
        self.critic_2.save_weights(f"weights/{episode}/self.critic_2.checkpoint_file.h5")
        self.value.save_weights(f"weights/{episode}/self.value.checkpoint_file.h5")
        self.target_value.save_weights(f"weights/{episode}/self.target_value.checkpoint_file.h5")

    def load_models(self, episode):

        print('... loading models ...')
        self.actor.load_weights(f"weights/{episode}/self.actor.checkpoint_file.h5")
        self.critic_1.load_weights(f"weights/{episode}/self.critic_1.checkpoint_file.h5")
        self.critic_2.load_weights(f"weights/{episode}/self.critic_2.checkpoint_file.h5")
        self.value.load_weights(f"weights/{episode}/self.value.checkpoint_file.h5")
        self.target_value.load_weights(f"weights/{episode}/self.target_value.checkpoint_file.h5")

    def learn(self):
        if self.memory.mem_cntr < 100:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            value_ = tf.squeeze(self.target_value(states_), 1)

            current_policy_actions, log_probs = self.actor.sample_normal(states,
                                                        reparameterize=False)
            log_probs = tf.squeeze(log_probs,1)
            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)
            critic_value = tf.squeeze(
                                tf.math.minimum(q1_new_policy, q2_new_policy), 1)

            value_target = critic_value - (self.temperature*log_probs)
            value_loss = 0.5 * keras.losses.MSE(value, value_target)

        value_network_gradient = tape.gradient(value_loss,
                                                self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(
                       value_network_gradient, self.value.trainable_variables))


        with tf.GradientTape() as tape:
            # in the original paper, they reparameterize here. We don't implement
            # this so it's just the usual action.
            new_policy_actions, log_probs = self.actor.sample_normal(states,
                                                reparameterize=True)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(
                                        q1_new_policy, q2_new_policy), 1)

            actor_loss = (self.temperature*log_probs) - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                            self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
                        actor_network_gradient, self.actor.trainable_variables))


        with tf.GradientTape(persistent=True) as tape:
            # I didn't know that these context managers shared values?
            q_hat = self.scale*reward + self.gamma*value_*(1-done)
            q1_old_policy = tf.squeeze(self.critic_1(state, action), 1)
            q2_old_policy = tf.squeeze(self.critic_2(state, action), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)

        critic_1_network_gradient = tape.gradient(critic_1_loss,
                                        self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss,
            self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(
            critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(
            critic_2_network_gradient, self.critic_2.trainable_variables))

        self.update_network_parameters()

        gc.collect()
        K.clear_session()
