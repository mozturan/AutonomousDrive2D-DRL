from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.models import Model
from keras.utils import plot_model
from keras.utils import model_to_dot 
from keras.optimizers import Adam
import pydot
import pydotplus
from pydotplus import graphviz
import keras
import tensorflow as tf
import os
import tensorflow_probability as tfp

model = Sequential()
model.add( Dense(128, activation='relu', input_dim = 28))
model.add( Dense(128, activation='relu') )
model.add( Dense(45))
model.compile(loss='mse',optimizer= Adam(learning_rate = 0.0003),metrics=["accuracy"]) # type: ignore
model.summary()
print("summary")
plot_model(model,
           show_shapes=True, show_layer_names=True,
            show_layer_activations= True,
            show_trainable=True,
            to_file='./model_ddqn_kinematics.png')


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

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)

    def call(self, state):
        prob = self.fc1(state)
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
        log_probs -= tf.math.log(1 - tf.math.pow(action,2)+self.noise) # type: ignore
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs


actor = ActorNetwork(n_actions=2, name='actor',
                                    max_action= 1.0)

actor.compile(optimizer=Adam(learning_rate=0.0003))
actor.build((None,28))

actor.summary()
plot_model(actor,
           show_shapes=True, show_layer_names=True,
            show_layer_activations= True,
            show_trainable=True,
            to_file='./model_SAC_kinematics.png')

# # Convert the Keras model to a dot format
# dot = model_to_dot(model, show_shapes=True, show_layer_names=True)

# # Save the dot file (optional)
# dot_file_path = 'model.dot'
# with open(dot_file_path, 'w') as f:
#     f.write(dot.to_string())

# # Convert the dot file to an SVG image (optional)
# svg_file_path = 'model.svg'
# dot.write_svg(svg_file_path)

# # Display the SVG image in Jupyter Notebook (optional)
# # SVG(svg_file_path)