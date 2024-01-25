#observations shape
import numpy as np

def observation_shape(observation):
    return (np.array(np.transpose(observation, (1, 2, 0)))) /255.0