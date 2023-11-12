import numpy as np

def _reward(info, x, y):
    reward = 0.
    done = False
    rewards = info["rewards"]
    r_alive = 0.1
    r_terminal = -5.
    # _distance = np.sqrt((x**2) + (y**2))
    # r_distance = 1. / _distance
    # _distance_cost = -0.3

    if (not info["crashed"]) and (rewards["on_road_reward"]):
        reward = rewards["lane_centering_reward"] + r_alive #+ (_distance_cost * r_distance)
        done = False
    else: 
        reward = r_terminal
        done = True

    return reward, done
