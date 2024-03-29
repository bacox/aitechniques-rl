import simple_grid
import numpy as np
import gym
from q_learning_skeleton import *

def value_iteration(env, max_iterations=100000, epsilon=0.9):
    stateValue = [[0 for j in range(env.nA)] for i in range(env.nS)]
    newStateValue = stateValue.copy()
    for i in range(max_iterations):
        for state in range(env.nS):
            action_values = []
            for action in range(env.nA):
                state_value = 0
                for i in range(len(env.P[state][action])):
                    prob, next_state, reward, done = env.P[state][action][i]
                    best_action = np.argmax(np.asarray(stateValue[next_state]))
                    state_action_value = prob * (reward + epsilon * stateValue[next_state][best_action])
                    state_value += state_action_value
                action_values.append(state_value)  # the value of each action
                # best_action = np.argmax()  # choose the action which gives the maximum value
                newStateValue[state] = np.asarray(action_values)  # update the value of the state
        if i > 1000:
            if sum(stateValue) - sum(newStateValue) < 1e-04:  # if there is negligible difference break the loop
                break
        else:
            stateValue = newStateValue.copy()
    return stateValue


env = gym.make('FrozenLake8x8-v0')


if __name__ == '__main__':
    gamma = DEFAULT_DISCOUNT

    env = simple_grid.DrunkenWalkEnv(map_name="theAlley")
    if (type(env.observation_space)  == gym.spaces.discrete.Discrete):
        num_o = env.observation_space.n
    else:
        raise("Qtable only works for discrete observations")

    stateValues = value_iteration(env, max_iterations=1000)
    string = ''
    for i, item in enumerate(stateValues):
        if i % env.ncol == 0 and i > 0:
            string += "\n"
        reward = np.max(item)
        for a in range(len(item)):
            if item[a] == reward:
                string += "{}".format(["L", "D", "R", "U"][a])
        string += " "
    env.render()
    print("")
    print("Policy:")
    print(string)