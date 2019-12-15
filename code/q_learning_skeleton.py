import random
import numpy as np

NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500

DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
EPSILON_MAX = 1.0
LEARNINGRATE = 0.1
exploration_decay_rate = 0.01


class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, env, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE): # You can add more arguments if you want
        self.name = "agent1"
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.state = 0
        self.q_values = {}
        for s in range(num_states):
            for a in range(num_actions):
                self.q_values[(s, a)] = 0

    def process_experience(self, state, action, next_state, reward, done): # You can add more arguments if you want
        """
        Update the Q-value based on the state, action, next state and reward.
        """

        q_old = self.q_values[(state, action)]
        a_prime = self.get_best_action(next_state)
        q_old_prime = self.q_values[(next_state, a_prime)]

        if not done:
            self.q_values[(state, action)] = (1 - LEARNINGRATE) * q_old + LEARNINGRATE * (reward + DEFAULT_DISCOUNT*q_old_prime)
        else:
            self.q_values[(state, action)] = (1 - LEARNINGRATE) * q_old + LEARNINGRATE * reward

        pass

    def get_exploration_rate(self, episode):
        return EPSILON + (EPSILON_MAX - EPSILON) * np.exp(-exploration_decay_rate * episode)

    def select_action(self, state, episode): # You can add more arguments if you want
        """
        Returns an action, selected based on the current state
        """
        # Generate a random probability number between 0 and 1
        random_prob = random.uniform(0, 1)

        # At epsilon probability, we perform a random action
        if random_prob <= self.get_exploration_rate(episode):
            return random.randint(0, self.num_actions - 1)

        # Otherwise, perform the best possible action
        return self.get_best_action(state)

    def get_best_actions(self, state):
        """
        Function to determine all best actions to take at given state,
        i.e. all actions with the highest Q values
        """
        q_values = [self.q_values[(state, a)] for a in range(self.num_actions)]
        max_q = max(q_values)
        max_q_indices = [i for i, q in enumerate(q_values) if q == max_q]

        return max_q_indices

    def get_best_action(self, state):
        """
        Function to determine the best action to take at given state
        If multiple actions are optimal, we pick one of these at random.
        """
        best_actions = self.get_best_actions(state)

        if len(best_actions) > 1:
            best_action = random.choice(best_actions)
        else:
            best_action = best_actions[0]

        return best_action

    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        print("--AGENT REPORT --")
        print(self.policy_map())

    def policy_map(self):
        """
        Function that returns a string view of the policy map.
        If multiple actions are optimal, all of these actions are printed.
        L: Left
        D: Down
        R: Right
        U: Up
        """
        output = ""

        for s in range(self.num_states):
            best_actions = self.get_best_actions(s)

            for a in best_actions:
                string = "{}".format(["L", "D", "R", "U"][a])
                output = output + string

            if s % self.env.ncol == self.env.ncol - 1:
                output = output + "\n"
            else:
                for i in range(self.num_actions + 1 - len(best_actions)):
                    output = output + " "

        return output
