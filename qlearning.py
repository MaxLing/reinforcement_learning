from maze import *
from value_plot import *
from evaluation import *
import numpy as np
import random

class QLearningAgent(object):
    """
        Adopted from http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
        A model-free Temporal Difference learning agent
    """
    def __init__(self, get_legal_actions, discount=0.9, learn_rate=0.2, explore_rate=0.5):
        self.get_legal_actions = get_legal_actions
        self.discount = discount
        self.learn_rate = learn_rate
        self.explore_rate = explore_rate
        self.qvalues = {}

    def getQValue(self, state, action):
        if (state, action) not in self.qvalues: # init
            self.qvalues[(state, action)] = 0
        return self.qvalues[(state, action)]

    def getValue(self, state):
        legal_actions = self.get_legal_actions(state)
        max_value = float('-inf')
        for action in legal_actions:
            q_value = self.getQValue(state, action)
            max_value = max(max_value, q_value)
        return max_value

    def getPolicy(self, state):
        legal_actions = self.get_legal_actions(state)
        best_action = None
        max_value = float('-inf')
        for action in legal_actions:
            q_value = self.getQValue(state, action)
            if q_value > max_value:
                max_value = q_value
                best_action = action
            elif q_value == max_value:
                best_action = random.choice([best_action, action]) # tie breaker
        return best_action

    def getAction(self, state):
        legal_actions = self.get_legal_actions(state)

        if np.random.random() < self.explore_rate:
            return random.choice(legal_actions)
        else:
            return self.getPolicy(state)

    def learn(self, state, action, next_state, reward):
        q_value_current = self.getQValue(state, action)
        q_value_sample = reward + self.discount * self.getValue(next_state)
        self.qvalues[(state, action)] = (1 - self.learn_rate) * q_value_current + self.learn_rate * q_value_sample

def dict2array(Q, snum, anum):
    # convert Q values from dictionary to array
    qvalues = np.zeros((snum, anum))
    for k, v in Q.items():
        qvalues[k] = v
    return qvalues

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

if __name__ == '__main__':
    maze = Maze()
    agent = QLearningAgent(maze.getLegalActions)

    # init plot
    eval_steps, eval_reward, eval_RMSE = [], [], []
    q_optimal = np.load('VI_Qvalues.npy')

    # Q-learning
    iterations = 5000
    for i in range(iterations):
        state = maze.reset()
        done = False
        while not done:
            action = agent.getAction(state)
            reward, next_state, done = maze.step(state, action)
            agent.learn(state, action, next_state, reward)
            state = next_state

        q_current = dict2array(agent.qvalues, maze.snum, maze.anum)
        avg_step, avg_reward = evaluation(maze, q_current)
        eval_steps.append(avg_step)
        eval_reward.append(avg_reward)
        eval_RMSE.append(rmse(q_current, q_optimal))


    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(np.arange(iterations), eval_steps)
    ax1.set(title = 'Q-Learning', ylabel='steps')
    ax2.plot(np.arange(iterations), eval_reward)
    ax2.set(ylabel='reward')
    ax3.plot(np.arange(iterations), eval_RMSE)
    ax3.set(ylabel='RMSE', xlabel='episodes')
    plt.savefig('qlearning.png')
    # value_plot(dict2array(agent.qvalues, maze.snum, maze.anum), maze)