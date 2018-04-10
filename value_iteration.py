from maze import *
from value_plot import *
import numpy as np

class ValueIterationAgent(object):
    """
        A Agent takes a MDP and runs value iteration using a discount factor
        for a given number of iterations
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = np.zeros(self.mdp.snum)
        self.qvalues = np.zeros((self.mdp.snum, self.mdp.anum))

        # value iteration
        for i in range(self.iterations):
            temp_value = np.zeros(self.mdp.snum)
            for state in range(self.mdp.snum):
                if self.mdp.idx2cell[int(state/8)] == self.mdp.goal_pos: # terminal state
                    continue
                max_value = float('-inf')
                for action in range(self.mdp.anum):
                    total_value = self.getQValue(state, action)
                    max_value = max(max_value, total_value)
                temp_value[state] = max_value
            self.values = np.copy(temp_value)  # update for next iteration

        # compute Q-value from optimal value
        for state in range(self.mdp.snum):
            for action in range(self.mdp.anum):
                self.qvalues[state, action] = self.getQValue(state, action)


    def getValue(self, state):
        return self.values[state]

    def getQValue(self, state, action):
        total_value = 0
        for reward, next_state, probability in self.mdp.getTransitionsAndRewards(state, action):
            total_value += probability * (reward + self.discount * self.getValue(next_state))
        return total_value

    def getPolicy(self, state):
        optimal_policy = np.argmax(self.qvalues[state,:])
        return optimal_policy

    def getAction(self, state):
        return self.getPolicy(state)


if __name__ == '__main__':
    maze = Maze()
    agent = ValueIterationAgent(maze)
    np.save('VI_Qvalues',agent.qvalues)
    value_plot(agent.qvalues, maze)
