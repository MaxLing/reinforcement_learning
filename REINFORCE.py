import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class PolicyEstimator():
    """
        A Policy Function Approximator
    """
    def __init__(self, state_size, action_size, learn_rate=0.001, scope="policy", layer1_size=5, layer2_size=5):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, shape=[None, state_size])
            self.action = tf.placeholder(tf.int32, shape=[None])
            self.target = tf.placeholder(tf.float32, shape=[None])

            fc1 = tf.layers.dense(self.state, units=layer1_size, activation=tf.nn.relu)
            fc2 = tf.layers.dense(fc1, units=layer2_size, activation=tf.nn.relu)

            self.action_softmax = tf.squeeze(tf.nn.softmax(tf.layers.dense(fc2, units=action_size)))
            action_mask = tf.one_hot(self.action, action_size)
            self.action_pred = tf.reduce_sum(self.action_softmax*action_mask, 1)

            self.loss = -tf.log(self.action_pred) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
            self.train = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_softmax, feed_dict={self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target, self.action: action}
        _, loss = sess.run([self.train, self.loss], feed_dict)
        return loss


class ValueEstimator():
    """
        A Value Function Approximator
    """
    def __init__(self, state_size, learn_rate=0.001, scope="value", layer1_size=5, layer2_size=5):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, shape=[None, state_size])
            self.target = tf.placeholder(tf.float32, shape=[None])

            fc1 = tf.layers.dense(self.state, units=layer1_size, activation=tf.nn.relu)
            fc2 = tf.layers.dense(fc1, units=layer2_size, activation=tf.nn.relu)

            self.value = tf.squeeze(tf.layers.dense(fc2, units=1))

            self.loss = tf.square(self.target - self.value)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
            self.train = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value, feed_dict={self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train, self.loss], feed_dict)
        return loss


def REINFORCE(env, policy_estimator, value_estimator, iterations, discount = 0.9):
    history = []
    for i in range(iterations):
        state = env.reset()
        episode = []
        for t in range(env._max_episode_steps):
            if (i+1)%100 == 0:
                env.render() # render episode
            action_softmax = policy_estimator.predict(state.reshape((1,-1)))
            action = np.random.choice(np.arange(len(action_softmax)), p=action_softmax)
            next_state, reward, done, _ = env.step(action)

            if done:
                episode.append([state, action, next_state, 1000, done]) # finish reward
                break
            episode.append([state, action, next_state, reward, done])
            state = next_state
        history.append((sum([step[3] for step in episode]),t+1))
        print("Episode " + str(i + 1) + " finished after " +str(t+1)+ " steps")

        # training batch: 1 episode
        states = np.asarray([episode[t][0] for t in range(len(episode))])
        actions = np.asarray([episode[t][1] for t in range(len(episode))])

        total_returns = np.asarray([sum([discount ** i * r for i, (s, a, s1, r, d) in enumerate(episode[t:])])
                                    for t in range(len(episode))])
        baselines = value_estimator.predict(states)
        advantages = total_returns - baselines

        value_estimator.update(states, total_returns)
        policy_estimator.update(states, advantages, actions)


    return history

if __name__ == '__main__':
    # # acrobot
    # env = gym.make('Acrobot-v1')
    # policy_estimator = PolicyEstimator(state_size=6, action_size=3)
    # value_estimator = ValueEstimator(state_size=6)

    # mountain car
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 2000
    policy_estimator = PolicyEstimator(state_size=2, action_size=3)
    value_estimator = ValueEstimator(state_size=2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        history = REINFORCE(env, policy_estimator, value_estimator, iterations=1000)

    # smoothing
    window = 10
    smooth_history = np.asarray([np.mean(history[i*window:(i+1)*window], axis=0) for i in range(int(len(history)/window))])

    # plot
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(np.arange(0, len(history), window), smooth_history[:,0])
    ax1.set(title = 'Policy Gradient - MountainCar', ylabel='reward')
    ax2.plot(np.arange(0, len(history), window), smooth_history[:,1])
    ax2.set(ylabel='length', xlabel='episodes')
    plt.savefig('pg_MountainCar.png')