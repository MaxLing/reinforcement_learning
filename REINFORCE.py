import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class PolicyEstimator():
    """
        A Policy Function Approximator
    """
    def __init__(self, state_size, action_size, learn_rate=0.002, scope="policy", layer1_size=5, layer2_size=5):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, shape=[None, state_size])
            self.action = tf.placeholder(tf.int32, shape=[None])
            self.target = tf.placeholder(tf.float32, shape=[None])

            fc1 = tf.layers.dense(self.state, units=layer1_size, activation=tf.nn.relu)
            # fc2 = tf.layers.dense(fc1, units=layer2_size, activation=tf.nn.relu)

            self.action_softmax = tf.squeeze(tf.nn.softmax(tf.layers.dense(fc1, units=action_size)))
            action_mask = tf.one_hot(self.action, action_size)
            self.action_pred = tf.reduce_sum(self.action_softmax*action_mask, 1)

            self.loss = -tf.log(self.action_pred) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
            self.train = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_softmax, feed_dict={self.state: [state]})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: [state], self.target: [target], self.action: [action]}
        _, loss = sess.run([self.train, self.loss], feed_dict)
        return loss


class ValueEstimator():
    """
        A Value Function Approximator
    """
    def __init__(self, state_size, learn_rate=0.002, scope="value", layer1_size=5, layer2_size=5):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, shape=[None, state_size])
            self.target = tf.placeholder(tf.float32, shape=[None])

            fc1 = tf.layers.dense(self.state, units=layer1_size, activation=tf.nn.relu)
            # fc2 = tf.layers.dense(fc1, units=layer2_size, activation=tf.nn.relu)

            self.value = tf.squeeze(tf.layers.dense(fc1, units=1))

            self.loss = tf.square(self.target - self.value)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
            self.train = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value, feed_dict={self.state: [state]})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: [state], self.target: [target]}
        _, loss = sess.run([self.train, self.loss], feed_dict)
        return loss

def REINFORCE(env, policy_estimator, value_estimator, iterations, discount = 0.9):
    history = []
    for i in range(iterations):
        state = env.reset()
        episode = []
        for t in range(env._max_episode_steps):
            # env.render()
            action_softmax = policy_estimator.predict(state)
            action = np.random.choice(np.arange(len(action_softmax)), p=action_softmax)
            next_state, reward, done, _ = env.step(action)

            episode.append([state, action, next_state, reward, done])
            if done:
                break
            state = next_state
        history.append((sum([step[3] for step in episode])/len(episode),t+1))
        print("Episode " + str(i + 1) + " finished after " +str(t+1)+ " steps")

        for t in range(len(episode)):
            total_return = sum([discount ** i * r for i, (s, a, s1, r, d) in enumerate(episode[t:])])
            state, action, next_state, reward, done = episode[t]
            baseline = value_estimator.predict(state)
            advantage = total_return - baseline
            value_estimator.update(state, total_return)
            policy_estimator.update(state, advantage, action)

    return history

if __name__ == '__main__':
    # # acrobot
    # env = gym.make('Acrobot-v1')
    # policy_estimator = PolicyEstimator(state_size=6, action_size=3)
    # value_estimator = ValueEstimator(state_size=6)
    # mountain car
    env = gym.make('MountainCar-v0')
    policy_estimator = PolicyEstimator(state_size=2, action_size=3)
    value_estimator = ValueEstimator(state_size=2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        history = REINFORCE(env, policy_estimator, value_estimator, iterations=100)

    # smoothing
    window = 10
    smooth_history = np.asarray([np.mean(history[i*window:(i+1)*window], axis=0) for i in range(int(len(history)/window))])

    # plot
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(np.arange(0, len(history), window), smooth_history[:,0])
    ax1.set(title = 'Policy Gradient - MountainCar', ylabel='avg reward')
    ax2.plot(np.arange(0, len(history), window), smooth_history[:,1])
    ax2.set(ylabel='length', xlabel='episodes')
    plt.savefig('pg_mountainCar.png')