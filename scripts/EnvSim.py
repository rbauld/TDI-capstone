# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 20:58:57 2017

@author: Reg

This script is for training and validating the reinforcement learning algorithm
"""

import sys
sys.path.insert(0, 'path excluded')
from bb.enviroments import BuySellPerfect
from bb.interfaces import BitInterface_train
from bb.enviroments import SimpleFeature
from pymongo import MongoClient
import numpy as np
from gym import spaces
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge, LSTM, Dropout
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from keras.layers.advanced_activations import LeakyReLU
from bb.features import rsi, cci_short, stoc_osc
import gc
from scipy import stats

# Initalize objects#####################################################################################################

start_ii = 375000
end_ii = 500000
end_buffer = 8000

client = MongoClient('mongodb ip here')
mongoCol = client.bittrex.btc_ltc
train_Int = BitInterface_train(mongoCol, 1, index_start=start_ii, index_end=end_ii, history=1000, fee=0, step=100,
                               book_length=500, end_buffer=end_buffer)

start_ii = 500000
end_ii = 500000 + 70000
test_Int = BitInterface_train(mongoCol, 1, index_start=start_ii, index_end=end_ii, history=1000, fee=0, step=100,
                              book_length=500, end_buffer=end_buffer)


# Build feature1 #######################################################################################################


def build_data(Interface, history_buffer, start, end):
    train = np.zeros((end - start - history_buffer, history_buffer))
    last_list = np.array(Interface.price_list_total)

    for ii in range(train.shape[0]):
        ii_s = ii + history_buffer
        tmp_mean = np.mean(last_list[ii_s + 1 - history_buffer:ii_s + 1])
        tmp_std = np.std(last_list[ii_s + 1 - history_buffer:ii_s + 1])
        train[ii, :] = ((last_list[ii_s + 1 - history_buffer:ii_s + 1] - tmp_mean) / tmp_std)

    trainX = train[:, :]
    trainX = np.mean(trainX.reshape((len(last_list) - history_buffer, 100, 10)), axis=2)

    # Reshape training data
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    return trainX


dataX_train = build_data(train_Int, 1000, 375000, 500000)
dataX_test = build_data(test_Int, 1000, 500000, 500000 + 70000)

drop = 0
model = Sequential()
model.add(LSTM(30, return_sequences=True, input_shape=(100, 1), dropout=drop, kernel_regularizer=l2(0.00)))
model.add(LSTM(30, return_sequences=True, dropout=drop, kernel_regularizer=l2(0)))
model.add(LSTM(30, return_sequences=False, dropout=drop, kernel_regularizer=l2(0)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

# feature_test = model.predict(build_data(test, 1000,start_ii, end_ii), verbose = 1, batch_size=1000)
model.load_weights('feature1A')
feature_trainA = model.predict(dataX_train, verbose=1, batch_size=1000)
model.load_weights('feature1B)
feature_trainB = model.predict(dataX_train, verbose=1, batch_size=1000)

feature_train = (feature_trainA + feature_trainB) / 2

# test
model.load_weights('feature1A')
feature_trainA = model.predict(dataX_test, verbose=1, batch_size=1000)
model.load_weights('feature1B)
feature_trainB = model.predict(dataX_test, verbose=1, batch_size=1000)

feature_test = (feature_trainA + feature_trainB) / 2


# Build enviroment #####################################################################################################

class SimpleFeature(BuySellPerfect):
    """
    This class replaced the 'perfect' feature in BuySellPerfect with an engineered features

    NEEDS UPDATING!
    """

    def __init__(self, BitIntObj, features=None):

        super(SimpleFeature, self).__init__(BitIntObj)

        self.features = features
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(15,))

        self.run_history = []
        self.run_current = []

    def get_observation(self):
        """
        build observation from enviromental data
        """

        price_list = self.BitInt.get_history()
        history = self.BitInt.history

        mean = np.mean(price_list)
        std = np.std(price_list)

        tmp_index = self.BitInt.index

        current_price = np.mean(price_list[-10:])
        current_value = self.curr_val[0]

        current_value = current_value / current_price

        value_diff = current_value - 1

        # Build observation
        observation = []

        # Price features ###########

        # medium time scale features

        # Coin quant
        observation.append(2 * (int(self.BitInt.get_coin() > 0) - 0.5))

        # Feature 1
        future_mean = self.features[tmp_index - history] * std + mean
        price_diff = future_mean / current_price - 1
        observation.append(np.tanh(price_diff * 1180.247 * 2))

        # Diff features
        price_list_norm = np.array(price_list) / current_price
        mean_diff1 = np.mean(np.diff(price_list_norm))
        mean_diff2 = np.mean(np.diff(np.diff(price_list_norm)))

        observation.append(np.tanh(mean_diff1 * 58233.31761 * 2))
        observation.append(np.tanh(mean_diff2 * 409026.7290 * 2))

        observation.append(np.tanh((rsi(price_list) - 0.5) * 77.8216073))
        observation.append(np.tanh((cci_short(price_list) + 0.6667) * 237 / 0.791584))
        observation.append(np.tanh((stoc_osc(price_list) - 0.5) * 3.2 / 0.944009 * 2))  # 7

        # short time scale features
        mean_diff1_short = np.mean(np.diff(price_list_norm[-200:]))
        mean_diff2_short = np.mean(np.diff(np.diff(price_list_norm[-200:])))
        observation.append(np.tanh(mean_diff1_short * 10000 * 5))
        observation.append(np.tanh(mean_diff2_short * 100000 * 5 / 6.086887))
        observation.append(np.tanh((rsi(price_list[-200:]) - 0.5) * 61))
        observation.append(np.tanh((cci_short(price_list[-200:]) + 0.6667) * 237 / 0.394910))
        observation.append(np.tanh((stoc_osc(price_list[-200:]) - 0.5) * 3.2 / 2.219762))

        # quant features ###########

        quant_list = self.BitInt.get_quant_hist()
        current_quant = np.mean(quant_list[-10:])
        quant_list_norm = quant_list / current_quant

        # medium time scale
        mean_diff1 = np.mean(np.diff(quant_list_norm))
        observation.append(np.tanh(mean_diff1 / 0.000109))

        # short time scale
        mean_diff1_short = np.mean(np.diff(quant_list_norm[-100:]))
        observation.append(np.tanh(mean_diff1_short / 0.000139))

        return np.array(observation)

    def _step(self, a):

        done = bool(False)

        value_before = self.BitInt.get_value()

        self.do_action(a)

        # Go to next timestep, calculate reward
        self.BitInt.inc()

        value_after = self.BitInt.get_value()
        value_gained = value_after / value_before - 1

        if value_gained > 0:
            value_gained = value_gained / 1.0156381008851967  # Rescale for class imbalance

        reward = value_gained * 1000
        self.past_reward.append(reward)
        observation = np.append(self.get_observation(), np.mean(self.past_reward))

        if a != 2:
            reward -= 0.01

        # Record values for troubleshooting
        run_data = {}
        run_data['index'] = self.BitInt.index - 1
        run_data['observation'] = observation
        run_data['BTC'] = self.BitInt.get_BTC()
        run_data['coin'] = self.BitInt.get_coin()
        run_data['value_before'] = self.value_before
        run_data['value_after'] = value_after
        run_data['value_gained'] = value_gained
        run_data['action'] = a

        self.run_current.append(run_data)

        return observation, reward, done, {}

    def _reset(self):

        Btc_after = self.BitInt.get_BTC()
        coin_after = self.BitInt.get_coin()
        value = Btc_after + coin_after * self.BitInt.get_last()
        if value < 0.98:
            self.run_history.append(self.run_current)

        observation = super(SimpleFeature, self)._reset()
        self.past_reward = [0]
        self.run_current = []

        return np.append(observation, np.mean(self.past_reward))


train_env = SimpleFeature(train_Int, features=feature_train)
test_env = SimpleFeature(test_Int, features=feature_test)

# Build and train trading model ###############################################


# Get the environment and extract the number of actions.
env = train_env
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Build model
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(30))
actor.add(LeakyReLU(0.01))
actor.add(Dropout(0.5))
actor.add(Dense(30))
actor.add(LeakyReLU(0.01))
actor.add(Dropout(0.5))
actor.add(Dense(30))
actor.add(LeakyReLU(0.01))
actor.add(Dropout(0.5))
actor.add(Dense(30))
actor.add(LeakyReLU(0.01))
actor.add(Dropout(0.5))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))

memory = SequentialMemory(limit=100000, window_length=1)

step_size = 100
env.BitInt.step = step_size
n_steps = end_buffer / step_size
gamma = 0.95

# Do prefitting
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, value_min=0.05, value_test=0,
                              nb_steps=100000)
agent = DQNAgent(model=actor, nb_actions=nb_actions, policy=policy, memory=memory, test_policy=policy,
                 nb_steps_warmup=50, gamma=gamma, target_model_update=10000,
                 train_interval=4, delta_clip=1.)
agent.compile(Adam(lr=.0005, clipnorm=1.), metrics=['mae'])
agent.fit(env, nb_steps=100000, visualize=False, verbose=1, nb_max_episode_steps=n_steps, log_interval=1000)

# Do early stopping fit

test_scores = []
train_scores = []
no_improvement = 0

f = open('log', 'w')
f.write('best'.rjust(10) +
        'test'.rjust(10) +
        'train'.rjust(10) +
        '\n')

f.flush()

# Find test score
agent.test(test_env, nb_episodes=100, nb_max_episode_steps=n_steps, visualize=False)
test_score = np.mean(test_env.value_history[-100:])
test_scores.append(test_score)

# Find train score
agent.test(train_env, nb_episodes=100, nb_max_episode_steps=n_steps, visualize=False)
train_score = np.mean(train_env.value_history[-100:])
train_scores.append(train_score)

best_score = min(test_score, train_score)

f.write('{0:.5f}'.format(best_score).rjust(10) +
        '{0:.5f}'.format(test_score).rjust(10) +
        '{0:.5f}'.format(train_score).rjust(10) +
        +'\n')

for ii in range(0, 300):

    print "Fitting epoch", ii

    # Do fitting iteration
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.05, value_min=0.05, value_test=0,
                                  nb_steps=10000)
    agent = DQNAgent(model=actor, nb_actions=nb_actions, policy=policy, memory=memory, test_policy=policy,
                     nb_steps_warmup=50, gamma=gamma, target_model_update=10000,
                     train_interval=4, delta_clip=1.)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    agent.fit(env, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=n_steps, log_interval=1000)

    # Get test score
    np.random.seed(123)
    env.seed(123)
    agent.test(test_env, nb_episodes=100, nb_max_episode_steps=n_steps, visualize=False)
    test_score = np.mean(test_env.value_history[-100:])
    test_scores.append(test_score)

    # Get train score
    agent.test(train_env, nb_episodes=100, nb_max_episode_steps=n_steps, visualize=False)
    train_score = np.mean(train_env.value_history[-100:])
    train_scores.append(train_score)

    if min(test_score, train_score) > best_score:
        print "saving best model..."
        print "best score", best_score
        print "current score", test_score

        # best_model = model
        best_score = min(test_score, train_score)
        no_improvement = 0

        agent.save_weights('/home/reg/PycharmProjects/btcbot/models/ENVSIM_{}_weights.h5f'.format('v15'),
                           overwrite=True)
    else:
        no_improvement += 1
        print "steps without improvment", no_improvement
        print "best score", best_score
        print "current score", test_score

    f.write('{0:.5f}'.format(best_score).rjust(10) +
            '{0:.5f}'.format(test_score).rjust(10) +
            '{0:.5f}'.format(train_score).rjust(10) +
            '\n')

    f.flush()

    env.value_history = []
    env.run_history = []
    gc.collect()


# Load best model
agent.load_weights('/home/reg/PycharmProjects/btcbot/models/ENVSIM_{}_weights.h5f'.format('v12_redo'))

# Check test set
agent.test(test_env, nb_episodes=500, nb_max_episode_steps=n_steps, visualize=False)
print np.mean(test_env.value_history[-500:])
print stats.sem(test_env.value_history[-500:])

# Check train set
agent.test(train_env, nb_episodes=500, nb_max_episode_steps=n_steps, visualize=False)
print np.mean(train_env.value_history[-500:])
print stats.sem(train_env.value_history[-500:])
