# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 20:58:57 2017

@author: Reg

This script is for training and validating the reinforcement learning algorithm
"""

import sys
sys.path.insert(0, 'path excluded')
from bb.interfaces import BitInterface_train
from bb.enviroments import SimpleFeature
from pymongo import MongoClient
import numpy as np
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from keras.layers.advanced_activations import LeakyReLU
import gc


# Initalize objects#####################################################################################################
start_ii = 375000
end_ii = 500000
end_buffer = 8000

client = MongoClient('mongodb://127.0.0.1:27017')

mongoCol = client.bittrex.btc_ltc
train_Int = BitInterface_train(mongoCol, 1,index_start=start_ii, index_end=end_ii, history=1000, fee=0, step=100,
                               book_length=500, end_buffer=end_buffer)

start_ii = 500000
end_ii = 500000+70000
test_Int = BitInterface_train(mongoCol, 1,index_start=start_ii, index_end=end_ii, history=1000, fee=0, step=100,
                               book_length=500, end_buffer=end_buffer)


# Build enviroment #####################################################################################################

train_env = SimpleFeature(train_Int)
test_env = SimpleFeature(test_Int)


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
n_steps = end_buffer/step_size
gamma = 0.95

# Do prefitting
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, value_min=0.05, value_test=0, nb_steps=100000)
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
f.write('best'.rjust(10)+
        'test'.rjust(10)+
        'train'.rjust(10)+
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

f.write('{0:.5f}'.format(best_score).rjust(10)+
        '{0:.5f}'.format(test_score).rjust(10)+
        '{0:.5f}'.format(train_score).rjust(10)+
        +'\n')


for ii in range(0, 300):
    
    print "Fitting epoch", ii
    
    # Do fitting iteration
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.05, value_min=0.05, value_test=0, nb_steps=10000)
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

        best_score = min(test_score, train_score)
        no_improvement = 0
        agent.save_weights('enter save directory here', overwrite=True)

    else:
        no_improvement += 1
        print "steps without improvment", no_improvement
        print "best score", best_score
        print "current score", test_score
    
    f.write('{0:.5f}'.format(best_score).rjust(10)+
            '{0:.5f}'.format(test_score).rjust(10)+
            '{0:.5f}'.format(train_score).rjust(10)+
            '\n') 
    
    f.flush()

    env.value_history = []
    env.run_history = []
    gc.collect()

