import os
import sys
import random
import time
import logging
import json
from collections import defaultdict
from itertools import product
from multiprocessing import Pool
from tempfile import NamedTemporaryFile

import pandas as pd
import click
from tqdm import tqdm as _tqdm

import matplotlib.pyplot as plt

tqdm = _tqdm

from simple_yahtzee import YahtzeeEnv
from human_agent import HumanAgent

DEFAULT_VALUE = 0
EPISODE_CNT = 17000
BENCH_EPISODE_CNT = 3000
MODEL_FILE = 'best_td_agent.dat'
EPSILON = 0.08
ALPHA = 0.4
CWD = os.path.dirname(os.path.abspath(__file__))

state_action_values = defaultdict(lambda: 0)


def reset_state_values():
    global state_action_values
    state_action_values = defaultdict(lambda: 0)


def set_state_action_value(state, action, value):
    state_action_values[(state, action)] = value


def best_val_indices(values, fn):
    best = fn(values)
    return [i for i, v in enumerate(values) if v == best]


class QLearningAgent(object):
    def egreedy_policy(self, state, ava_actions, epsilon):
        e = random.random()
        if e < epsilon:
            action = self.random_action(ava_actions)
        else:
            action = self.greedy_action(state, ava_actions)
        return action

    def random_action(self, ava_actions):
        return random.choice(ava_actions)

    def greedy_action(self, state, ava_actions):
        assert len(ava_actions) > 0

        ava_values = []
        for action in ava_actions:
            state_action_val = self.ask_value(state, action)
            ava_values.append(state_action_val)

        indices = best_val_indices(ava_values, max)

        # tie breaking by random choice
        aidx = random.choice(indices)

        action = ava_actions[aidx]

        return action

    def ask_value(self, state, action):
        return state_action_values[(state, action)]

    def ask_max_value(self, state, max_ava_actions):
        values = [state_action_values[(state, action)] for action in max_ava_actions]
        return max(values)


def learn(max_episode=50000, eps_start=1.0, eps_decay=0.99995, eps_min=0.3, alpha=0.4, gamma=1, save_file='state_action_value.dat'):
    reset_state_values()
    env = YahtzeeEnv(3, 3, 'learn')
    agent = QLearningAgent()
    eps = eps_start
    for i in tqdm(range(max_episode)):
        episode = i + 1

        # reset agent for new episode
        agent.episode_rate = episode / float(max_episode)

        state = env.reset()
        done = False
        while not done:
            ava_actions = env.available_actions()
            eps = max(eps * eps_decay, eps_min)

            action = agent.egreedy_policy(state, ava_actions, eps)
            # update (no rendering)
            nstate, reward, done, info = env.step(action)

            if done:
                env.show_result()
                # set terminal state value
                set_state_action_value(state, action, reward)
            else:
                val = agent.ask_value(state, action)

                max_ava_actions = [i for i in range(env.num_dice_face)]
                target_val = reward + gamma * agent.ask_max_value(nstate, max_ava_actions)
                new_val = val + alpha * (target_val - val)
                set_state_action_value(state, action, new_val)

            state = nstate

    # save states
    save_model(save_file, max_episode, alpha)
    # print(state_action_values)


def save_model(save_file, max_episode, alpha):
    with open(save_file, 'wt') as f:
        # write model info
        info = dict(type="td", max_episode=max_episode, alpha=alpha)
        # write state values
        f.write('{}\n'.format(json.dumps(info)))
        for key, value in state_action_values.items():
            state, action = key
            f.write('{}\t{}\t{:f}\n'.format(state, action, value))


def load_model(filename):
    with open(filename, 'rb') as f:
        # read model info
        info = json.loads(f.readline().decode('ascii'))
        for line in f:
            elms = line.decode('ascii').split('\t')
            state = eval(elms[0])
            action = eval(elms[1])
            value = eval(elms[2])
            state_action_values[(state, action)] = value
    return info


def play(load_file, num_episode):
    load_model(load_file)
    env = YahtzeeEnv(3, 3, 'human')
    agent = QLearningAgent()  # prevent exploring
    total_scores = []
    for i in range(num_episode):
        # start agent rotation
        state = env.reset()
        done = False
        while not done:
            ava_actions = env.available_actions()
            action = agent.greedy_action(state, ava_actions)

            state, reward, done, info = env.step(action)
            env.render()

            if done:
                env.show_result()
                total_scores.append(env.get_total_score())
                break

    fig, ax = plt.subplots(1,1)
    ax.plot(total_scores)
    plt.show()
    print(sum(total_scores)/len(total_scores))


def load_and_print(filename='state_action_value.dat'):
    with open(filename, 'rb') as f:
        # read model info
        info = json.loads(f.readline().decode('ascii'))
        for line in f:
            print(line)


if __name__ == '__main__':
    # learn(5000000)
    play('state_action_value.dat', 100)
    # load_and_print()