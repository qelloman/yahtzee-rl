import logging

import gym
from gym import spaces

from collections import defaultdict
import numpy as np
import sys


class YahtzeeEnv(gym.Env):
    def __init__(self, num_dice=3, num_dice_face=4, mode='human'):
        self.mode = mode
        self.dice = None
        self.num_dice = num_dice
        self.num_dice_face = num_dice_face
        self.action_space = spaces.Discrete(self.num_dice_face)
        # (combo status, dice status, score)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_dice_face), spaces.Discrete(self.num_dice)))
        self.seed()
        self.category_status = [0] * self.num_dice_face
        self.category_score = [0] * self.num_dice_face

    def reset(self):
        self.category_status = [0] * self.num_dice_face
        self.category_score = [0] * self.num_dice_face
        self.dice = self.roll_dice()
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action)
        ava_categories = [i+1 for i in self.available_actions()]
        category = int(action + 1)
        score = self.update_scoreboard(self.dice, category)
        self.show_action(category, ava_categories, score)

        if self.is_finished():
            done = True
            reward = float(self.get_total_score())
        else:
            done = False
            reward = float(0.0)
            self.dice = self.roll_dice()
        state = self._get_obs()

        return state, reward, done, {}

    def render(self):
        if self.mode == 'human':
            self._show_scoreboard(print)
            print('')
        else:
            self._show_scoreboard(logging.info)
            logging.info('')

    def _get_obs(self):
        return tuple(self.category_status), tuple(self.dice)

    def available_actions(self):
        return [i for i in range(len(self.category_status)) if self.category_status[i] == 0]

    def roll_dice(self):
        dice = np.sort(np.random.randint(1, self.num_dice_face + 1, self.num_dice))
        if self.mode == 'human':
            print("You rolled {} dice. => {}".format(self.num_dice, dice))
        else:
            logging.info("You rolled {} dice. => {}".format(self.num_dice, dice))
        return dice

    def check_score(self, dice, category):
        if category > self.num_dice_face or category < 1:
            print(category)
            raise ValueError("The number is out of range")

        if type(category) != int:
            print(category)
            raise ValueError("The number is wrong type")

        cnt = np.count_nonzero(np.array(dice) == category)
        score = cnt * category
        return score

    def update_scoreboard(self, dice, category):
        score = self.check_score(dice, category)
        category_idx = category - 1
        self.category_status[category_idx] = 1
        self.category_score[category_idx] = score
        return score

    def is_finished(self):
        return len(self.available_actions()) == 0

    def get_total_score(self):
        total_score = sum(self.category_score)
        return total_score

    def _show_scoreboard(self, showfn):
        showfn("| category | score |")
        for i in range(0, self.num_dice_face):
            showfn("|     {}     |  {:02d} |".format(i+1, self.category_score[i]))

    def show_action(self, category, ava_categories, score):
        if self.mode == 'human':
            print("you picked {} out of {}. => score = {}".format(category, ava_categories, score))
        else:
            logging.info("you picked {} out of {}. => score = {}".format(category, ava_categories, score))

    def show_result(self):
        self._show_result(print if self.mode == 'human' else logging.info)

    def _show_result(self, showfn):
        showfn("==== Total Score: {} ====".format(self.get_total_score()))
        showfn("")
