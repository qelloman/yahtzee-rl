
import logging
import gym
from gym import spaces
import numpy as np


class SuperSimpleYahtzeeEnv(gym.Env):
    def __init__(self, num_dice=3, num_eyes=4, mode='human'):
        self.mode = mode
        self.dice_count = None
        self.num_dice = num_dice
        self.num_eyes = num_eyes
        self.action_space = spaces.Discrete(self.num_eyes)
        # (combo status, dice status, score)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_eyes), spaces.Discrete(self.num_dice)))
        self.seed()
        self.category_filled = [0] * self.num_eyes
        self.category_score = [0] * self.num_eyes

    def reset(self):
        self.category_filled = [0] * self.num_eyes
        self.category_score = [0] * self.num_eyes
        self.dice_count = self.roll_dice()
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action)
        ava_categories = [i+1 for i in self.available_actions()]
        category = int(action + 1)
        score = self.update_scoreboard(self.dice_count, category)
        self.show_action(category, ava_categories, score)

        reward = score
        if self.is_finished():
            done = True
        else:
            done = False
            self.dice_count = self.roll_dice()
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
        return tuple(self.category_filled), tuple(self.dice_count)

    def available_actions(self):
        return [i for i in range(len(self.category_filled)) if self.category_filled[i] == 0]

    def roll_dice(self):
        dice = np.sort(np.random.randint(1, self.num_eyes + 1, self.num_dice))
        if self.mode == 'human':
            print("You rolled {} dice. => {}".format(self.num_dice, dice))
        else:
            logging.info("You rolled {} dice. => {}".format(self.num_dice, dice))
        dice_count = np.bincount(dice, minlength=self.num_eyes + 1)[1:]
        return dice_count

    def check_score(self, dice_count, category):
        if category > self.num_eyes or category < 1:
            print(category)
            raise ValueError("The number is out of range")

        if type(category) != int:
            print(category)
            raise ValueError("The number is wrong type")
        category_idx = category - 1
        score = category * dice_count[category_idx]
        return score

    def update_scoreboard(self, dice_count, category):
        score = self.check_score(dice_count, category)
        category_idx = category - 1
        self.category_filled[category_idx] = 1
        self.category_score[category_idx] = score
        return score

    def is_finished(self):
        return len(self.available_actions()) == 0

    def get_total_score(self):
        total_score = sum(self.category_score)
        return total_score

    def _show_scoreboard(self, showfn):
        showfn("| category | score |")
        for i in range(0, self.num_eyes):
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
