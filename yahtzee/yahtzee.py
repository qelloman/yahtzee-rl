import logging
import gym
from gym import spaces
import numpy as np


def dec_to_bin_arr(decimal, num_digit):
    bin_arr = [int(i) for i in bin(decimal)[2:]]
    if len(bin_arr) < num_digit:
        return [0] * (num_digit - len(bin_arr)) + bin_arr
    elif len(bin_arr) == num_digit:
        return bin_arr
    else:
        raise ValueError("More digits are required")


def cnt_arr_to_dice_arr(cnt_arr):
    dice_arr = []
    for idx, cnt in enumerate(cnt_arr):
        eye = idx + 1
        dice_arr += [eye] * cnt
    return dice_arr


def dice_arr_to_cnt_arr(dice_arr, num_eyes):
    return np.bincount(dice_arr, minlength=num_eyes + 1)[1:]

category_names = [
    "Ones",
    "Twos",
    "Threes",
    "Fours",
    "Fives",
    "Sixes",
    "Three of a kind",
    "Four of a kind",
    "Full house",
    "Small Straight",
    "Large Straight",
    "Chance",
    "Yahtzee"
]

################################
# | category | name            |
# | 0        | ones            |
# | 1        | twos            |
# | 2        | threes          |
# | 3        | fours           |
# | 4        | fives           |
# | 5        | sixes           |
# | 6        | three of a kind |
# | 7        | four of a kind  |
# | 8        | full house      |
# | 9        | small straight  |
# | 10       | large straight  |
# | 11       | chance          |
# | 12       | yahtzee         |
################################

class YahtzeeEnv(gym.Env):
    def __init__(self, num_dice=5, num_eyes=6, num_category=13, mode='human'):
        self.mode = mode
        self.num_to_roll = 3
        self.num_dice = num_dice
        self.num_eyes = num_eyes
        self.num_category = num_category
        self.category_filled = [0] * self.num_category
        self.category_score = [0] * self.num_category
        self.seed()
        self.dice_count = self.roll_dice()

    def reset(self):
        self.num_to_roll = 3
        self.dice_count = self.roll_dice()
        self.category_filled = [0] * self.num_category
        self.category_score = [0] * self.num_category
        return self._get_obs()

    def step(self, action):
        # assert action in self.available_actions()
        done = False
        reward = 0.0
        # 주사위를 더 굴릴 수 있고, 주사위를 굴리는 action을 취했을 때.
        if self.num_to_roll > 0:
            fixed_pos_arr = dec_to_bin_arr(action, self.num_dice)
            self.dice_count = self.roll_dice_with_fixed_pos_arr(fixed_pos_arr)
            reward = 0
        # 주사위를 더 굴릴 수 없거나 (num_to_roll == 0) 혹은 바로 fill하는 액션을 취했을 때.
        else:
            category = action
            score = self.fill_scoreboard(self.dice_count, category)
            reward = score
            self.render()
            if self.is_finished():
                done = True
            else:
                self.num_to_roll = 3
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
        return self.num_to_roll, tuple(self.category_filled), tuple(self.dice_count)

    def available_actions(self):
        if self.num_to_roll == 0:
            category_to_fill = [i + pow(2, self.num_dice) for i in range(len(self.category_filled)) if
                                self.category_filled[i] == 0]
            return category_to_fill
        else:
            dice_to_keep = [i for i in range(pow(2, self.num_dice))]
            category_to_fill = [i + pow(2, self.num_dice) for i in range(len(self.category_filled)) if
                                self.category_filled[i] == 0]
            return dice_to_keep + category_to_fill

    def roll_dice(self):
        self.num_to_roll -= 1
        dice = np.sort(np.random.randint(1, self.num_eyes + 1, self.num_dice))
        if self.mode == 'human':
            print("You rolled {} dice. => {}".format(self.num_dice, dice))
        else:
            logging.info("You rolled {} dice. => {}".format(self.num_dice, dice))
        dice_count = np.bincount(dice, minlength=self.num_eyes + 1)[1:]
        return dice_count

    def roll_dice_with_fixed_pos_arr(self, fixed_pos_arr):
        self.num_to_roll -= 1
        dice = cnt_arr_to_dice_arr(self.dice_count)
        new_dice = []
        for idx, fix in enumerate(fixed_pos_arr):
            if fix == 1:
                new_dice.append(dice[idx])

        if self.mode == 'human':
            print("You fixed {} dice.".format(new_dice))
        else:
            logging.info("You fixed {} dice.".format(new_dice))

        num_dice_to_roll = self.num_dice - len(new_dice)
        new_dice += list(np.random.randint(1, self.num_eyes + 1, num_dice_to_roll))
        new_dice = np.sort(new_dice)

        if self.mode == 'human':
            print("You got {}.".format(new_dice))
        else:
            logging.info("You got {}.".format(new_dice))
        dice_count = dice_arr_to_cnt_arr(new_dice, self.num_eyes)
        return dice_count

    def check_score(self, dice_count, category):
        if type(category) is not int:
            category = category.item()

        if category < 6:
            score = (category + 1) * dice_count[category]
        elif category == 6:  # three of a kind
            if np.any(dice_count >= 3):
                score = np.sum(cnt_arr_to_dice_arr(dice_count))
            else:
                score = 0
        elif category == 7:  # four of a kin
            if np.any(dice_count >= 4):
                score = np.sum(cnt_arr_to_dice_arr(dice_count))
            else:
                score = 0
        elif category == 8:  # full house
            if np.any(dice_count == 3) and np.any(dice_count == 2):
                score = 25
            else:
                score = 0
            pass
        elif category == 9: # small straight
            if (dice_count[0] > 0 and dice_count[1] > 0 and dice_count[2] > 0 and dice_count[3] > 0) or \
                    (dice_count[1] > 0 and dice_count[2] > 0 and dice_count[3] > 0 and dice_count[4] > 0) or \
                    (dice_count[2] > 0 and dice_count[3] > 0 and dice_count[4] > 0 and dice_count[5] > 0):
                score = 30
            else:
                score = 0
        elif category == 10: # large straight
            if (dice_count[0] > 0 and dice_count[1] > 0 and dice_count[2] > 0 and dice_count[3] > 0 and dice_count[4] > 0) or \
                    (dice_count[1] > 0 and dice_count[2] > 0 and dice_count[3] > 0 and dice_count[4] > 0 and dice_count[5] > 0):
                score = 40
            else:
                score = 0
        elif category == 11: # chance
            score = sum(cnt_arr_to_dice_arr(dice_count))
        elif category == 12:
            if np.any(dice_count == 5):
                score = 50
            else:
                score = 0
        return score

    def fill_scoreboard(self, dice_count, category):
        score = self.check_score(dice_count, category)
        self.category_filled[category] = 1
        self.category_score[category] = score
        if self.mode == 'human':
            print("You filled category [{}].".format(category))
            print("You got {} pts.".format(score))
        else:
            logging.info("You filled category [{}].".format(category))
            logging.info("You got {} pts.".format(score))
        return score

    def is_finished(self):
        return sum(self.category_filled) == self.num_category

    def get_total_score(self):
        total_score = sum(self.category_score)
        return total_score

    def _show_scoreboard(self, showfn):
        showfn("| {:<14} | score |".format("category"))
        for i in range(self.num_category):
            showfn("| {:<15} |    {:02d} |".format(category_names[i], self.category_score[i]))

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
