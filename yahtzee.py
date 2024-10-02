import gymnasium as gym
import numpy as np
from gymnasium import spaces


class YahtzeeEnv(gym.Env):
    def __init__(self):
        super(YahtzeeEnv, self).__init__()
        
        # Action space: 31 (2^5 - 1 combinations for rerolling dice) + 13 (categories) = 44
        self.action_space = spaces.Discrete(44)
        
        # Observation space: 5 dice + 13 categories + rolls left
        self.observation_space = spaces.Box(low=0, high=6, shape=(19,), dtype=np.int32)
        
        self.dice = np.zeros(5, dtype=np.int32)
        self.categories = np.zeros(13, dtype=np.int32)
        self.rolls_left = 3
        self.current_round = 0
        self.current_step = 0
        self.max_step = 50
        
    def reset(self):
        self.dice = np.zeros(5, dtype=np.int32)
        self.categories = np.zeros(13, dtype=np.int32)
        self.category_filled = np.zeros(13, dtype=np.int32)
        self.rolls_left = 3
        self.current_round = 0
        self.current_step = 0
        self._roll_dice()
        obs = self._get_observation()
        info = {'valid_actions': self._get_valid_actions()}
        return obs, info

    def _get_valid_actions(self):
        valid_actions = []
        if self.rolls_left >= 1:
            for i in range(31):
                valid_actions.append(i)

        for i in range(31, 44):
            category = i - 31
            if self.category_filled[category] == 0:
                valid_actions.append(i)
        return valid_actions
    
    def step(self, action):
        reward = 0
        done = False
        trunc = True if self.current_step >= self.max_step else False
        info = {}

        if action < 31:  # Reroll dice
            if self.rolls_left >= 1:
                keep_position = [bool(action & (1 << i)) for i in range(5)]
                self._roll_dice(keep_position)
            else:
                # 이미 3번 다 굴렸는데 또 굴리려고 하는 것.
                reward = -500  # Penalty for invalid action
        else:  # Choose category
            category = action - 31
            if self.category_filled[category] == 0:
                self.category_filled[category] = 1
                score = self._calculate_score(category)
                self.categories[category] = score
                reward = score
                self.current_round += 1
                self.rolls_left = 3
                self._roll_dice()
                
                if self.current_round == 13:
                    done = True
            else:
                # 똑같은 카테고리에 또 넣으려고 하는 것.
                reward = -500  # Penalty for invalid action

        info['valid_actions'] = self._get_valid_actions()
        obs = self._get_observation()
        self.current_step += 1
        return obs, reward, done, trunc, info
    
    def _get_observation(self):
        return np.concatenate([self.dice, self.categories, [self.rolls_left]])
    
    def _roll_dice(self, keep_position=None):
        if keep_position is None:
            self.dice = np.random.randint(1, 7, 5)
        else:
            for i in range(5):
                if not keep_position[i]:
                    self.dice[i] = np.random.randint(1, 7)
        self.rolls_left -= 1
    
    def _calculate_score(self, category):
        if category < 6:  # Upper section
            return np.sum(self.dice[self.dice == category + 1])
        elif category == 6:  # Three of a kind
            return np.sum(self.dice) if np.any(np.bincount(self.dice)[1:] >= 3) else 0
        elif category == 7:  # Four of a kind
            return np.sum(self.dice) if np.any(np.bincount(self.dice)[1:] >= 4) else 0
        elif category == 8:  # Full house
            counts = np.bincount(self.dice)[1:]
            return 25 if (2 in counts and 3 in counts) else 0
        elif category == 9:  # Small straight
            return 30 if len(set(self.dice)) >= 4 and ({1,2,3,4} <= set(self.dice) or {2,3,4,5} <= set(self.dice) or {3,4,5,6} <= set(self.dice)) else 0
        elif category == 10:  # Large straight
            return 40 if len(set(self.dice)) == 5 and ({1,2,3,4,5} <= set(self.dice) or {2,3,4,5,6} <= set(self.dice)) else 0
        elif category == 11:  # Yahtzee
            return 50 if np.any(np.bincount(self.dice)[1:] == 5) else 0
        elif category == 12:  # Chance
            return np.sum(self.dice)