import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Seed
seed = 42
np.random.seed(seed)


class YahtzeeOneShotEnv(gym.Env):
    def __init__(self):
        super(YahtzeeOneShotEnv, self).__init__()
        
        self.num_dice = 5
        self.dice = np.zeros(self.num_dice, dtype=int)
        
        self.categories = [
            'Ones',
            'Twos',
            'Threes',
            'Fours',
            'Fives',
            'Sixes',
            # 'Three of a Kind',
            # 'Four of a Kind',
            # 'Full House',
            # 'Small Straight',
            # 'Large Straight',
            # 'Yahtzee',
            # 'Chance'
        ]

        self.num_categories = len(self.categories)
        self.scorecard = np.zeros(self.num_categories, dtype=int)
        
        self.rolls_left = 3

        # Define the observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(33,), dtype=np.float32)

        # Define the action space
        # Actions 0-30: Rolling dice with different hold combinations (2^5 - 1 = 31)
        # Actions 31-43: Choosing a category to score (13 categories)
        self.action_space = spaces.Discrete(pow(2, self.num_dice) - 1 + self.num_categories)

    def _reset(self):
        self.rolls_left = 3
        self._roll_dice()
            
    def _roll_dice(self, hold=[0, 0, 0, 0, 0]):
        for i in range(self.num_dice):
            if not hold[i]:
                self.dice[i] = np.random.randint(1, 7)

        self.rolls_left -= 1
        
    def _get_observation(self):
        # 주사위 값에 대한 원-핫 인코딩
        dice_one_hot = np.eye(6)[self.dice - 1].flatten()  # 6x5 = 30 features

        # 남은 굴림 횟수에 대한 원-핫 인코딩
        rolls_left_one_hot = np.eye(3)[self.rolls_left]  # 3 features

        # 모든 처리된 특성을 하나의 벡터로 결합
        obs = np.concatenate([dice_one_hot, rolls_left_one_hot])

        return obs
    
    def _get_action_mask(self):
        # action 0~30: 주사위 굴리기
        # action 31~43: 카테고리 선택
        action_mask = np.zeros(pow(2, self.num_dice) - 1 + self.num_categories, dtype=int)
        if self.rolls_left > 0:
            action_mask[:pow(2, self.num_dice) - 1] = 1
        
        for i in range(self.num_categories):
            if self.category_filled[i] == 0:
                idx = pow(2, self.num_dice) - 1 + i
                action_mask[idx] = 1

        return action_mask
    
    def _get_info(self):
        info = {}
        info['action_mask'] = self._get_action_mask()
        return info
    
    def _calculate_score(self, category):
        counts = np.bincount(self.dice, minlength=7)
        total = sum(self.dice)
        
        if category == 'Ones':
            return counts[1] * 1
        elif category == 'Twos':
            return counts[2] * 2
        elif category == 'Threes':
            return counts[3] * 3
        elif category == 'Fours':
            return counts[4] * 4
        elif category == 'Fives':
            return counts[5] * 5
        elif category == 'Sixes':
            return counts[6] * 6
        elif category == 'Three of a Kind':
            if any(count >= 3 for count in counts[1:]):
                return total
            return 0
        elif category == 'Four of a Kind':
            if any(count >= 4 for count in counts[1:]):
                return total
            return 0
        elif category == 'Full House':
            if sorted(counts[1:])[-2:] == [2, 3]:
                return 25
            return 0
        elif category == 'Small Straight':
            straights = [counts[i:i+4] for i in range(1, 4)]
            if any(np.all(s >= 1) for s in straights):
                return 30
            return 0
        elif category == 'Large Straight':
            straights = [counts[i:i+5] for i in range(1, 3)]
            if any(np.all(s >= 1) for s in straights):
                return 40
            return 0
        elif category == 'Yahtzee':
            if any(count == 5 for count in counts[1:]):
                return 50
            return 0
        elif category == 'Chance':
            return total
        else:
            return 0
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.scorecard = np.zeros(self.num_categories, dtype=int)
        self.category_filled = np.zeros(self.num_categories, dtype=int)
        self._reset()
        
        return self._get_observation(), self._get_info()
            
    def step(self, action):
        done = False
        reward = 0
        
        # 액션 마스크를 확인하여 유효하지 않은 액션인지 검사
        action_mask = self._get_action_mask()
        if action_mask[action] == 0:
            # 유효하지 않은 액션 선택 시 큰 페널티 부여
            reward = -50
        else:
            # 주사위 굴리기 액션
            if action < pow(2, self.num_dice) - 1:
                # hold_mask 계산
                hold_mask = [int(bool(action & (1 << (self.num_dice - 1 - i)))) for i in range(self.num_dice)]
                # 주사위 굴리기
                self._roll_dice(hold_mask)
            else:
                # 카테고리 선택 액션
                category_idx = action - (pow(2, self.num_dice) - 1)
                
                # 점수 계산 및 점수판 업데이트
                score = self._calculate_score(self.categories[category_idx])
                self.scorecard[category_idx] = score
                self.category_filled[category_idx] = 1
                reward = score
                
                # 모든 카테고리가 채워졌으면 에피소드 종료
             
                done = True
             
        
        reward = reward / 50.0  # 정규화
                
        return self._get_observation(), reward, done, False, self._get_info()
    

    def render(self):
        print("현재 주사위 상태:", self.dice)
        print("점수판:")
        for category, score in zip(self.categories, self.scorecard):
            print(f"{category}: {score if score is not None else '미채점'}")
        print(f"남은 롤링 횟수: {self.rolls_left}")
        print("-" * 30)
        print("-" * 30)


def test_env():
    env = YahtzeeOneShotEnv()
    obs, info = env.reset()
    print("초기 관측값:", obs)
    print("초기 정보:", info)
    
    done = False
    while not done:
        action = np.random.choice(range(44))
        obs, reward, done, _, info = env.step(action)
        print("액션:", action)
        print("다음 관측값:", obs)
        print("보상:", reward)
        print("정보:", info)
        env.render()
        print()

def test_env_with_human_input():
    env = YahtzeeOneShotEnv()
    obs, info = env.reset()
    
    
    done = False
    while not done:
        env.render()
        action = int(input("액션을 입력하세요: "))
        obs, reward, done, _, info = env.step(action)
        print("보상:", reward)
        print("정보:", info)
        

if __name__ == "__main__":
    test_env_with_human_input()
