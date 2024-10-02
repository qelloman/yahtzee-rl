import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from yahtzee import YahtzeeEnv

# 랜덤 시드 고정
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def process_observation(obs):
    # 주사위 값 (5개)
    dice = obs[:5]
    # 카테고리 점수 (13개)
    categories = obs[5:18]
    # 남은 굴림 횟수 (1개)
    rolls_left = obs[18]

    # 1. 주사위 값에 대한 원-핫 인코딩
    dice_one_hot = np.eye(6)[dice - 1].flatten()  # 6x5 = 30 features

    # 2. 카테고리 점수 정규화
    max_scores = np.array([30, 30, 30, 30, 30, 30, 30, 30, 25, 30, 40, 50, 30])
    normalized_categories = categories / max_scores

    # 3. 남은 굴림 횟수에 대한 원-핫 인코딩
    rolls_left_one_hot = np.eye(3)[2 - rolls_left]  # 3 features

    # 모든 처리된 특성을 하나의 벡터로 결합
    processed_obs = np.concatenate([
        dice_one_hot,           # 30 features
        normalized_categories,  # 13 features
        rolls_left_one_hot      # 3 features
    ])

    return processed_obs

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

def safe_normalize(probs, mask, epsilon=1e-10):
    masked_prob = probs * mask
    sum_probs = masked_prob.sum(dim=-1, keepdim=True)
    
    # 합이 epsilon보다 작으면 균등 분포로 설정
    is_small = sum_probs < epsilon
    safe_sum = torch.where(is_small, torch.ones_like(sum_probs), sum_probs)
    
    normalized = masked_prob / safe_sum
    
    # 합이 0인 경우 마스크 내에서 균등 분포로 설정
    num_valid = mask.sum(dim=-1, keepdim=True)
    uniform_prob = mask / torch.max(num_valid, torch.ones_like(num_valid))
    normalized = torch.where(is_small, uniform_prob, normalized)
    
    return normalized


class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, value_coef, entropy_coef):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = PPOActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def get_action(self, state, valid_actions):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, _ = self.actor_critic(state)
        
        # 유효하지 않은 액션의 확률을 0으로 설정
        mask = torch.zeros_like(action_probs).to(self.device)
        mask[0, valid_actions] = 1
        masked_probs = safe_normalize(action_probs, mask)
        if masked_probs.sum() == 0:
            print("Zero masekd probs")
        masked_probs = masked_probs / masked_probs.sum()  # 확률 재정규화

        dist = Categorical(masked_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, states, actions, old_log_probs, returns, advantages, valid_actions_list, writer, step):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        # returns = torch.FloatTensor(returns)
        # advantages = torch.FloatTensor(advantages).to(self.device)

        for _ in range(1):  # PPO 업데이트 횟수
            action_probs, values = self.actor_critic(states)
            
            # 유효한 액션만 고려
            masked_probs = []
            for probs, valid_actions in zip(action_probs, valid_actions_list):
                mask = torch.zeros_like(probs)
                mask[valid_actions] = 1
                
                masked_prob = safe_normalize(probs, mask)
                masked_probs.append(masked_prob)

            new_masked_probs = torch.stack(masked_probs)
            dist = Categorical(new_masked_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (returns - values).pow(2).mean()

            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()

            max_norm = 1.0  # 최대 노름 값 설정
            clip_grad_norm_(self.actor_critic.parameters(), max_norm)

            self.optimizer.step()
            writer.add_scalar("train/actor_loss", actor_loss.item(), global_step=step)
            writer.add_scalar("train/critic_loss", critic_loss.item(), global_step=step)

def train_ppo():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/ppo/{current_time}"
    writer = SummaryWriter(log_dir=log_dir)

    env = YahtzeeEnv()
    eval_env = YahtzeeEnv()
    eval_freq = 100
    state_dim = 46  # process_observation 함수의 출력 차원
    action_dim = 44  # Yahtzee 환경의 액션 수
    ppo = PPO(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01)

    # 학습 루프
    for episode in tqdm(range(10000)):

        # PPO update here
        # please write PPO update code here
        # Collect trajectories and compute returns and advantages
        states, actions, rewards, log_probs, valid_actions_list = [], [], [], [], []
        state, info = env.reset()
        done, trunc = False, False
        while not done and not trunc:
            processed_state = process_observation(state)
            valid_actions = info['valid_actions']
            action, log_prob = ppo.get_action(processed_state, valid_actions)
            next_state, reward, done, trunc, info = env.step(action)

            states.append(processed_state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob.item())
            valid_actions_list.append(valid_actions)

            state = next_state

        # Compute returns and advantages
        returns, advantages = [], []
        G = 0
        for reward in reversed(rewards):
            G = reward + ppo.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(ppo.device)
        values = ppo.actor_critic.critic(torch.FloatTensor(states).to(ppo.device)).squeeze()
        advantages = returns - values.detach().squeeze()

        # Update PPO
        ppo.update(states, actions, log_probs, returns, advantages, valid_actions_list, writer, episode)


        if episode % eval_freq == 0:
            pass
            # please write evaluation code here
            # for name, params in ppo.actor_critic.named_parameters():
            #     print(name, params)

            # 평가 코드 예시
            eval_rewards = []
            for _ in range(100):
                state, info = eval_env.reset()
                done, trunc = False, False
                total_reward = 0
                while not done and not trunc:
                    processed_state = process_observation(state)
                    valid_actions = info['valid_actions']
                    action, _ = ppo.get_action(processed_state, valid_actions)
                    next_state, reward, done, trunc, info = eval_env.step(action)
                    total_reward += reward
                    state = next_state
                eval_rewards.append(total_reward)

            # 평가 결과 출력
            print(f"Episode {episode}, Evaluation Reward: {np.mean(eval_rewards)}")


def test_process_observation():
    # 사용 예시
    sample_obs = np.array([1, 2, 3, 4, 5] + [0]*13 + [3])  # 예시 observation
    processed = process_observation(sample_obs)
    print(f"Original observation shape: {sample_obs.shape}")
    print(f"Processed observation shape: {processed.shape}")
    print(f"Processed observation: {processed}")


if __name__ == "__main__":
    train_ppo()