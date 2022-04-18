
import time
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

now_str = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = './a2c/' + now_str
writer = SummaryWriter(log_dir=log_dir)

from tqdm import tqdm
from simple_yahtzee import SimpleYahtzeeEnv2

from torch.distributions import Categorical
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def state_to_tensor(state):
    num_to_roll, category_filled, dice_count = state
    category_filled = list(category_filled)
    dice_count = list(dice_count)
    tensor_out = torch.tensor([num_to_roll] + category_filled + dice_count, dtype=torch.float32)
    tensor_out = tensor_out.view(-1, tensor_out.size(0))
    return tensor_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float)
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
num_dice = 4
num_dice_face = 6
epoch_cnt = 0
LEARNING_RATE = 0.0002


class A2CNetwork(nn.Module):
    def __init__(self, num_dice, num_eyes, num_categories):
        super().__init__()
        self.input_shapes = [1, 6, num_categories, 2]
        self.keep_action_space = 2 ** num_dice
        self.num_categories = num_categories
        self.use_cuda = torch.cuda.is_available()
        self.player_scorecard_idx = slice(sum(self.input_shapes[:1]), sum(self.input_shapes[:2]))

        self.common_net = nn.Sequential(
            nn.Linear(sum(self.input_shapes[:3]), 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
        )
        self.lin_value = nn.Linear(128,1)
        self.lin_keep = nn.Linear(128, self.keep_action_space)
        self.lin_category = nn.Linear(128, num_categories)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):

        rolls_left = inputs[..., 0, None] # None으로 차원을 늘려줌.

        x = self.common_net(inputs)

        out_value = self.lin_value(x)

        out_keep = self.lin_keep(x)

        out_category = self.lin_category(x)
        # Check what caused error
        out_category = torch.where(inputs[..., self.player_scorecard_idx] == 1, -float('inf'), out_category.double())

        def pad_action(logit, num_pad):
            pad_shape = (0, num_pad) + (0, 0) * (logit.ndim - 1)
            return F.pad(logit, pad_shape, value=torch.finfo(float).min)

        if self.keep_action_space < self.num_categories:
            out_keep = pad_action(out_keep, self.num_categories - self.keep_action_space)

        elif self.keep_action_space > self.num_categories:
            out_category = pad_action(out_category, self.keep_action_space - self.num_categories)

        out_action = torch.where(rolls_left == 0, out_category.double(), out_keep.double())
        out_action = self.softmax(out_action)

        return out_action, torch.squeeze(out_value, axis=-1)


class A2CAgent(object):
    def __init__(self, num_dice=4, num_eyes=6, num_category=6, learning_rate=LEARNING_RATE):
        self.a2c_net = A2CNetwork(num_dice=num_dice, num_eyes=num_eyes, num_categories=num_category)
        self.optimizer = optim.Adam(self.a2c_net.parameters(), lr=learning_rate)
        self.steps_done = 0
        self.saved_actions = []
        self.rewards = []

    def load_a2c_net(self, load_file='a2c_model.dat'):
        self.a2c_net.load_state_dict(torch.load(load_file))

    def save_a2c_net(self, save_file='a2c_model.dat'):
        torch.save(self.a2c_net.state_dict(), save_file)

    def select_action(self, state):
        state = state_to_tensor(state).float()
        probs, state_value = self.a2c_net(state)

        c = Categorical(probs)
        action = c.sample()
        self.saved_actions.append((torch.log(probs[0][action]), state_value))
        return action

    def clean_rewards(self):
        self.rewards = []

    def clean_saved_actions(self):
        self.saved_actions = []

def write_summary(log_dir, num_episodes, lr):
    with open(log_dir+'/summary.txt', 'w') as f:
        f.write('num_episode=\t'+str(num_episodes)+'\n')
        f.write('lr=\t'+str(lr)+'\n')
    f.close()


def learn(num_episodes=500, learning_rate=LEARNING_RATE, save_file='a2c_model.dat'):
    global epoch_cnt
    num_dice = 4
    num_eyes = 6
    env = SimpleYahtzeeEnv2(num_dice, num_eyes, num_eyes, mode='comp')
    write_summary(log_dir, num_episodes, learning_rate)
    agent = A2CAgent(num_dice, num_eyes, num_eyes, learning_rate=LEARNING_RATE)
    agent.load_a2c_net('./a2c/20220418-145546/a2c_model.dat')
    for i_episode in tqdm(range(num_episodes)):
        # 환경과 상태 초기화
        state = env.reset()
        ep_reward = 0
        for t in count():
            # 행동 선택과 수행
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # next_state = torch.tensor(next_state, device=device)
            ep_reward += reward
            agent.rewards.append(reward)
            # 다음 상태로 이동
            state = next_state

            # (정책 네트워크에서) 최적화 한단계 수행
            if done:
                break
        # 목표 네트워크 업데이트, 모든 웨이트와 바이어스 복사
        returns = []
        policy_losses = []
        value_losses = []
        R = 0
        for r in agent.rewards[::-1]:
            # calculate the discounted value
            R = r + GAMMA * R
            returns.insert(0, R)
        eps = np.finfo(np.float).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for (log_prob, value), R in zip(agent.saved_actions, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).float()))

        # sum up all the values of policy_losses and value_losses
        agent.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        total_score = env.get_total_score()
        loss.backward()
        agent.optimizer.step()
        writer.add_scalar('train/score', total_score, i_episode)
        writer.add_scalar('train/loss', loss, i_episode)
        agent.clean_rewards()
        agent.clean_saved_actions()
        # perform backprop
    print("Training Complete")

    # for var_name in agent.optimizer.state_dict():
    #     print(var_name, "\t", agent.optimizer.state_dict()[var_name])

    agent.save_a2c_net(log_dir + '/a2c_model.dat')
    return agent


def play(num_episode):
    num_dice = 4
    num_eyes = 6
    env = SimpleYahtzeeEnv2(num_dice, num_eyes, num_eyes, 'human')
    agent = A2CAgent(num_dice, num_eyes, num_eyes, learning_rate=LEARNING_RATE)
    agent.load_a2c_net('./a2c/20220418-113543/a2c_model.dat')
    total_scores = []
    for i in range(num_episode):
        # start agent rotation
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)

            state, reward, done, info = env.step(action)
            # env.render()

            if done:
                env.show_result()
                total_scores.append(env.get_total_score())
                break

    # fig, ax = plt.subplots(1,1)
    # ax.plot(total_scores)
    # plt.show()
    avg = sum(total_scores)/len(total_scores)
    print(avg)
    print(total_scores)
    return avg


if __name__ == '__main__':
    learn(1000000)
    # play(1)