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
from simple_yahtzee import SuperSimpleYahtzeeEnv
from torch.distributions import Categorical
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

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
LEARNING_RATE = 0.002


def state_to_tensor(state):
    category_status, dice_count = state
    category_status = list(category_status)
    dice_count = list(dice_count)
    tensor_out = torch.tensor(category_status + dice_count, dtype=torch.float)
    tensor_out = tensor_out.view(-1, tensor_out.size(0))
    return tensor_out


def show_param_stat(model):
    for name, param in model.named_parameters():
        print("{} : min: {}, max: {}, median: {}, var: {}".format(name, param.min(), param.max(), param.median(), param.var()))


class A2C(nn.Module):
    def __init__(self, num_dice_face=6):
        super(A2C, self).__init__()
        self.fc1 = nn.Linear(num_dice_face * 2, 128)

        self.action_fc = nn.Linear(128, num_dice_face)
        self.value_fc = nn.Linear(128, 1)
        self.float()
    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됩니다.
    # ([[left0exp,right0exp]...]) 를 반환합니다.

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_fc(x), dim=-1)
        state_value = self.value_fc(x)
        return action_prob, state_value


class A2CAgent(object):
    def __init__(self, num_dice_face=6, learning_rate=LEARNING_RATE):
        self.a2c_net = A2C(num_dice_face=num_dice_face)
        self.optimizer = optim.Adam(self.a2c_net.parameters(), lr=learning_rate)
        self.steps_done = 0
        self.saved_actions = []
        self.rewards = []

    def load_a2c_net(self, load_file='a2c_model.dat'):
        self.a2c_net.load_state_dict(torch.load(load_file))

    def save_a2c_net(self, save_file='a2c_model.dat'):
        torch.save(self.a2c_net.state_dict(), save_file)

    def select_action(self, state, ava_actions):
        state = state_to_tensor(state).float()
        probs, state_value = self.a2c_net(state)
        probs_norm = probs.reshape(-1)
        probs_norm = probs_norm[ava_actions]
        probs_norm = probs_norm / sum(probs_norm)
        try:
            m = Categorical(probs)
        except ValueError:
            print(probs)
        try:
            m_norm = Categorical(probs_norm)
        except ValueError:
            print(probs_norm)
        ava_action_idx = m_norm.sample().item()
        action = ava_actions[ava_action_idx]
        self.saved_actions.append(SavedAction(m.log_prob(torch.tensor(action)), state_value))
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
    num_dice_face = 6
    env = SuperSimpleYahtzeeEnv(num_dice, num_dice_face, 'computer', 'every')
    write_summary(log_dir, num_episodes, learning_rate)
    agent = A2CAgent(num_dice_face)
    # agent.load_policy_net('a2c_model.dat')
    for i_episode in tqdm(range(num_episodes)):
        # 환경과 상태 초기화
        state = env.reset()
        ep_reward = 0
        for t in count():
            # 행동 선택과 수행
            ava_actions = env.available_actions()
            action = agent.select_action(state, ava_actions)
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
        eps = np.finfo(np.double).eps.item()
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

    agent.save_a2c_net(log_dir+'/a2c_model.dat')
    return agent


def play(num_episode):
    env = SuperSimpleYahtzeeEnv(num_dice, num_dice_face, 'comp')
    agent = A2CAgent(num_dice_face)  # prevent exploring
    agent.load_a2c_net('./a2c/20220309-125350/a2c_model.dat')
    total_scores = []
    for i in range(num_episode):
        # start agent rotation
        state = env.reset()
        done = False
        while not done:
            ava_actions = env.available_actions()
            action = agent.select_action(state, ava_actions)

            state, reward, done, info = env.step(int(action))
            env.render()

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
    # agent = learn(100000)
    play(100)