
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
log_dir = './dqn/' + now_str
writer = SummaryWriter(log_dir=log_dir)

from tqdm import tqdm
from super_simple_yahtzee import SuperSimpleYahtzeeEnv


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
num_dice = 4
num_dice_face = 6
epoch_cnt = 0
LEARNING_RATE = 0.000005


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


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, num_dice_face=3):
        super(DQN, self).__init__()
        # num_dice_face => score_category => binary
        # num_dice => hexary
        self.fc1 = nn.Linear(num_dice_face * 2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_dice_face)
    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됩니다.
    # ([[left0exp,right0exp]...]) 를 반환합니다.

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        out = self.fc3(x)
        return out


class DQNAgent(object):
    def __init__(self, num_dice_face=6, learning_rate=LEARNING_RATE):
        self.policy_net = DQN(num_dice_face=num_dice_face)
        self.target_net = DQN(num_dice_face=num_dice_face)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def load_policy_net(self, load_file='dqn_model.dat'):
        self.policy_net.load_state_dict(torch.load(load_file))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_policy_net(self, save_file='dqn_model.dat'):
        torch.save(self.policy_net.state_dict(), save_file)

    def egreedy_action(self, state, ava_actions):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            return self.greedy_action(state, ava_actions)
        else:
            return self.random_action(state, ava_actions)

    def greedy_action(self, state, ava_actions):
        with torch.no_grad():
            self.policy_net.eval()
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            state_tensor = state_to_tensor(state)
            actions = self.policy_net(state_tensor)
            indices = torch.argmax(actions[0][ava_actions]).tolist()
            if type(indices) != list:
                aidx = indices
            else:
                aidx = random.choice(indices)
            return torch.tensor(ava_actions[aidx], device=device, dtype=torch.long).view(1, 1)

    def random_action(self, state, ava_actions):
        return torch.tensor([[random.choice(ava_actions)]], device=device, dtype=torch.long).view(1, 1)

    def optimize_model(self):
        global epoch_cnt, writer
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
        # 전환합니다.
        batch = Transition(*zip(*transitions))

        # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
        # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
        # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
        self.policy_net.train()
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 모든 다음 상태를 위한 V(s_{t+1}) 계산
        # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
        # max(1)[0]으로 최고의 보상을 선택하십시오.
        # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Huber 손실 계산
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        writer.add_scalar("Loss/train", loss, epoch_cnt)
        epoch_cnt = epoch_cnt + 1
        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

def write_summary(log_dir, num_episodes, lr):
    with open(log_dir+'/summary.txt', 'w') as f:
        f.write('num_episode=\t'+str(num_episodes)+'\n')
        f.write('lr=\t'+str(lr)+'\n')
    f.close()

def learn(num_episodes=500, learning_rate=LEARNING_RATE, save_file='dqn_model.dat'):
    global epoch_cnt
    num_dice = 4
    num_dice_face = 6
    env = SuperSimpleYahtzeeEnv(num_dice, num_dice_face, 'computer', 'every')
    agent = DQNAgent(num_dice_face)
    write_summary(log_dir, num_episodes, learning_rate)
    agent.load_policy_net('./dqn/20220309-110258/dqn_model.dat')
    for i_episode in tqdm(range(num_episodes)):
        # 환경과 상태 초기화
        state = env.reset()
        for t in count():
            # 행동 선택과 수행
            ava_actions = env.available_actions()
            action = agent.egreedy_action(state, ava_actions)
            next_state, reward, done, _ = env.step(action.item())
            # next_state = torch.tensor(next_state, device=device)
            reward = torch.tensor([reward], device=device)

            # 메모리에 변이 저장
            agent.memory.push(state_to_tensor(state), action, state_to_tensor(next_state), reward)

            # 다음 상태로 이동
            state = next_state

            # (정책 네트워크에서) 최적화 한단계 수행
            agent.optimize_model()
            if done:
                total_score = env.get_total_score()
                writer.add_scalar('train/score', total_score, epoch_cnt)
                break
        # 목표 네트워크 업데이트, 모든 웨이트와 바이어스 복사

        if i_episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print("Training Complete")

    show_param_stat(agent.policy_net)

    # for var_name in agent.optimizer.state_dict():
    #     print(var_name, "\t", agent.optimizer.state_dict()[var_name])
    agent.save_policy_net(save_file=log_dir+'/dqn_model.dat')
    return agent


def play(num_episode):
    env = SuperSimpleYahtzeeEnv(num_dice, num_dice_face, 'comp')
    agent = DQNAgent(num_dice_face)  # prevent exploring
    agent.load_policy_net('./dqn/20220309-110258/dqn_model.dat')
    total_scores = []
    for i in range(num_episode):
        # start agent rotation
        state = env.reset()
        done = False
        while not done:
            ava_actions = env.available_actions()
            action = agent.greedy_action(state, ava_actions)

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
    # agent = learn(50000)
    play(100)