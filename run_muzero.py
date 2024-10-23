import math
import os
import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from envs.yahtzee_simple import YahtzeeSimpleEnv
from mcts.game import Game
from mcts.tree import MCTS, MinMaxStats, Node
from models.muzero import (
    DynamicNetwork,
    Networks,
    PolicyNetwork,
    RepresentationNetwork,
    RewardNetwork,
    ValueNetwork,
)


class ReplayBuffer(object):
    """
    Store training data acquired through self-play
    """
    def __init__(self, config):
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.buffer = deque(maxlen=self.buffer_size) # deque: list-like container with fast appends and pops on either end
        self.td_steps = config['td_steps']
        self.unroll_steps = config['num_unroll_steps']

    def save_game(self, game):
        """
        Save a game into replay buffer.
        Max number of games saved in the buffer is defined as self.buffer_size
        """
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        """
        Sample batch_size games, along with an associated start position in each game
        Make the targets for the batch to be used in training
        """
        games = [self.sample_game() for _ in range(self.batch_size)] # randomly sample batch_size games
        game_pos = [self.sample_position(g) for g in games] # randomly sample position from the game
        batch = []
        for (g, i) in zip(games, game_pos):
            # create training targets (output) and actions (input)
            targets, actions = g.make_target(
                i, self.unroll_steps, self.td_steps) # each target = (value, reward, policy)
            batch.append(
                (g.state_history[i], actions, targets))
        state_batch, actions_batch, targets_batch = zip(*batch) # unpack batch
        actions_batch = list(zip(*actions_batch)) # unpack action
        targets_init_batch, *targets_recurrent_batch = zip(*targets_batch) # unpack targets_batch, targets_init_batch: initial target, targets_recurrent_batch: subsequent targets
        # * operator is used for extended unpacking, meaning that any additional targets beyond the initial one are packed into targets_recurrent_batch.
        batch = (state_batch, targets_init_batch, targets_recurrent_batch,
                 actions_batch)

        return batch

    def sample_game(self):
        """
        Ramdonly sample a game from buffer
        """
        game = np.random.choice(self.buffer)
        return game

    def sample_position(self, game):
        """
        Randomply sample position from a game to start unrolling
        """
        sampled_index = np.random.randint(
            len(game.reward_history)-self.unroll_steps) # limit the sample in the space where we can unroll # of unroll_steps
        return sampled_index
    

def scale_gradient(tensor, scale):
    """
    Function to scale gradient as described in MuZero Appendix
    """
    return tensor * scale + tensor.detach() * (1. - scale)


def train_network(config, network, replay_buffer, optimizer, train_results):
    """
    Train Networks
    """
    for _ in range(config['train_per_epoch']):
        batch = replay_buffer.sample_batch()
        update_weights(config, network, optimizer, batch, train_results)


def update_weights(config, network, optimizer, batch, train_results):
    """
    Train networks by sampling games from repay buffer
    config: dictionary specifying parameter configurations
    network: network class to train
    optimizer: optimizer used to update the network_model weights
    batch: batch of experience
    train_results: class to store the train results
    """
    # for every game in sample batch, unroll and update network_model weights
    def loss():
        mse = torch.nn.MSELoss()

        loss = 0
        total_value_loss = 0
        total_reward_loss = 0
        total_policy_loss = 0
        (state_batch, targets_init_batch, targets_recurrent_batch,
         actions_batch) = batch

        state_batch = torch.tensor(state_batch)

        # get prediction from initial model (i.e. combination of dynamic, value, and policy networks)
        hidden_representation, initial_values, policy_logits = network.initial_model(state_batch)

        # create a value and policy target from batch data
        target_value_batch, _, target_policy_batch = zip(*targets_init_batch) # (value, reward, policy)
        target_value_batch = torch.tensor(target_value_batch).float()
        target_value_batch = network._scalar_to_support(target_value_batch) # transform into a multi-dimensional target

        # compute the error for the initial inference
        # reward error is always 0 for initial inference
        value_loss = F.cross_entropy(initial_values, target_value_batch)
        policy_loss = F.cross_entropy(policy_logits, torch.tensor(target_policy_batch))
        loss = 0.25 * value_loss + policy_loss

        total_value_loss = 0.25 * value_loss.item()
        total_policy_loss = policy_loss.item()

        # unroll batch with recurrent inference and accumulate loss
        for actions_batch, targets_batch in zip(actions_batch, targets_recurrent_batch):
            target_value_batch, target_reward_batch, target_policy_batch = zip(*targets_batch)

            # get prediction from recurrent_model (i.e. dynamic, reward, value, and policy networks)
            actions_batch_onehot = F.one_hot(torch.tensor(actions_batch), num_classes=network.action_size).float()
            state_with_action = torch.cat((hidden_representation, actions_batch_onehot), dim=1)
            hidden_representation, rewards, values, policy_logits = network.recurrent_model(state_with_action)

            # create a value, policy, and reward target from batch data
            target_value_batch = torch.tensor(target_value_batch).float()
            target_value_batch = network._scalar_to_support(target_value_batch)
            target_policy_batch = torch.tensor(target_policy_batch).float()
            target_reward_batch = torch.tensor(target_reward_batch).float()

            # compute the loss for recurrent_inference 
            value_loss = F.cross_entropy(values, target_value_batch)
            policy_loss = F.cross_entropy(policy_logits, target_policy_batch)
            reward_loss = mse(rewards, target_reward_batch)

            # accumulate loss
            loss_step = (0.25 * value_loss + reward_loss + policy_loss)
            total_value_loss += 0.25 * value_loss.item()
            total_policy_loss += policy_loss.item()
            total_reward_loss += reward_loss.item()
                        
            # gradient scaling
            gradient_loss_step = scale_gradient(loss_step,(1/config['num_unroll_steps']))
            loss += gradient_loss_step
            scale = 0.5
            hidden_representation = hidden_representation / scale
            
        # store loss result for plotting
        train_results.total_losses.append(loss.item())
        train_results.value_losses.append(total_value_loss)
        train_results.policy_losses.append(total_policy_loss)
        train_results.reward_losses.append(total_reward_loss)
        return loss

    optimizer.zero_grad()
    loss = loss()
    loss.backward() # Compute gradients of loss with respect to parameters
    optimizer.step() # Update parameters based on gradients
    network.train_steps += 1


class TrainResults(object):
    def __init__(self):
        self.value_losses = []
        self.reward_losses = []
        self.policy_losses = []
        self.total_losses = []

    def plot_total_loss(self):
        x = np.arange(len(self.total_losses))
        plt.figure()
        plt.plot(x, self.total_losses, label="Train Loss", color='k')
        plt.xlabel("Train Steps", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.show()
        # plt.savefig('./RL/ModelBasedML/figure/total_loss.png')

    def plot_individual_losses(self):
        x = np.arange(len(self.total_losses))
        plt.figure()
        plt.plot(x, self.value_losses, label="Value Loss", color='r')
        plt.plot(x, self.policy_losses, label="Policy Loss", color='b')
        plt.plot(x, self.reward_losses, label="Reward Loss", color='g')
        plt.xlabel("Train Episode", fontsize=15)
        plt.ylabel("Loss", fontsize=15)
        plt.legend()
        plt.show()
        # plt.savefig('./RL/ModelBasedML/figure/individual_loss.png')


class TestResults(object):

    def __init__(self):
        self.test_rewards = []

    def add_reward(self, reward):
        self.test_rewards.append(reward)

    def plot_rewards(self):
        x = np.arange(len(self.test_rewards))
        plt.subplots()
        plt.plot(x, self.test_rewards, label="Test Reward", color='orange')
        plt.xlabel("Test Episode", fontsize=15)
        plt.ylabel("Reward", fontsize=15)
        plt.show()
        # plt.savefig('./RL/ModelBasedML/figure/test_reward.png')

def self_play(env, config, replay_buffer, network):
    # create objects to store data for plotting
    test_rewards = TestResults()
    train_results = TrainResults()
    
    # create optimizer for training
    optimizer = torch.optim.Adam(network.parameters(), lr=config['lr_init'])
    
    # self-play and network training iterations
    for i in range(config['num_epochs']):  # Number of Steps of train/play alternations
        print(f"===Epoch Number {i}===")
        score = play_games(
            config, replay_buffer, network, env)
        print("Average traininig score:", score)
        train_network(config, network, replay_buffer, optimizer, train_results)
        print("Average test score:", test(config, network, env, test_rewards))

    # plot
    train_results.plot_individual_losses()
    train_results.plot_total_loss()
    test_rewards.plot_rewards()


def play_games(config, replay_buffer, network, env):
    """
    Play multiple games and store them in the replay buffer
    """
    returns = 0

    for _ in range(config['games_per_epoch']):
        game = play_game(config, network, env)
        replay_buffer.save_game(game)
        returns += sum(game.reward_history)

    return returns / config['games_per_epoch']


def play_game(config, network: Networks, env):
    """
    Plays one game
    """
    # Initialize environment
    start_state, _ = env.reset()
    
    game = Game(config['action_space_size'], config['discount'], start_state)        
    mcts = MCTS(config)
    
    # Play a game using MCTS until game will be done or max_moves will be reached
    while not game.done and len(game.action_history) < config['max_moves']:
        root = Node(0)
        
        # Create MinMaxStats Object to normalize values
        min_max_stats = MinMaxStats(config['min_value'], config['max_value'])
        
        # Expand the current root node
        curr_state = game.curr_state
        value = mcts.expand_root(root, list(range(config['action_space_size'])),
                            network, curr_state)
        mcts.backpropagate([root], value, config['discount'], min_max_stats)
        mcts.add_exploration_noise(config, root)

        # Run MCTS
        mcts.run_mcts(config, root, network, min_max_stats)

        # Select an action to take
        action = mcts.select_action(config, root)

        # Take an action and store tree search statistics
        game.take_action(action, env)
        game.store_search_statistics(root)
    print(f'Total reward for a train game: {sum(game.reward_history)}')
    return game


def test(config, network, env, test_rewards):
    """
    Test performance using trained networks
    """
    mcts = MCTS(config)
    returns = 0
    for _ in range(config['episodes_per_test']):
        # env.seed(1) # use for reproducibility of trajectories
        start_state, _ = env.reset()
        game = Game(config['action_space_size'], config['discount'], start_state)
        while not game.done and len(game.action_history) < config['max_moves']:
            min_max_stats = MinMaxStats(config['min_value'], config['max_value'])
            curr_state = game.curr_state
            root = Node(0)
            value = mcts.expand_root(root, list(range(config['action_space_size'])),
                                network, curr_state)
            # don't run mcts.add_exploration_noise for test
            mcts.backpropagate([root], value, config['discount'], min_max_stats)
            mcts.run_mcts(config, root, network, min_max_stats)
            action = mcts.select_action(config, root, test=True) # argmax action selection
            game.take_action(action, env)
        total_reward = sum(game.reward_history)
        print(f'Total reward for a test game: {total_reward}')
        test_rewards.add_reward(total_reward)
        returns += total_reward
    return returns / config['episodes_per_test']


config = {
          # Simulation and environment Config
          'action_space_size': 37, # number of action
          'state_shape': 45,
          'games_per_epoch': 20,
          'num_epochs': 25,
          'train_per_epoch': 30,
          'episodes_per_test': 10,
          'cartpole_stop_reward': 200,

          'visit_softmax_temperature_fn': 1,
          'max_moves': 18,
          'num_simulations': 50,
          'discount': 0.997,
          'min_value': 0,
          'max_value': 105, # (1 + ... + 6) * 5

          # Root prior exploration noise.
          'root_dirichlet_alpha': 0.1,
          'root_exploration_fraction': 0.25,

          # UCB parameters
          'pb_c_base': 19652,
          'pb_c_init': 1.25,

          # Model fitting config
          'embedding_size': 8,
          'hidden_neurons': 48,
          'buffer_size': 200,
          'batch_size': 512,
          'num_unroll_steps': 5,
          'td_steps': 10,
          'lr_init': 0.01,
          }

SEED = 0
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
value_support_size = math.ceil(math.sqrt(config['max_value'])) + 1


# Set seeds for reproducibility
set_seeds()

# Create networks
rep_net = RepresentationNetwork(input_size=config['state_shape'], hidden_neurons=config['hidden_neurons'], embedding_size=config['embedding_size']) # representation function
val_net = ValueNetwork(input_size=config['embedding_size'], hidden_neurons=config['hidden_neurons'], value_support_size=value_support_size) # prediction function
pol_net = PolicyNetwork(input_size=config['embedding_size'], hidden_neurons=config['hidden_neurons'], action_size=config['action_space_size']) # prediction function
dyn_net = DynamicNetwork(input_size=config['embedding_size']+config['action_space_size'], hidden_neurons=config['hidden_neurons'], embedding_size=config['embedding_size']) # dynamics function
rew_net = RewardNetwork(input_size=config['embedding_size']+config['action_space_size'], hidden_neurons=config['hidden_neurons']) # from dynamics function
network = Networks(rep_net, val_net, pol_net, dyn_net, rew_net, max_value=config['max_value'])

# Create environment
env = YahtzeeSimpleEnv()

# Create buffer to store games
replay_buffer = ReplayBuffer(config)
self_play(env, config, replay_buffer, network)
