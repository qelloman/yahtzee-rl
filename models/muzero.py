import math

import numpy as np
import torch
import torch.nn as nn


class RepresentationNetwork(nn.Module):
    """
    Input: raw state of the current root
    Output: latent state of the current root
    """
    def __init__(self, input_size, hidden_neurons, embedding_size):
        super(RepresentationNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, embedding_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)
    
    
class ValueNetwork(nn.Module):
    """
    Input: latent state
    Output: expected value at the input latent state
    """
    def __init__(self, input_size, hidden_neurons, value_support_size):
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, value_support_size)
        )

    def forward(self, x):
        return self.layers(x)
    
    
class PolicyNetwork(nn.Module):
    """
    Input: latent state
    Output: policy at the input latent state
    """
    def __init__(self, input_size, hidden_neurons, action_size):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, action_size)
        )

    def forward(self, x):
        return self.layers(x)
    
    
class DynamicNetwork(nn.Module):
    """
    Input: latent state & action to take
    Output: next latent state
    """
    def __init__(self, input_size, hidden_neurons, embedding_size):
        super(DynamicNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, embedding_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)
    
    
class RewardNetwork(nn.Module):
    """
    Input: latent state & action to take
    Output: expected immediate reward
    """
    def __init__(self, input_size, hidden_neurons):
        super(RewardNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 1)
        )

    def forward(self, x):
        return self.layers(x)
    
    
class InitialModel(nn.Module):
    """
    Combine Representation, Value, and Policy networks
    """
    def __init__(self, representation_network, value_network, policy_network):
        super(InitialModel, self).__init__()
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network

    def forward(self, state):
        hidden_representation = self.representation_network(state)
        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)
        return hidden_representation, value, policy_logits


class RecurrentModel(nn.Module):
    """
    Combine Dynamic, Reward, Value, and Policy network
    """
    def __init__(self, dynamic_network, reward_network, value_network, policy_network):
        super(RecurrentModel, self).__init__()
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network
        self.value_network = value_network
        self.policy_network = policy_network

    def forward(self, state_with_action):
        hidden_representation = self.dynamic_network(state_with_action)
        reward = self.reward_network(state_with_action)
        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)
        return hidden_representation, reward, value, policy_logits
    
    
class Networks(nn.Module):
    """
    Create both InitialModel and RecurrentModel class objects 
    and helper functions to run MCTS and train models
    """
    def __init__(self, representation_network, value_network, policy_network, dynamic_network, reward_network, max_value):
        super().__init__()
        self.train_steps = 0
        self.action_size = 2
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network
        self.initial_model = InitialModel(self.representation_network, self.value_network, self.policy_network)
        self.recurrent_model = RecurrentModel(self.dynamic_network, self.reward_network, self.value_network,
                                              self.policy_network)
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1

    def initial_inference(self, state):
        hidden_representation, value, policy_logits = self.initial_model(state)
        assert isinstance(self._value_transform(value), float)
        return self._value_transform(value), 0, policy_logits, hidden_representation

    def recurrent_inference(self, hidden_state, action):
        hidden_state_with_action = self._hidden_state_with_action(hidden_state, action)
        hidden_representation, reward, value, policy_logits = self.recurrent_model(hidden_state_with_action)
        return self._value_transform(value), self._reward_transform(reward), policy_logits, hidden_representation

    def _value_transform(self, value_support):
        """
        Apply invertable transformation to get a numpy scalar value
        """
        epsilon = 0.001
        value = torch.nn.functional.softmax(value_support)
        value = np.dot(value.detach().numpy(), range(self.value_support_size))
        value = np.sign(value) * (
                ((np.sqrt(1 + 4 * epsilon
                 * (np.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1
        )
        return value

    def _reward_transform(self, reward):
        """
        Transform reward into a numpy scalar value
        """
        return reward.detach().cpu().numpy()  # Assuming reward is a PyTorch tensor

    def _hidden_state_with_action(self, hidden_state, action):
        """
        Merge hidden state and one hot encoded action
        """
        hidden_state_with_action = torch.concat(
            (hidden_state, torch.tensor(self._action_to_one_hot(action, self.action_size))[0]), axis=0)
        return hidden_state_with_action
    
    def _action_to_one_hot(self, action, action_space_size):
        """
        Compute one hot of action to be combined with state representation
        """
        return np.array([1 if i == action else 0 for i in range(action_space_size)]).reshape(1, -1)
    
    def _scalar_to_support(self, target_value):
        """
        Transform value into a multi-dimensional target value to train a network
        """
        batch = target_value.size(0)
        targets = torch.zeros((batch, self.value_support_size))
        target_value = torch.sign(target_value) * \
            (torch.sqrt(torch.abs(target_value) + 1)
            - 1 + 0.001 * target_value)
        target_value = torch.clamp(target_value, 0, self.value_support_size)
        floor = torch.floor(target_value)
        rest = target_value - floor
        targets[torch.arange(batch, dtype=torch.long), floor.long()] = 1 - rest
        indexes = floor.long() + 1
        mask = indexes < self.value_support_size
        batch_mask = torch.arange(batch)[mask]
        rest_mask = rest[mask]
        index_mask = indexes[mask]
        targets[batch_mask, index_mask] = rest_mask
        return targets

