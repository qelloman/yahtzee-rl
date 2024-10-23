import numpy as np
import torch


class Node(object):
    
    def __init__(self, prior):
        """
        Node in MCTS
        prior: The prior policy on the node, computed from policy network
        """
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_representation = None
        self.reward = 0
        self.expanded = False

    def value(self):
        """
        Compute expected value of a node
        """
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count
        

class MCTS():
    
    def __init__(self, config):
        self.config = config
        
    def run_mcts(self, config, root, network, min_max_stats):
        """
        Run the main loop of MCTS for config['num_simulations'] simulations

        root: the root node
        network: the network
        min_max_stats: the min max stats object
        """
        for i in range(config['num_simulations']):
            history = []
            node = root
            search_path = [node]

            # expand node until reaching the leaf node
            while node.expanded:
                action, node = self.select_child(config, node, min_max_stats)
                history.append(action)
                search_path.append(node)
            parent = search_path[-2]
            action = history[-1]
            
            # expand the leaf node
            value = self.expand_node(node, list(
                range(config['action_space_size'])), network, parent.hidden_representation, action)
            
            # perform backpropagation
            self.backpropagate(search_path, value,
                        config['discount'], min_max_stats)


    def select_action(self, config, node, test=False):
        """
        Select an action to take
        train mode: action selection is performed stochastically (softmax)
        test mode: action selection is performed deterministically (argmax)
        """
        visit_counts = [
            (child.visit_count, action) for action, child in node.children.items()
        ]
        if not test:
            t = config['visit_softmax_temperature_fn']
            action = self.softmax_sample(visit_counts, t)
        else:
            action = self.softmax_sample(visit_counts, 0)
        return action


    def select_child(self, config, node, min_max_stats):
        """
        Select a child at an already expanded node
        Selection is based on the UCB score
        """
        best_action, best_child = None, None
        ucb_compare = -np.inf
        for action,child in node.children.items():
            ucb = self.ucb_score(config, node, child, min_max_stats)
            if ucb > ucb_compare:
                ucb_compare = ucb
                best_action = action # action, int
                best_child = child # node object
        return best_action, best_child


    def ucb_score(self, config, parent, child, min_max_stats):
        """
        Compute UCB Score of a child given the parent statistics
        Appendix B of MuZero paper
        """
        pb_c = np.log((parent.visit_count + config['pb_c_base'] + 1)
                    / config['pb_c_base']) + config['pb_c_init']
        pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c*child.prior.detach().numpy()
        if child.visit_count > 0:
            value_score = min_max_stats.normalize(
                child.reward + config['discount']*child.value())
        else:
            value_score = 0
        return prior_score + value_score


    def expand_root(self, node, actions, network, current_state):
        """
        Expand the root node given the current state
        """
        # obtain the latent state, policy, and value of the root node 
        # by using a InitialModel
        observation = torch.tensor(current_state)
        transformed_value, reward, policy_logits, hidden_representation = network.initial_inference(observation)
        node.hidden_representation = hidden_representation
        node.reward = reward # always 0 for initial inference

        # extract softmax policy and set node.policy
        softmax_policy = torch.nn.functional.softmax(torch.squeeze(policy_logits))
        node.policy = softmax_policy

        # instantiate node's children with prior values, obtained from the predicted policy
        for action, prob in zip(actions, softmax_policy):
            child = Node(prob)
            node.children[action] = child

        # set node as expanded
        node.expanded = True
        
        return transformed_value


    def expand_node(self, node, actions, network, parent_state, parent_action):
        """
        Expand a leaf node given the parent state and action
        """
        # run recurrent inference at the leaf node
        transformed_value, reward, policy_logits, hidden_representation = network.recurrent_inference(parent_state, parent_action)
        node.hidden_representation = hidden_representation
        node.reward = reward

        # compute softmax policy and store it to node.policy
        softmax_policy = torch.nn.functional.softmax(torch.squeeze(policy_logits))
        node.policy = softmax_policy

        # instantiate node's children with prior values, obtained from the predicted softmax policy
        for action, prob in zip(actions,softmax_policy):
            child = Node(prob)
            node.children[action] = child

        # set node as expanded
        node.expanded = True
        
        return transformed_value


    def add_exploration_noise(self, config, node):
        """
        Add exploration noise by adding dirichlet noise to the prior over children
        This is governed by root_dirichlet_alpha and root_exploration_fraction
        """
        actions = list(node.children.keys())
        noise = np.random.dirichlet([config['root_dirichlet_alpha']]*len(actions))
        frac = config['root_exploration_fraction']
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1-frac) + n*frac


    def backpropagate(self, path, value, discount, min_max_stats):
        """
        Update a discounted total value and total visit count
        """
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value 
            min_max_stats.update(node.value())
            value = node.reward + discount * value


    def softmax_sample(self, visit_counts, temperature):
        """
        Sample an action
        """
        counts_arr = np.array([c[0] for c in visit_counts])
        if temperature == 0: # argmax
            action_idx = np.argmax(counts_arr)
        else: # softmax
            numerator = np.power(counts_arr,1/temperature)
            denominator = np.sum(numerator)
            dist = numerator / denominator
            action_idx = np.random.choice(np.arange(len(counts_arr)),p=dist)

        return action_idx
    
    
class MinMaxStats(object):
    """
    Store the min-max values of the environment to normalize the values
    Max value will be 1 and min value will be 0
    """

    def __init__(self, minimum, maximum):
        self.maximum = maximum
        self.minimum = minimum

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
