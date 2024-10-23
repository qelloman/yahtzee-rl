import numpy as np


class Game:
    """
    A single episode of interaction with the environment.
    """
    def __init__(self, action_space_size, discount, curr_state):

        self.action_space_size = action_space_size
        self.curr_state = curr_state
        self.done = False
        self.discount = discount
        self.priorities = None

        self.state_history = [self.curr_state]
        self.action_history = []
        self.reward_history = []

        self.root_values = []
        self.child_visits = []

    def store_search_statistics(self, root):
        """
        Stores the search statistics for the current root node
        
        root: Node object including the infomration of the current root node
        """
        # Stores the normalized root node child visits (i.e. policy target)
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append(np.array([
            root.children[a].visit_count
            / sum_visits if a in root.children else 0
            for a in range(self.action_space_size)
        ]))
        
        # Stores the root node value, computed from the MCTS (i.e. vlaue target)
        self.root_values.append(root.value())

    def take_action(self, action, env):
        """
        Take an action and store the action, reward, and new state into history
        """
        observation, reward, terminated, truncated, _ = env.step(action)
        self.curr_state = observation
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.done = terminated | truncated
        if not self.done:
            self.state_history.append(self.curr_state)

    def make_target(self, state_index, num_unroll_steps, td_steps):
        """
        Makes the target data for training

        state_index: the start state
        num_unroll_steps: how many times to unroll from the current state
                          each unroll forms a new target
        td_steps: the number of td steps used in bootstrapping the value function
        """
        targets = [] # target = (value, reward, policy)
        actions = []

        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps

            # target value of the current node is the sum of 1) discounted rewards up to bootstrap index + 2) discounted value at bootstrap index            
            
            # compute 2)
            # assign value=0 if bootstrap_index is after the end of episode
            # otherwise, assign discounted value at bootstrap_index state
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index][0] * (self.discount**td_steps)
            else:
                value = 0
            
            # compute 1)  
            # add discounted reward values earned between current_index and bootstrap_index
            for i, reward in enumerate(self.reward_history[current_index:bootstrap_index]):
                value += reward * (self.discount**i)

            # if current_index is after the end of episode, assign 0 as last_reward
            # otherwise, assign the reward from last step as last_reward, which will be used as reward target
            if current_index > 0 and current_index <= len(self.reward_history):
                last_reward = self.reward_history[current_index-1]
            else:
                last_reward = 0
                
            if current_index < len(self.root_values): # current_index is within the episode, 
                targets.append((value, last_reward,
                                self.child_visits[current_index]))
                actions.append(self.action_history[current_index])
            else: # current_index is after the end of episode
                # State which pasts the end of the game are treated as an absorbing state.
                num_actions = self.action_space_size
                targets.append(
                    (0, last_reward, np.array([1.0 / num_actions for _ in range(num_actions)]))) # assign value 0 and uniform policy
                actions.append(np.random.choice(num_actions)) # assign a random action
        return targets, actions