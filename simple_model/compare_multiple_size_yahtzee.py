
from tqdm import tqdm as _tqdm
from qlearning_agent import QLearningAgent
from simple_yahtzee import SimpleYahtzeeEnv
from base_agent import BaseAgent

tqdm = _tqdm

def learn(num_dice, num_dice_face, max_episode=50000, eps_start=1.0, eps_decay=0.99995, eps_min=0.3, alpha=0.4, gamma=1, save_file='state_action_value.dat'):
    env = SimpleYahtzeeEnv(num_dice, num_dice_face, 'learn')
    agent = QLearningAgent()
    save_file='state_action_values-dice{}face{}.dat'.format(int(num_dice), int(num_dice_face))
    eps = eps_start
    for i in tqdm(range(max_episode)):
        episode = i + 1

        # reset agent for new episode
        agent.episode_rate = episode / float(max_episode)

        state = env.reset()
        done = False
        while not done:
            ava_actions = env.available_actions()
            eps = max(eps * eps_decay, eps_min)

            action = agent.egreedy_policy(state, ava_actions, eps)
            # update (no rendering)
            nstate, reward, done, info = env.step(action)

            if done:
                env.show_result()
                # set terminal state value
                agent.set_state_action_value(state, action, reward)
            else:
                val = agent.ask_value(state, action)

                max_ava_actions = [i for i in range(env.num_dice_face)]
                target_val = reward + gamma * agent.ask_max_value(nstate, max_ava_actions)
                new_val = val + alpha * (target_val - val)
                agent.set_state_action_value(state, action, new_val)

            state = nstate

    # save states
    agent.save_model(save_file, max_episode, alpha)


def play_by_qagent(num_dice, num_dice_face, num_episode):
    load_file='state_action_values-dice{}face{}.dat'.format(int(num_dice), int(num_dice_face))
    env = SimpleYahtzeeEnv(num_dice, num_dice_face, 'computer_play')
    # env = SimpleYahtzeeEnv(num_dice, num_dice_face, 'computer_play', 'last')
    agent = QLearningAgent()  # prevent exploring
    agent.load_model(load_file)
    total_scores = []
    for i in range(num_episode):
        # start agent rotation
        state = env.reset()
        done = False
        while not done:
            ava_actions = env.available_actions()
            action = agent.greedy_action(state, ava_actions)

            state, reward, done, info = env.step(action)
            env.render()

            if done:
                env.show_result()
                total_scores.append(env.get_total_score())
                break

    return sum(total_scores)/len(total_scores)

def play_by_baseagent(num_dice, num_dice_face, num_episode):
    env = SimpleYahtzeeEnv(num_dice, num_dice_face, 'silent')
    # env = SimpleYahtzeeEnv(num_dice, num_dice_face, 'computer_play', 'last')
    agent = BaseAgent()
    total_scores = []

    for _ in range(num_episode):
        state = env.reset()
        done = False
        while not done:
            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions)
            state, reward, done, info = env.step(action)
        env.render()
        env.show_result()
        total_scores.append(env.get_total_score())

    return sum(total_scores)/len(total_scores)

if __name__ == '__main__':
    game_list = [(3, 2), (4, 2), (5, 2), (3, 3), (4, 3), (5, 3)]
    for game in game_list:
        num_dice, num_dice_face = game
        learn(num_dice, num_dice_face, 5000000)
        qnumber = play_by_qagent(num_dice, num_dice_face, 1000)
        basenumber = play_by_baseagent(num_dice, num_dice_face, 1000)
        print(num_dice, num_dice_face, basenumber, qnumber )