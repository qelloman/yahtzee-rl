import random

from super_simple_yahtzee import SuperSimpleYahtzeeEnv


class BaseAgent(object):
    def act(self, state, ava_actions):
        return random.choice(ava_actions)


def play(max_episode=100, num_dice=4, num_eyes=6):
    env = SuperSimpleYahtzeeEnv(num_dice, num_eyes, 'silent')
    agent = BaseAgent()
    total_scores = []

    for _ in range(max_episode):
        state = env.reset()
        done = False
        while not done:
            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions)
            state, reward, done, info = env.step(action)
        env.render()
        env.show_result()
        total_scores.append(env.get_total_score())

    print(total_scores)
    print(sum(total_scores)/len(total_scores))

if __name__ == '__main__':
    play()
