import random

from simple_yahtzee import SimpleYahtzeeEnv


class BaseAgent(object):
    def act(self, state, ava_actions):
        return random.choice(ava_actions)


def play(max_episode=100):
    env = SimpleYahtzeeEnv(3, 3, 'silent')
    print(env.available_actions())
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

    print(sum(total_scores)/len(total_scores))

if __name__ == '__main__':
    play()
