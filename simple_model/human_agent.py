import sys, os
import click

from simple_yahtzee import YahtzeeEnv

class HumanAgent(object):
    def act(self, ava_actions):
        while True:
            ava_categories = [i+1 for i in ava_actions]
            print("Available categories = {}".format(ava_categories))
            combo_num = input("Enter category[1-#], q for quit: ")
            if combo_num.lower() == 'q':
                return None
            try:
                action = int(combo_num) - 1
                if action not in ava_actions:
                    raise ValueError()
            except ValueError:
                print("Illegal value: '{}'".format(combo_num))
            else:
                break

        return action


@click.command(help="Play human agent.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number in the board.")
def play(show_number):
    env = YahtzeeEnv()
    print(env.available_actions())
    agent = HumanAgent()
    episode = 0
    while True:
        # os.system('clear')
        state = env.reset()
        done = False
        env.render()
        while not done:
            ava_actions = env.available_actions()
            action = agent.act(ava_actions)
            if action is None:
                sys.exit()

            os.system('clear')
            state, reward, done, info = env.step(action)

            print('')
            env.render()
            if done:
                env.show_result()
                break

        episode += 1


if __name__ == '__main__':
    play()