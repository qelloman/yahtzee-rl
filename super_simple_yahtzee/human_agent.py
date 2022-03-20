import sys, os
import time
from simple_yahtzee import SuperSimpleYahtzeeEnv
import click


def speak(msg, action=None):
    click.echo("> ", nl=False)
    time.sleep(0.2)

    for char in msg:
        click.echo(char, nl=False)
        time.sleep(0.02)

    time.sleep(0.1)

    if action == "confirm":
        return click.confirm("")

    if action == "confirm-abort":
        return click.confirm("", abort=True)

    if action == "prompt":
        return click.prompt("", prompt_suffix=" ")

    click.echo("")


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


def play(num_dice=4, num_eyes=6):
    env = SuperSimpleYahtzeeEnv(num_dice=num_dice, num_eyes=num_eyes)
    agent = HumanAgent()
    episode = 1
    total_rewards = []
    while True:
        # os.system('clear')
        state = env.reset()
        done = False
        env.render()
        while True:
            ava_actions = env.available_actions()
            action = agent.act(ava_actions)
            if action is None:
                print(total_rewards)
                print(sum(total_rewards) / episode)
                sys.exit()

            os.system('clear')
            state, reward, done, info = env.step(action)

            print('')
            env.render()
            if done:
                env.show_result()
                total_rewards.append(env.get_total_score())
                break

        episode += 1


if __name__ == '__main__':
    play()