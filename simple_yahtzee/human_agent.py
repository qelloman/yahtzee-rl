import sys, os
import time
from simple_yahtzee import SimpleYahtzeeEnv
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
            print("Available actions = {}".format(ava_actions))
            action = input("Enter category[1-#], q for quit: ")
            if action.lower() == 'q':
                return None
            try:
                action = int(action)
                if action not in ava_actions:
                    raise ValueError()
            except ValueError:
                print("Illegal value: '{}'".format(action))
            else:
                break

        return action


@click.command(help="Play human agent.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number in the board.")
def play(show_number):
    env = SimpleYahtzeeEnv(num_dice=4, num_eyes=6)
    agent = HumanAgent()
    episode = 1
    total_rewards = []
    while True:
        # os.system('clear')
        state = env.reset()
        done = False
        env.render()
        while not done:
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