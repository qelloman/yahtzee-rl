

import re
import json


def load_write_model(filename):
    state_action_values = {}
    with open(filename, 'rb') as f:
        info = json.loads(f.readline().decode('ascii'))
        outfile = re.split('\.', filename)[0] + '.csv'
        with open(outfile, 'wt') as of:
            # read model info
            for line in f:
                elms = line.decode('ascii').split('\t')
                state = eval(elms[0])
                action = eval(elms[1])
                value = eval(elms[2])
                of.write('{};{};{};{}\n'.format(state[1], state[0], action, value))


def load_and_print(filename='state_action_value.dat'):
    with open(filename, 'rb') as f:
        # read model info
        info = json.loads(f.readline().decode('ascii'))
        for line in f:
            print(line)


if __name__ == '__main__':
    load_write_model('state_action_values-dice3face2.dat')
    load_write_model('state_action_values-dice4face2.dat')
