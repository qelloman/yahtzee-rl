
import gymnasium as gym

from algorithms.ppo import PPO
from util.args import ppo_parse_args
from util.env import make_sync_vector_env


def run_ppo(args):
    
    envs = make_sync_vector_env(args.env_id, args.num_envs, args.seed)
    eval_env = gym.make(args.env_id)

    ppo = PPO(envs, eval_env, args)
    ppo.train()
    ppo.close()

if __name__ == "__main__":
    args = ppo_parse_args()
    run_ppo(args)

