import gymnasium as gym
from gymnasium.envs.registration import register

from envs.yahtzee_oneshot import YahtzeeOneShotEnv

register(
    id='YahtzeeOneShot-v0',
    entry_point='envs.yahtzee_oneshot:YahtzeeOneShotEnv',
)

register(
    id='YahtzeeSimple-v0',
    entry_point='envs.yahtzee_simple:YahtzeeSimpleEnv',
)

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def make_sync_vector_env(env_id, num_envs, seed):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed + i) for i in range(num_envs)]
    )
    return envs
