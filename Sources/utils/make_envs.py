
import gymnasium as gym

def make_env(args):
    env = gym.make(args.env.name)
    return env