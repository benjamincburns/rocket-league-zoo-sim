from pettingzoo.test import parallel_api_test
from rlzoo_sim.env import RocketSimEnv

env = RocketSimEnv()
parallel_api_test(env, num_cycles=1000)
