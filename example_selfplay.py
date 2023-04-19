import rlgymnasium_sim
import time

env = rlgymnasium_sim.make(spawn_opponents=True)

while True:
    obs = env.reset()
    obs_1 = obs[0]
    obs_2 = obs[1]
    terminated = truncated = False
    steps = 0
    ep_reward = 0
    t0 = time.time()
    while not (terminated or truncated):
        actions_1 = env.action_space.sample()
        actions_2 = env.action_space.sample()
        actions = [actions_1, actions_2]
        new_obs, reward, terminated, truncated, state = env.step(actions)
        ep_reward += reward[0]
        obs_1 = new_obs[0]
        obs_2 = new_obs[1]
        steps += 1

    length = time.time() - t0
    print("Step time: {:1.5f} ({:.2f} ts/s) | Episode time: {:.2f} ({:1.5f} ep/s) | Episode Reward: {:.2f}".format(length / steps, steps / length, length, 1/length, ep_reward))
