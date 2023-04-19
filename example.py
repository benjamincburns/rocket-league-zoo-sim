import rlgymnasium_sim
import time

env = rlgymnasium_sim.make()

while True:
    obs = env.reset()
    terminated = truncated = False
    steps = 0
    ep_reward = 0
    t0 = time.time()
    while not (terminated or truncated):
        actions = env.action_space.sample()  # agent.act(obs) | Your agent should go here
        new_obs, reward, terminated, truncated, state = env.step(actions)
        ep_reward += reward
        obs = new_obs
        steps += 1

    length = time.time() - t0
    print("Step time: {:1.5f} ({:.2f} ts/s) | Episode time: {:.2f} ({:1.5f} ep/s) | Episode Reward: {:.2f}".format(length / steps, steps / length, length, 1/length, ep_reward))
