#!/usr/bin/env python
# coding: utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import gym
import deeprl_hw1.lake_envs as lake_env
import deeprl_hw1.rl as rl
import time

import numpy
import matplotlib.pylab as plt

def plot_color_image(value_func, map):
    m = []
    width = 0
    if map == 'Deterministic-4x4-FrozenLake-v0':
        width = 4
    if map == 'Deterministic-8x8-FrozenLake-v0':
        width = 8

    for i in range(width):
        sub_m = []
        for j in range(width):
            sub_m.append(value_func[i * width + j])
        m.append(sub_m)

    matrix = numpy.matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.show()

def run_policy(env, policy, gamma):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    discount = 1
    while True:
        state, reward, is_terminal, debug_info = env.step(policy[state])
        env.render()

        total_reward += reward * discount
        discount *= gamma
        num_steps += 1

        if is_terminal:
            break

        time.sleep(1)

    return total_reward, num_steps

def Part1():
    print("Problem 2, Part I")

    action_names = {lake_env.LEFT: 'L', lake_env.RIGHT: 'R', lake_env.DOWN: 'D', lake_env.UP: 'U'}
    gamma = 0.9

    env = gym.make('Deterministic-8x8-FrozenLake-v0')

    time_before = int(round(time.time() * 1000))
    policy, value_func, policy_improvement_iteration, value_iteration = rl.policy_iteration_sync(env, gamma)
    time_after = int(round(time.time() * 1000))
    time_excute = time_after - time_before
    print(time_excute)
    print("policy_improvement_iteration: ", policy_improvement_iteration)
    print(value_iteration)
    print(value_func.tolist())
    rl.print_policy(policy, action_names)
    total_reward, num_steps = run_policy(env, policy, gamma)
    print("total_reward: ", total_reward)

    time_before = int(round(time.time() * 1000))
    value_func, iteration = rl.value_iteration_sync(env, gamma)
    time_after = int(round(time.time() * 1000))
    time_excute = time_after - time_before
    print(time_excute)
    print(value_func.tolist())
    print("value iteration: ", iteration)
    policy = rl.value_function_to_policy(env, gamma, value_func)
    rl.print_policy(policy, action_names)
    total_reward, num_steps = run_policy(env, policy, gamma)
    print("total_reward: ", total_reward)

    env = gym.make('Stochastic-8x8-FrozenLake-v0')

    time_before = int(round(time.time() * 1000))
    policy, value_func, policy_improvement_iteration, value_iteration = rl.policy_iteration_sync(env, gamma)
    time_after = int(round(time.time() * 1000))
    time_excute = time_after - time_before
    print(time_excute)
    print("policy_improvement_iteration: ", policy_improvement_iteration)
    print(value_iteration)
    print(value_func.tolist())
    rl.print_policy(policy, action_names)
    total_reward, num_steps = run_policy(env, policy, gamma)
    print("total_reward: ", total_reward)

    time_before = int(round(time.time() * 1000))
    value_func, iteration = rl.value_iteration_sync(env, gamma)
    time_after = int(round(time.time() * 1000))
    time_excute = time_after - time_before
    print(time_excute)
    print(value_func.tolist())
    print("value iteration: ", iteration)
    policy = rl.value_function_to_policy(env, gamma, value_func)
    rl.print_policy(policy, action_names)
    total_reward, num_steps = run_policy(env, policy, gamma)
    print("total_reward: ", total_reward)


    # Problem 2, Part I, h)
    env = gym.make('Deterministic-8x8-FrozenLake-v0')

    time_before = int(round(time.time() * 1000))
    policy, value_func, policy_improvement_iteration, value_iteration = rl.policy_iteration_async_ordered(env, gamma)
    time_after = int(round(time.time() * 1000))
    time_excute = time_after - time_before
    print(time_excute)
    print("policy_improvement_steps: ", policy_improvement_iteration)
    print("value_iteration: ", value_iteration)
    print(value_func.tolist())
    rl.print_policy(policy, action_names)
    run_policy(env, policy, gamma)

    time_before = int(round(time.time() * 1000))
    policy, value_func, policy_improvement_iteration, value_iteration = rl.policy_iteration_async_randperm(env, gamma)
    time_after = int(round(time.time() * 1000))
    time_excute = time_after - time_before
    print(time_excute)
    print("policy_improvement_steps: ", policy_improvement_iteration)
    print("value_iteration: ", value_iteration)

    # Problem 2, Part I, i)
    env = gym.make('Deterministic-8x8-FrozenLake-v0')

    time_before = int(round(time.time() * 1000))
    value_func, iteration = rl.value_iteration_async_ordered(env, gamma)
    time_after = int(round(time.time() * 1000))
    time_excute = time_after - time_before
    print(time_excute)
    print("value iteration: ", iteration)

    time_before = int(round(time.time() * 1000))
    value_func, iteration = rl.value_iteration_async_randperm(env, gamma)
    time_after = int(round(time.time() * 1000))
    time_excute = time_after - time_before
    print(time_excute)
    print("value iteration: ", iteration)

def Part2():
    print("Problem 2, Part II")

    action_names = {lake_env.LEFT: 'L', lake_env.RIGHT: 'R', lake_env.DOWN: 'D', lake_env.UP: 'U'}
    gamma = 0.9
    env = gym.make('Stochastic-4x4-FrozenLake-v0')

    value_func, iteration = rl.value_iteration_sync(env, gamma)
    print(value_func.tolist())
    policy = rl.value_function_to_policy(env, gamma, value_func)
    rl.print_policy(policy, action_names)
    avg_reward = 0
    for episode in range(100):
        print(episode)
        total_reward, num_steps = run_policy(env, policy, gamma)
        avg_reward += total_reward
        print("reward: ", avg_reward)
    print("avg_reward: ", avg_reward / 100)

    env = gym.make('Stochastic-8x8-FrozenLake-v0')

    value_func, iteration = rl.value_iteration_sync(env, gamma)
    print(value_func.tolist())
    policy = rl.value_function_to_policy(env, gamma, value_func)
    rl.print_policy(policy, action_names)
    avg_reward = 0
    for episode in range(100):
        print(episode)
        total_reward, num_steps = run_policy(env, policy, gamma)
        avg_reward += total_reward
        print("reward: ", avg_reward)
    print("avg_reward: ", avg_reward / 100)

def Part3():
    print("Problem 2, Part III")
    action_names = {lake_env.LEFT: 'L', lake_env.RIGHT: 'R', lake_env.DOWN: 'D', lake_env.UP: 'U'}
    gamma = 0.9
    env = gym.make('Stochastic-8x8-FrozenLake-v0')
    time_before = int(round(time.time() * 1000))
    value_func, iteration = rl.value_iteration_async_custom(env, gamma)
    time_after = int(round(time.time() * 1000))
    time_excute = time_after - time_before
    print(time_excute)
    print(value_func.tolist())
    print(iteration)

def main():
    # value_func = [0.0002271757988339024, 0.00031912911959736817, 0.0005377639742655986, 0.0008516442185070242, 0.0011604755876659125, 0.0009021220973609936, 0.0006336913018907875, 0.0004773216780386816, 0.0001954476916394231, 0.00027479707763635835, 0.0004940858723062496, 0.0009275861051706311, 0.0018636376729975633, 0.0009893073293105754, 0.000612981475313936, 0.00048668924611896137, 0.0, 0.0, 0.0, 0.0, 0.004161245311285345, 0.0, 0.0003224751706648394, 0.0005314184780298636, 0.09047908825202661, 0.05997398297006182, 0.03612993668939459, 0.02117128508037401, 0.012075047924061597, 0.005869376567353005, 0.0, 0.0008164768869194285, 0.1519967026493321, 0.07477348806730404, 0.03973415013114384, 0.02266788074055852, 0.013370989597047083, 0.007489540633781754, 0.0026641134147989635, 0.0014496867070404968, 0.26597980908126106, 0.0, 0.01839673814543576, 0.01510046314060824, 0.010128951744971491, 0.005886260779441204, 0.0, 0.0005931399842652265, 0.47045317794288966, 0.0, 0.0076382498603868275, 0.007911663712117448, 0.0, 0.002238766792956485, 0.0, 0.0, 0.8332289522667574, 0.0, 0.0, 0.004095860747743281, 0.002446580314998586, 0.001946856277394342, 0.0010530634521998823, 0.0006793726832478852]
    # map = 'Deterministic-8x8-FrozenLake-v0'
    # plot_color_image(value_func, map)

    Part3()


if __name__ == '__main__':
    main()
