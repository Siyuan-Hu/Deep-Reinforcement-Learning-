# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import random
import math


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    

    policy = np.zeros(env.nS, dtype='int')

    for state in range(env.nS):
      optimal_action = 0
      optimal_value = 0
      for action in range(env.nA):
        transition_table_row = env.P[state][action]
        value = 0
        for prob, nextstate, reward, is_terminal in transition_table_row:
          value += prob * (reward + gamma * value_function[nextstate])
        if value >= optimal_value or action == 0:
          optimal_action = action
          optimal_value = value
      policy[state] = optimal_action

    return policy


def evaluate_policy_sync(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    v = np.zeros(env.nS)
    count = 0

    while True:
      new_v = np.zeros(env.nS)
      end_loop = True
      count += 1

      for state in range(env.nS):
        transition_table_row = env.P[state][policy[state]]
        for prob, nextstate, reward, is_terminal in transition_table_row:
          new_v[state] += prob * (reward + gamma * v[nextstate])
        if abs(new_v[state] - v[state]) >= tol:
          end_loop = False

      if end_loop or count >= max_iterations:
        break
      else:
        v = new_v

    return v, count


def evaluate_policy_async_ordered(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """

    v = np.zeros(env.nS)
    count = 0

    while True:
      end_loop = True
      count += 1

      for state in range(env.nS):
        new_v = 0
        transition_table_row = env.P[state][policy[state]]
        for prob, nextstate, reward, is_terminal in transition_table_row:
          new_v += prob * (reward + gamma * v[nextstate])
        if abs(new_v - v[state]) >= tol:
          end_loop = False
        v[state] = new_v

      if end_loop or count >= max_iterations:
        break

    return v, count


def evaluate_policy_async_randperm(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    v = np.zeros(env.nS)
    states = range(env.nS)
    count = 0

    while True:
      random.shuffle(states)
      end_loop = True
      count += 1

      for state in states:
        new_v = 0
        transition_table_row = env.P[state][policy[state]]
        for prob, nextstate, reward, is_terminal in transition_table_row:
          new_v += prob * (reward + gamma * v[nextstate])
        if abs(new_v - v[state]) >= tol:
          end_loop = False
        v[state] = new_v

      if end_loop or count >= max_iterations:
        break

    return v, count


def evaluate_policy_async_custom(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluate the value of a policy. Updates states by a student-defined
    heuristic. 

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    return np.zeros(env.nS), 0


def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.
    
    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """

    policy_stable = True

    for state in range(env.nS):
      old_action = policy[state]
      new_action = old_action
      state_v = 0
      for action in range(env.nA):
        transition_table_row = env.P[state][action]
        v = 0
        for prob, nextstate, reward, is_terminal in transition_table_row:
          v += prob * (reward + gamma * value_func[nextstate])
        if action == 0 or v >= state_v:
          state_v = v
          new_action = action
      if new_action != old_action:
        policy[state] = new_action
        policy_stable = False

    return not policy_stable, policy


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.
    
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    policy_improvement_iteration = 0
    value_iteration = 0

    while True:
      policy_improvement_iteration += 1
      value_func, count = evaluate_policy_sync(env, gamma, policy, max_iterations, tol)
      value_iteration += count
      improved, new_policy = improve_policy(env, gamma, value_func, policy)
      if not improved:
        break
      else:
        policy = new_policy

    return policy, value_func, policy_improvement_iteration, value_iteration


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """

    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    policy_improvement_iteration = 0
    value_iteration = 0

    while True:
      policy_improvement_iteration += 1
      value_func, count = evaluate_policy_async_ordered(env, gamma, policy, max_iterations, tol)
      value_iteration += count
      improved, new_policy = improve_policy(env, gamma, value_func, policy)
      if not improved:
        break
      else:
        policy = new_policy

    return policy, value_func, policy_improvement_iteration, value_iteration


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    policy_improvement_iteration = 0
    value_iteration = 0

    while True:
      policy_improvement_iteration += 1
      value_func, count = evaluate_policy_async_randperm(env, gamma, policy, max_iterations, tol)
      value_iteration += count
      improved, new_policy = improve_policy(env, gamma, value_func, policy)
      if not improved:
        break
      else:
        policy = new_policy

    return policy, value_func, policy_improvement_iteration, value_iteration


def policy_iteration_async_custom(env, gamma, max_iterations=int(1e3),
                                  tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_custom methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    return policy, value_func, 0, 0


def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """

    value_func = np.zeros(env.nS)
    iteration = 0
    while True:
      iteration += 1
      delta = 0
      new_value_func = np.zeros(env.nS)
      for state in range(env.nS):
        for action in range(env.nA):
          new_value = 0
          transition_table_row = env.P[state][action]
          for prob, nextstate, reward, is_terminal in transition_table_row:
            new_value += prob * (reward + gamma * value_func[nextstate])
          new_value_func[state] = max(new_value_func[state], new_value)
        delta = max(delta, abs(value_func[state] - new_value_func[state]))
      if delta < tol or iteration >= max_iterations:
        break
      else:
        value_func = new_value_func

    return value_func, iteration


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)
    iteration = 0
    while True:
      iteration += 1
      delta = 0
      for state in range(env.nS):
        old_state_value = value_func[state]
        for action in range(env.nA):
          new_value = 0
          transition_table_row = env.P[state][action]
          for prob, nextstate, reward, is_terminal in transition_table_row:
            new_value += prob * (reward + gamma * value_func[nextstate])
          value_func[state] = max(value_func[state], new_value)
        delta = max(delta, abs(value_func[state] - old_state_value))
      if delta < tol or iteration >= max_iterations:
        break

    return value_func, iteration


def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)
    iteration = 0
    states = range(env.nS)
    while True:
      iteration += 1
      delta = 0
      random.shuffle(states)
      for state in states:
        old_state_value = value_func[state]
        for action in range(env.nA):
          new_value = 0
          transition_table_row = env.P[state][action]
          for prob, nextstate, reward, is_terminal in transition_table_row:
            new_value += prob * (reward + gamma * value_func[nextstate])
          value_func[state] = max(value_func[state], new_value)
        delta = max(delta, abs(value_func[state] - old_state_value))
      if delta < tol or iteration >= max_iterations:
        break

    return value_func, iteration

def distance_to_goal(nS, state, goal):
    width = int(math.sqrt(nS))
    horizon_dist = abs(state % width - goal % width)
    vertical_dist = abs(state / width - goal / width)
    return horizon_dist + vertical_dist


def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    
    states = []

    for state in range(env.nS):
      terminal = True
      for action in range(env.nA):
        for prob, nextstate, reward, is_terminal in env.P[state][action]:
          if nextstate != state:
            terminal = False
          if reward == 1:
            goal = nextstate
      if not terminal:
        states.append(state)

    states.sort(key = lambda state: distance_to_goal(env.nS, state, goal))

    value_func = np.zeros(env.nS)
    iteration = 0
    individual_state_updates = 0
    while True:
      iteration += 1
      delta = 0
      for state in states:
        individual_state_updates += 1
        old_state_value = value_func[state]
        for action in range(env.nA):
          new_value = 0
          transition_table_row = env.P[state][action]
          for prob, nextstate, reward, is_terminal in transition_table_row:
            new_value += prob * (reward + gamma * value_func[nextstate])
          value_func[state] = max(value_func[state], new_value)
        delta = max(delta, abs(value_func[state] - old_state_value))
      if delta < tol or iteration >= max_iterations:
        break

    print("individual_state_updates: ", individual_state_updates)

    return value_func, iteration

