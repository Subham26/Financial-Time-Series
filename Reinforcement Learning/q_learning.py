# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 12:23:17 2021

@author: Subham
"""

import gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=True)

ACTION_SPACE = env.action_space.n
DISCOUNT = 0.9

# state = env.reset()
# action = random.randint(0, 4)
# print(state)
# state, rewards, done, info = env.step(action)
# print(state, rewards, done, info)


def print_state(state, done):
    statement = "Still Alive!"
    if done:
        statement = "Cocoa Time!" if state == 15 else "Game Over!" 
    print(state, "-", statement)
    
    
def get_action(q_map, q_table, state_row, random_rate):
    """Find max-valued actions and randomly select from them."""
    if random.random() < random_rate:
        return random.randint(0, ACTION_SPACE-1)

    action_values = q_table[state_row]
    max_indexes = np.argwhere(action_values == action_values.max())
    max_indexes = np.squeeze(max_indexes, axis=-1)
    action = np.random.choice(max_indexes)
    return action

def update_q(q_table, new_state_row, reward, old_value):
    """Returns an updated Q-value based on the Bellman Equation."""
    learning_rate = .1  # Change to be between 0 and 1.
    future_value = reward + DISCOUNT * np.max(q_table[new_state_row])
    return old_value + learning_rate * (future_value - old_value)

def play_game(q_table, q_map, random_rate, render=False):
    state = env.reset()
    step = 0
    done = False

    while not done:
        state_row = q_map[state]
        action = get_action(q_map, q_table, state_row, random_rate)
        new_state, _, done, _ = env.step(action)

        #Add new state to table and mapping if it isn't there already.
        if new_state not in q_map:
            q_map[new_state] = len(q_table)
            q_table = np.append(q_table, new_row, axis=0)
        new_state_row = q_map[new_state]

        reward = -.01  #Encourage exploration.
        if done:
            reward = 1 if new_state == 15 else -1
        current_q = q_table[state_row, action]
        q_table[state_row, action] = update_q(
            q_table, new_state_row, reward, current_q)

        step += 1
        if render:
            env.render()
            print_state(new_state, done)
        state = new_state
        
    return q_table, q_map


random_rate = 1
new_row = np.zeros((1, env.action_space.n))
q_table = np.copy(new_row)
q_map = {0: 0}

for count in range(100):
    q_table, q_map = play_game(q_table, q_map, random_rate)
    random_rate = random_rate * .99
    print(f"Try_{count}: ", q_map)
print(random_rate)
