import config
from environment import Environment, Quit

from q_learning import train_q_learning, evaluate_q_learning, get_action_eps_greedy_policy, plot_results
import numpy as np


environment = Environment(f'maps/map.txt')

# Definisanje parametara za Q-learning
num_episodes = 7000
max_steps = 100
lr = 0.05
gamma = 0.95
eps_min = 0.005
eps_max = 1.0
eps_dec_rate = 0.001

environment.render(config.FPS)
environment.reset()
q_table, avg_returns, avg_steps = train_q_learning(num_episodes, max_steps, lr, gamma, eps_min, eps_max, eps_dec_rate,environment)
print(q_table,avg_returns,avg_steps)

evaluate_q_learning(1, max_steps, environment, q_table)
plot_results(avg_returns, avg_steps)
'''
try:
    environment.render(config.FPS)
    while True:
        action = environment.get_random_action()
        _, _, done = environment.step(action)
        print(environment.get_agent_position())
        environment.render(config.FPS)
        if done:
            break
except Quit:
    pass
'''
