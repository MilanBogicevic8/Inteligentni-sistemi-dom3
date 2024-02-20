import numpy as np
import random
import matplotlib.pyplot as plt

import config
from environment import Action

def get_action_number(action_enum):
    return action_enum.value


def get_action_enum(number):
    return Action(number)


def get_action_eps_greedy_policy(env, q_tab, st, eps):
    prob = random.uniform(0, 1)
    return get_action_enum(np.argmax(q_tab[st])) if prob > eps else get_action_enum(env.get_random_action())


def train_q_learning(num_episodes, max_steps, lr, gamma, eps_min, eps_max, eps_dec_rate, env):
    avg_returns = []
    avg_steps = []
    q_tab = np.zeros((env.get_field_map_size(), len(env.get_all_actions())))


    for episode in range(num_episodes):
        avg_returns.append(0.)
        avg_steps.append(0)
        eps = eps_min + (eps_max - eps_min) * np.exp(-eps_dec_rate * episode)

        env.reset()
        st = env.get_agent_position()

        st_id=st[0]*env.get_field_map_length()+st[1]
       # print("---------------------------------------")
       # print(st_id)

        for step in range(max_steps):
            act = get_action_eps_greedy_policy(env, q_tab, st_id, eps)
            new_st, rew, done = env.step(act)
            new_st_id = new_st[0]*env.get_field_map_length()+new_st[1]
            #print(new_st_id)
            act=get_action_number(act)

            q_tab[st_id][act] = q_tab[st_id][act] + lr * (rew + gamma * np.max(q_tab[new_st_id]) - q_tab[st_id][act])

            if done:
                avg_returns[-1] += rew
                avg_steps[-1] += step + 1
                break

            st_id=new_st_id

    return q_tab, avg_returns, avg_steps

def evaluate_q_learning(num_episodes, max_steps, env, q_tab):
    ep_rew_lst = []
    steps_lst = []

    env.render(config.FPS)
    for episode in range(num_episodes):


        st = env.get_agent_position()
        st_id=st[0]*env.get_field_map_length()+st[1]

        env.reset()
        step_cnt = 0
        ep_rew = 0
        #env.render(config.FPS)
        for step in range(max_steps):
            act = get_action_enum(np.argmax(q_tab[st_id]))
            print(f"Step: {step + 1}, Action: {act}, State: {st}")
            new_st, rew, done = env.step(act)
            env.render(config.FPS)
            step_cnt += 1
            ep_rew += rew

            if done:
                break

            st = new_st
            st_id=new_st[0]*env.get_field_map_length()+new_st[1]

        ep_rew_lst.append(ep_rew)
        steps_lst.append(step_cnt)

    print(f'TEST Mean reward: {np.mean(ep_rew_lst):.2f}')
    print(f'TEST STD reward: {np.std(ep_rew_lst):.2f}')
    print(f'TEST Mean steps: {np.mean(steps_lst):.2f}')

def plot_results(avg_returns, avg_steps):
    episodes = list(range(1, len(avg_returns) + 1))

    # Plotting average returns
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episodes, avg_returns, label='Average Return')
    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.title('Average Return per Episode')
    plt.legend()

    # Plotting average steps
    plt.subplot(1, 2, 2)
    plt.plot(episodes, avg_steps, label='Average Steps')
    plt.xlabel('Episodes')
    plt.ylabel('Average Steps')
    plt.title('Average Steps per Episode')
    plt.legend()

    plt.tight_layout()
    plt.show()
