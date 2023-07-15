import torch
import matplotlib.pyplot as plt
from multi_armed_bandit_env import BanditEnv
from rich.progress import track


bandit_payout = [0.01, 0.015, 0.03]
bandit_reward = [1, 1, 1]
bandit_env = BanditEnv(bandit_payout, bandit_reward)

n_episode = 100000
n_action = len(bandit_payout)
action_count = torch.tensor([0. for _ in range(n_action)])
action_total_reward = [0 for _ in range(n_action)]
action_avg_reward = [[] for action in range(n_action)]

def upper_confidence_bound(Q, action_count, t):
    ucb = torch.sqrt((2 * torch.log(
        torch.tensor(float(t)))) / action_count) + Q
    return torch.argmax(ucb)

Q = torch.empty(n_action)

for episode in track(range(n_episode), description='Training'):
    action = upper_confidence_bound(Q, action_count, episode)
    reward = bandit_env.step(action)
    action_count[action] += 1
    action_total_reward[action] += reward
    Q[action] = action_total_reward[action] / action_count[action]
    for a in range(n_action):
        if action_count[a]:
            action_avg_reward[a].append(
                action_total_reward[a] / action_count[a]
            )
        else:
            action_avg_reward[a].append(0)

for action in range(n_action):
    plt.plot(action_avg_reward[action])
plt.legend(['Arm {}'.format(action) for action in range(n_action)])
plt.title('Average reward over time')
plt.xscale('log')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()