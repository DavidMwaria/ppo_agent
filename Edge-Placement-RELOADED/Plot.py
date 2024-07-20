import torch
from IPython.display import clear_output
import matplotlib
import matplotlib.pyplot as plt

# plot reward curve for along episodes
plot_explicit = True
episode_rewards = []     # RL path + RL pattern
episode_rewards_HR = []  # Heuristic path + RL pattern
episode_rewards_RH = []  # RL path + Heuristic pattern
base_rewards = []        # Heuristic path + Heuristic pattern
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


def plot_rewards(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    rewards_t_HR = torch.tensor(episode_rewards_HR, dtype=torch.float)
    rewards_t_RH = torch.tensor(episode_rewards_RH, dtype=torch.float)
    base_rewards_t = torch.tensor(base_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # plt.plot(rewards_t.numpy())
    # plt.plot(base_rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 50:
        means = rewards_t.unfold(0, 50, 1).mean(1).view(-1)
        means_HR = rewards_t_HR.unfold(0, 50, 1).mean(1).view(-1)
        means_RH = rewards_t_RH.unfold(0, 50, 1).mean(1).view(-1)
        base_means = base_rewards_t.unfold(0, 50, 1).mean(1).view(-1)
        # means = torch.cat((torch.zeros(49), means))
        line1, = plt.plot(base_means.numpy(), label='H+H')
        line2, = plt.plot(means.numpy(), label='RL+RL')
        line3, = plt.plot(means_HR.numpy(), label='H+RL')
        line4, = plt.plot(means_RH.numpy(), label='RL+H')

        plt.legend(loc='lower right', handles=[line1, line2, line3, line4], fontsize='large')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
