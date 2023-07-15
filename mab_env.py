import torch

class BanditEnv:
    """
    Multi-armed bandit environment
    payout_list:
        A list of probabilities of the likelihood that a particular bandit will pay out
    reward_list:
        A list of rewards of the payout that bandit has
    """
    def __init__(self, payout_list, reward_list):
        self.payout_list = payout_list
        self.reward_list = reward_list

    def step(self, action):
        if torch.rand(1).item() < self.payout_list[action]:
            return self.reward_list[action]
        return 0