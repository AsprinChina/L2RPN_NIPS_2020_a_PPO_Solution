"""
In this file, a PPO reward is designed.
"""
from grid2op.Reward.BaseReward import BaseReward


class PPO_Reward(BaseReward):
    def __init__(self):
        BaseReward.__init__(self)

    def initialize(self, env):
        self.reward_min = -10
        self.reward_std = 2

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done or is_illegal or is_ambiguous or has_error:
            return self.reward_min
        rho_max = env.get_obs().rho.max()
        return self.reward_std - rho_max * (1 if rho_max < 0.95 else 2)
