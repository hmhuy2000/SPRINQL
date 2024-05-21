import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch import distributions as pyd
from torch.autograd import Variable, grad

import Sources.utils.utils as utils

EPS = 1e-7

def orthogonal_init_(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(DoubleQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q1 architecture
        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        # Q2 architecture
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action, both=False):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        if both:
            return q1, q2
        else:
            return torch.min(q1, q2)

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        expert_data = torch.cat([obs1, action1], 1)
        policy_data = torch.cat([obs2, action2], 1)

        alpha = torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [self.obs_dim, self.action_dim], dim=1)
        q = self.forward(interpolated_state, interpolated_action, both=True)
        ones = torch.ones(q[0].size()).to(policy_data.device)
        gradient = grad(
            outputs=q,
            inputs=interpolated,
            grad_outputs=[ones, ones],
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

class SingleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(SingleQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q architecture
        self.Q = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q = self.Q(obs_action)

        return q

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        expert_data = torch.cat([obs1, action1], 1)
        policy_data = torch.cat([obs2, action2], 1)

        alpha = torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [self.obs_dim, self.action_dim], dim=1)
        q = self.forward(interpolated_state, interpolated_action)
        ones = torch.ones(q.size()).to(policy_data.device)
        gradient = grad(
            outputs=q,
            inputs=interpolated,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

class RewardFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(RewardFunction, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_lower_bound = args.train.reward_lower_bound
        self.reward_scale = args.train.reward_scale
        self.args = args

        self.Q = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q = self.Q(obs_action)
        return self.reward_lower_bound + torch.sigmoid(q)

    def get_reward(self,obs,action):
        reward = self.forward(obs,action)*self.reward_scale
        return reward