import numpy as np
import torch
from torch import nn

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False
    
def evaluate(actor, env,shift,scale, num_episodes=10):
    total_returns = []

    while len(total_returns) < num_episodes:
        state,_ = env.reset()
        done = False
        total_return = 0
        with eval_mode(actor):
            while True:
                state = (state + shift) * scale
                action = actor.choose_action(state, sample=False)
                next_state, reward, done,trunc, info = env.step(action)
                total_return += reward
                state = next_state
                if done or trunc:
                    total_returns.append(total_return)
                    break

    return total_returns

def evaluate_actor(actor, env,shift,scale, num_episodes=10):
    total_returns = []

    while len(total_returns) < num_episodes:
        state,_ = env.reset()
        done = False
        total_return = 0
        with eval_mode(actor):
            while True:
                state = (state + shift) * scale
                input_state = torch.FloatTensor(state).cuda().unsqueeze(0)
                action = actor(input_state).mean.detach().cpu().numpy()[0]
                next_state, reward, done,trunc, info = env.step(action)
                total_return += reward
                state = next_state
                if done or trunc:
                    total_returns.append(total_return)
                    break

    return total_returns

def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)

def concat_data(add_batches, args):
    batch_state = []
    batch_next_state = []
    batch_action = []
    batch_reward  =[]
    batch_done = []
    
    ls_size = [0]
    for reward,batch in zip(args.expert.reward_arr,add_batches):
        add_batch_state, add_batch_next_state, add_batch_action, add_batch_reward, add_batch_done = batch
        ls_size.append(ls_size[-1]+add_batch_state.shape[0])
        batch_state.append(add_batch_state)
        batch_next_state.append(add_batch_next_state)
        batch_action.append(add_batch_action)
        batch_reward.append(torch.full_like(add_batch_reward,reward))
        batch_done.append(add_batch_done)
    batch_state = torch.cat(batch_state, dim=0)
    batch_next_state = torch.cat(batch_next_state, dim=0)
    batch_action = torch.cat(batch_action, dim=0)
    batch_reward = torch.cat(batch_reward, dim=0)
    batch_done = torch.cat(batch_done, dim=0)
    return batch_state, batch_next_state, batch_action, batch_reward, batch_done,ls_size

def evaluate(actor, env,shift,scale, num_episodes=100, vis=True):
    total_returns = []
    while len(total_returns) < num_episodes:
        state,_ = env.reset()
        done = False
        total_return = 0
        with eval_mode(actor):
            while True:
                state = (state + shift) * scale
                action = actor.choose_action(state, sample=False)
                next_state, reward, done,trunc, info = env.step(action)
                total_return += reward
                state = next_state
                if done or trunc:
                    total_returns.append(total_return)
                    break

    return total_returns

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk