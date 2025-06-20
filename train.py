import torch

from config import args
from torch.optim import Optimizer
from torch.nn import functional as F
from model import Actor, Critic
from replay_buffer import ReplayBuffer

def train_critic(critic: Critic, critic_optimizer: Optimizer, critic_target: Critic, actor_target: Actor, replay_buffer: ReplayBuffer):

    states, actions, next_states, rewards, not_dones = replay_buffer.sample(args.batch_size)

    with torch.no_grad():

        noise = (torch.randn_like(actions) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
        next_actions = (actor_target(next_states) + noise).clamp(-actor_target.max_action, actor_target.max_action)

        target_q1, target_q2 = critic_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards + not_dones * args.gamma * target_q
    
    q1, q2 = critic(states, actions)

    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    with torch.no_grad():
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)


def train_actor(actor: Actor, actor_optimizer: Optimizer, critic: Critic, replay_buffer: ReplayBuffer):

    states, _, _, _, _ = replay_buffer.sample(args.batch_size)

    actor_loss = -critic.Q1(states, actor(states)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
