import torch
import torch.nn as nn
import torch.nn.functional as F

from config import args

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: torch.Tensor):
        
        super().__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = torch.from_numpy(max_action).to(torch.float32).detach().to(args.device)

        self.fitness = None


    def forward(self, state: torch.Tensor):

        a = F.tanh(self.l1(state))
        a = F.tanh(self.l2(a))
        
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int):
        
        super().__init__()

        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state: torch.Tensor, action: torch.Tensor):
        
        sa = torch.cat([state, action], dim=-1)

        q1 = F.leaky_relu(self.l1(sa))
        q1 = F.leaky_relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.leaky_relu(self.l4(sa))
        q2 = F.leaky_relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2
    
    def Q1(self, state: torch.Tensor, action: torch.Tensor):

        sa = torch.cat([state, action], dim=-1)

        q1 = F.leaky_relu(self.l1(sa))
        q1 = F.leaky_relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1