import gymnasium as gym
import torch
import numpy as np

import os
import json
import datetime
import random
import string

from copy import deepcopy
from config import args
from model import Actor, Critic

class LearningCurve:

    def __init__(self, env_name: str, mu_actor: Actor):

        self.steps = 0
        self.learning_curve_steps = []
        self.learning_curve_scores = []

        self.env_name = env_name
        self.mu_actor = deepcopy(mu_actor)

        self.n_reused_list = []
        self.reused_idx_list = []
        self.n_reused_number = 0
        self.reused_idx = []

        self.test_initial_performance()


    def update(self, mu_actor: Actor, n_reused_number: int, reused_idx: list):
        
        self.mu_actor = deepcopy(mu_actor)
        self.n_reused_number = n_reused_number
        self.reused_idx = reused_idx


    def test_initial_performance(self):

        self.learning_curve_steps.append(0)
        self.learning_curve_scores.append(self.test_performance(self.mu_actor))
        self.n_reused_list.append(0)
        self.reused_idx_list.append([])


    def add_step(self):

        self.steps += 1

        if((self.steps % args.test_performance_freq == 0) and (self.steps <= args.max_steps)):

            self.learning_curve_steps.append(self.steps)
            self.learning_curve_scores.append(self.test_performance(self.mu_actor))

            self.n_reused_list.append(self.n_reused_number)
            self.reused_idx_list.append(self.reused_idx)


    def test_performance(self, actor: Actor):

        env = gym.make(self.env_name)
        env.reset(seed=args.seed + 555)

        avg_score = 0

        with torch.no_grad():
            
            for t in range(args.test_n):

                state, _ = env.reset()

                done = False
                reach_step_limit = False

                while not (done or reach_step_limit):

                    state = torch.from_numpy(state).to(torch.float32).detach().to(args.device)

                    action = actor(state)
                    action = action.cpu().numpy()

                    state, reward, done, reach_step_limit, _ = env.step(action)

                    avg_score += reward
                
        avg_score = avg_score / args.test_n

        return avg_score
    

    def save(self):

        os.makedirs(args.output_path, exist_ok=True)

        file_name = f"[{args.algorithm}][{args.env_name}][{args.seed}][{datetime.date.today()}][Learning Curve][{''.join(random.choices(string.ascii_uppercase, k=6))}].json"
        path = os.path.join(args.output_path, file_name)

        result = {
            "Config" : vars(args),
            "Learning Curve" : {
                "Steps" : self.learning_curve_steps,
                "mu_actor" : self.learning_curve_scores
            },
            "Number of reused" : self.n_reused_list,
            "Reused idndex" : self.reused_idx_list
        }

        with open(path, mode='w') as file:
            
            json.dump(result, file)