import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from config import args
from model import Actor, Critic
from replay_buffer import ReplayBuffer
from learning_curve import LearningCurve

def evaluate(env, episodes: int, actor: Actor, replay_buffer: ReplayBuffer, learning_curve: LearningCurve):
    
    with torch.no_grad():

        scores: list[float] = []
        evaluate_steps = 0

        for e in range(episodes):

            state, _ = env.reset()
            done = False
            reach_steps_limit = False
            episode_score = 0

            while not (done or reach_steps_limit):
                evaluate_steps += 1
                learning_curve.add_step()

                state_tensor = torch.from_numpy(state).to(torch.float32).to(args.device)
                action = actor(state_tensor).cpu().numpy()
                next_state, reward, done, reach_steps_limit, _ = env.step(action)
                episode_score += reward
                replay_buffer.push(state, action, next_state, reward, not done)
                state = next_state

            scores.append(episode_score)
        
        return np.mean(scores), evaluate_steps