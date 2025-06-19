import gymnasium as gym
import torch
import numpy as np
from copy import deepcopy

from config import args
from datetime import datetime
from learning_curve import LearningCurve
from model import Actor, Critic
from replay_buffer import ReplayBuffer
from EA.ES import CEM
from EA.EA_utils import gene_to_phene, phenes_to_genes
from evaluate import evaluate
from train import train_critic, train_actor


if (__name__ == "__main__"):

    start_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ###### 建立環境 ######
    env = gym.make(args.env_name)

    ###### 設定隨機種子 ######
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    ###### 確定維度 ######
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high

    ###### 初始化 actor, critics ######
    mu_actor = Actor(state_dim, action_dim, max_action).to(args.device)

    critic = Critic(state_dim, action_dim).to(args.device)
    critic_target = deepcopy(critic)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

    ###### 初始化 EA ######
    ea = CEM()
    offsprings = ea.get_init_actor_population(mu_actor)

    ###### 初始化 replay buffer ######
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    ###### 初始化 learning curve ######
    learning_curve = LearningCurve(args.env_name, mu_actor)

    ###### 初始化環境 ######
    state , _ = env.reset()
    done = False
    reach_step_limit = False
    steps = 0
    actor_steps = 0

    ###### 最一開始的 population 計算 fitness ######
    for actor in offsprings:
        fitness, evaluate_steps = evaluate(env, 1, actor, replay_buffer, learning_curve)
        actor.fitness = fitness
        steps += evaluate_steps

    
    while (steps < args.max_steps): 

        # CEM 抽新的 offspring
        offsprings = ea.variate(offsprings, args.population_size)

        # 更新 learning curve 裡的 mu actor
        mu_actor_model = gene_to_phene(deepcopy(mu_actor), ea.mu_actor)
        learning_curve.update(mu_actor_model)

        if steps > args.start_steps:

            # 訓練一半
            for i in range(args.n_grad):

                actor = offsprings[i]
                actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
                actor_target = deepcopy(actor).requires_grad_(False)

                for _ in range(actor_steps // len(offsprings)):
                    train_critic(critic, critic_optimizer, critic_target, actor_target, replay_buffer)

                for _ in range(actor_steps):
                    train_actor(actor, actor_optimizer, critic, replay_buffer)
         
        ###### 評估所有 offspring 的 actor ######
        actor_steps = 0
        for offspring in offsprings:

            fitness, evaluate_steps = evaluate(env, 1, offspring, replay_buffer, learning_curve)
            offspring.fitness = fitness
            actor_steps += evaluate_steps

        steps += actor_steps

        print(f"steps={learning_curve.steps}  score={learning_curve.learning_curve_scores[-1]:.3f}")
        
    if args.save_result:
        learning_curve.save()

    end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
                
