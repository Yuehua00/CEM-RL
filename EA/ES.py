import torch
import numpy as np
import math

import os
import json
import datetime
import random
import string

from config import args
from .EA_utils import phenes_to_genes, gene_to_phene
from model import Actor
from copy import deepcopy

class CEM:

    def __init__(self):
        
        self.mu_actor = None
        self.actor_sigma = None

        self.actor_parents_size = int(args.population_size * args.CEM_parents_ratio_actor)
        self.actor_parents_weights = torch.tensor([np.log((self.actor_parents_size + 1) / i) for i in range(1, self.actor_parents_size + 1)])
        self.actor_parents_weights /= self.actor_parents_weights.sum()
        self.actor_parents_weights = self.actor_parents_weights.unsqueeze(dim=0)
        self.actor_parents_weights = self.actor_parents_weights.to(torch.float32).detach().to(args.device)

        self.damp = torch.tensor(1e-3).to(torch.float32).detach().to(args.device)
        self.damp_limit = torch.tensor(1e-5).to(torch.float32).detach().to(args.device)
        self.damp_tau = torch.tensor(0.95).to(torch.float32).detach().to(args.device)

        # self.actor_cov_discount = torch.tensor(args.CEM_cov_discount_actor).to(torch.float32).detach().to(args.device)


    def get_init_actor_population(self, mu_actor: Actor) -> list[Actor]:

        self.mu_actor =  phenes_to_genes([mu_actor]).squeeze(0)  # shape = (params_size,)
        params_size = self.mu_actor.numel()

        cov = (args.CEM_sigma_init) * torch.ones(params_size, dtype=torch.float32, device=args.device)

        epsilon_half = torch.randn(args.population_size // 2, params_size, dtype=torch.float32, device=args.device)
        epsilon = torch.cat([epsilon_half, -epsilon_half], dim=0)

        new_gene = self.mu_actor + epsilon * cov.sqrt()

        actor_population: list[Actor] = []

        for i in range(args.population_size):

            actor = deepcopy(mu_actor)
            actor = gene_to_phene(actor, new_gene[i])

            actor_population.append(actor)

        return actor_population
    

    def variate(self, actor_population: list[Actor], offspring_size: int) -> list[Actor]:

        with torch.no_grad():

            actor_population = deepcopy(actor_population)
            actor_population.sort(key=lambda actor: actor.fitness, reverse=True)

            genes = phenes_to_genes(actor_population)  # shape = (population_size , params_size)

            old_mu = self.mu_actor.unsqueeze(0)  # shape = (1, params_size)
            self.mu_actor = torch.matmul(self.actor_parents_weights, genes[: self.actor_parents_size]).squeeze(0)  # (1 , actor_parents_size) * (actor_parents_size , params_size) -> (1 , params_size) -> (params_size,)

            cov = genes[ : self.actor_parents_size] - old_mu      # shape = (actor_parents_size , params_size)
            cov = cov.pow(2)                                      # shape = (actor_parents_size , params_size)
            cov = torch.matmul(self.actor_parents_weights , cov)  # (1 , actor_parents_size) * (actor_parents_size , params_size) -> (1 , params_size)
            cov += self.damp                                      # shape = (1 , params_size)

            # cov = self.actor_cov_discount * cov                   # shape = (1 , params_size)

            self.damp = self.damp_tau * self.damp + (1 - self.damp_tau) * self.damp_limit

            params_size = self.mu_actor.numel()
            epsilon_half = torch.randn(args.population_size // 2, params_size, dtype=torch.float32, device=args.device)
            epsilon = torch.cat([epsilon_half, -epsilon_half], dim=0)  # shape = (population_size , params_size)

            new_genes = self.mu_actor + epsilon * cov.sqrt()  # shape = (population_size , params_size)

            for i in range(offspring_size):

                ctr = 0

                for param in actor_population[i].parameters():

                    n = param.numel()

                    param.data.copy_(new_genes[i , ctr : ctr + n].view(param.shape))

                    ctr += n


        return actor_population[:offspring_size]


class CEM_IM:

    def __init__(self):

        self.mu_actor = None
        self.cov = None

        self.old_mu = None
        self.old_cov = None

        self.actor_parents_size = int(args.population_size * args.CEM_parents_ratio_actor)
        self.actor_parents_weights = torch.tensor([np.log((self.actor_parents_size + 1) / i) for i in range(1, self.actor_parents_size + 1)])
        self.actor_parents_weights /= self.actor_parents_weights.sum()
        self.actor_parents_weights = self.actor_parents_weights.unsqueeze(dim=0)
        self.actor_parents_weights = self.actor_parents_weights.to(torch.float32).detach().to(args.device)

        self.damp = torch.tensor(1e-3).to(torch.float32).detach().to(args.device)
        self.damp_limit = torch.tensor(1e-5).to(torch.float32).detach().to(args.device)
        self.damp_tau = torch.tensor(0.95).to(torch.float32).detach().to(args.device)

        self.n_reused_list = []
        self.reused_idx_list = []


    def old_log_pdf(self, sample):

        sample = phenes_to_genes([deepcopy(sample)])
        
        return (
            -0.5 * torch.log(2 * torch.tensor(math.pi))
            - torch.log(self.old_cov.sqrt())
            - 0.5 * ((sample - self.old_mu) / self.old_cov.sqrt()) ** 2
        ).sum()

    def new_log_pdf(self, sample):

        sample = phenes_to_genes([deepcopy(sample)])
        
        return (
            -0.5 * torch.log(2 * torch.tensor(math.pi))
            - torch.log(self.cov.sqrt())
            - 0.5 * ((sample - self.mu_actor) / self.cov.sqrt()) ** 2
        ).sum()
    
    def sample_one_actor(self, actor):
        
        actor_copy = deepcopy(actor)

        param_size = self.mu_actor.numel()
        epsilon = torch.randn(param_size, dtype=torch.float32, device=args.device)
        individual = (self.mu_actor + epsilon * self.cov.sqrt()).view(-1)

        ctr = 0
        for param in actor_copy.parameters():

            n = param.numel()

            param.data.copy_(individual[ctr : ctr + n].view(param.shape))

            ctr += n

        actor_copy.fitness = None
        
        return actor_copy


    def get_init_actor_population(self, mu_actor: Actor) -> list[Actor]:

        self.mu_actor =  phenes_to_genes([mu_actor]).squeeze(0)  # shape = (params_size,)
        params_size = self.mu_actor.numel()

        self.cov = (args.CEM_sigma_init) * torch.ones(params_size, dtype=torch.float32, device=args.device)

        self.old_mu = self.mu_actor
        self.old_cov = self.cov
        
        epsilon_half = torch.randn(args.population_size // 2, params_size, dtype=torch.float32, device=args.device)
        epsilon = torch.cat([epsilon_half, -epsilon_half], dim=0)

        new_gene = self.mu_actor + epsilon * self.cov.sqrt()

        actor_population: list[Actor] = []

        for i in range(args.population_size):

            actor = deepcopy(mu_actor)
            actor = gene_to_phene(actor, new_gene[i])

            actor_population.append(actor)

        return actor_population


    def variate(self, actor_population: list[Actor], offspring_size: int) -> list[Actor]:

        with torch.no_grad():

            actor_population = deepcopy(actor_population)
            actor_population.sort(key=lambda actor: actor.fitness, reverse=True)

            genes = phenes_to_genes(actor_population)  # shape = (population_size , params_size)

            self.old_mu = self.mu_actor.unsqueeze(0)  # shape = (1, params_size)
            self.mu_actor = torch.matmul(self.actor_parents_weights, genes[: self.actor_parents_size]).squeeze(0)  # (1 , actor_parents_size) * (actor_parents_size , params_size) -> (1 , params_size) -> (params_size,)

            self.cov = genes[ : self.actor_parents_size] - self.old_mu      # shape = (actor_parents_size , params_size)
            self.cov = self.cov.pow(2)                                      # shape = (actor_parents_size , params_size)
            self.cov = torch.matmul(self.actor_parents_weights , self.cov)  # (1 , actor_parents_size) * (actor_parents_size , params_size) -> (1 , params_size)
            self.cov += self.damp                                      # shape = (1 , params_size)

            # cov = self.actor_cov_discount * cov                   # shape = (1 , params_size)

            self.damp = self.damp_tau * self.damp + (1 - self.damp_tau) * self.damp_limit

            # importance mixing
            n_reused = 0
            n_sampled = 0

            idx_reused = []
            old_samples: list[Actor] = actor_population

            new_population: list[Actor] = [None] * args.population_size

            for i in range(args.population_size):

                sample = old_samples[i]

                if n_reused + n_sampled < args.population_size:

                    u = np.random.uniform(0, 1)
                    if np.log(u) < self.new_log_pdf(sample) - self.old_log_pdf(sample):
                        
                        new_population[n_reused] = sample
                        idx_reused.append(i)
                        n_reused += 1
                
                if n_reused + n_sampled < args.population_size:

                    new_sample = self.sample_one_actor(sample)

                    u = np.random.uniform(0, 1)
                    if np.log(1-u) >= self.old_log_pdf(new_sample) - self.new_log_pdf(new_sample):
                        
                        new_population[-n_sampled-1] = new_sample
                        n_sampled += 1

                if n_reused + n_sampled >= args.population_size:

                    break

            cpt = n_reused + n_sampled
            while cpt < args.population_size:

                new_sample = self.sample_one_actor(old_samples[0])
                new_population[cpt - n_sampled] = new_sample
                cpt += 1

            self.old_mu = self.mu_actor
            self.old_cov = self.cov

            self.n_reused_list.append(n_reused)
            self.reused_idx_list.append(idx_reused)

            return new_population
        
    
    def save(self):

        os.makedirs(args.output_path, exist_ok=True)

        file_name = f"[{args.algorithm}][{args.env_name}][{args.seed}][{datetime.date.today()}][Reused Information][{''.join(random.choices(string.ascii_uppercase, k=6))}].json"
        path = os.path.join(args.output_path, file_name)

        result = {
            "Number of reused" : self.n_reused_list,
            "Reused idndex" : self.reused_idx_list
        }

        with open(path, mode='w') as file:
            
            json.dump(result, file)