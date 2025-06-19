import torch
import torch.nn as nn

from copy import deepcopy

def phenes_to_genes(populations: list[nn.Module]) -> torch.tensor:

    genes = []

    for individual in populations:

        genes.append(torch.cat([param.view(-1) for param in individual.parameters()]))

    genes = torch.stack(genes)  # shape = (population_size , params_size)
    genes = genes.detach()

    return genes


def gene_to_phene(individual: nn.Module, gene) -> nn.Module:

    individual = deepcopy(individual)
    ctr = 0
    for param in individual.parameters():

        n = param.numel()

        param.data.copy_(gene[ctr : ctr+n].view(param.shape))
        ctr += n

    return individual