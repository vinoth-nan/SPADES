import torch
import torch.nn as nn
from torch.nn.utils import prune

## These pruning functions are taken from the repository for LAMP (ICLR'21), please see: https://github.com/jaeho-lee/layer-adaptive-sparsity

def _is_prunable_module(m):
    return (isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d))

def get_modules(model):
    modules = []
    for m in model.modules():
        if _is_prunable_module(m):
            modules.append(m)
    return modules

def prune_weights_reparam(model):
    module_list = get_modules(model)
    for m in module_list:
        prune.identity(m,name="weight")

def prune_weights_l1predefined(model,amounts):
    mlist = get_modules(model)
    for idx,m in enumerate(mlist):
        prune.l1_unstructured(m,name="weight",amount=float(amounts[idx]))

def prune_weights_global(model,amount):
    parameters_to_prune = _extract_weight_tuples(model)
    prune.global_unstructured(parameters_to_prune,pruning_method = prune.L1Unstructured,amount=amount)

def prune_weights_erk(model,amount):
    assert amount <= 1
    amounts = _compute_erk_amounts(model,amount)
    prune_weights_l1predefined(model,amounts)

def _extract_weight_tuples(model):
    mlist = get_modules(model)
    return tuple([(m,'weight') for m in mlist])

def global_prune_threshold(model,amount):
    """    This does exactly what your function "global_pruned_threshold" does.  One difference is that this method uses GPU operations. """
    flattened_weights = [w.abs().view(-1) for w in get_weights(model)]
    concat_weights = torch.cat(flattened_scores,dim=0)
    topks,_ = torch.topk(concat_scores,int(concat_scores.size(0)*amount))
    return topks[-1]

def layerwise_mp_threshold(weight,amount):
    topk=[weight.numel(), amount, int(weight.numel()*amount), weight.view(-1).abs()]
    topks,_ = torch.topk(weight.view(-1).abs(),int(weight.numel()*amount))
    return topks[-1] #topk

def _compute_erk_amounts(model,amount):
    unmaskeds = _count_unmasked_weights(model)
    erks = _compute_erks(model)
    return _amounts_from_eps(unmaskeds,erks,amount)

def _amounts_from_eps(unmaskeds,ers,amount):
    num_layers = ers.size(0)
    layers_to_keep_dense = torch.zeros(num_layers)
    total_to_survive = (1.0-amount)*unmaskeds.sum() # Total to keep.
    # Determine some layers to keep dense.
    is_eps_invalid = True
    while is_eps_invalid:
        unmasked_among_prunables = (unmaskeds*(1-layers_to_keep_dense)).sum()
        to_survive_among_prunables = total_to_survive - (layers_to_keep_dense*unmaskeds).sum()
        ers_of_prunables = ers*(1.0-layers_to_keep_dense)
        survs_of_prunables = torch.round(to_survive_among_prunables*ers_of_prunables/ers_of_prunables.sum())
        layer_to_make_dense = -1
        max_ratio = 1.0
        for idx in range(num_layers):
            if layers_to_keep_dense[idx] == 0:
                if survs_of_prunables[idx]/unmaskeds[idx] > max_ratio:
                    layer_to_make_dense = idx
                    max_ratio = survs_of_prunables[idx]/unmaskeds[idx]
        if layer_to_make_dense == -1:
            is_eps_invalid = False
        else:
            layers_to_keep_dense[layer_to_make_dense] = 1    
    amounts = torch.zeros(num_layers) 
    for idx in range(num_layers):
        if layers_to_keep_dense[idx] == 1:
            amounts[idx] = 0.0
        else:
            amounts[idx] = 1.0 - (survs_of_prunables[idx]/unmaskeds[idx])
    return amounts

def _compute_erks(model):
    wlist = get_weights(model)
    erks = torch.zeros(len(wlist))
    for idx,w in enumerate(wlist):
        if w.dim() == 4:
            erks[idx] = w.size(0)+w.size(1)+w.size(2)+w.size(3)
        else:
            erks[idx] = w.size(0)+w.size(1)
    return erks

def _count_layerwise_weights(model):
    weights = get_weights(model)
    numlist = []
    for w in weights:
        numlist.append(w.numel())
    return numlist

def _count_all_weights(model):
    weights = get_weights(model)
    total = 0
    for w in weights:
        total += w.numel()
    return total

def _count_unmasked_weights(model):
    """    Return a 1-dimensional tensor of #unmasked weights. """
    mlist = get_modules(model)
    unmaskeds = []
    for m in mlist:
        unmaskeds.append(m.weight_mask.sum())
    return torch.FloatTensor(unmaskeds)

def _normalize_scores(scores):
    # sort scores in an ascending order
    sorted_scores,sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape,device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
    # normalize by cumulative sum
    sorted_scores /= (scores.sum() - scores_cumsum)
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape,device=scores.device)
    new_scores[sorted_idx] = sorted_scores
    
    return new_scores.view(scores.shape)

def get_weights(model):
    weights = []
    for m in model.modules():
        if _is_prunable_module(m):
            weights.append(m.weight)
    return weights

def unmask(net):
  net.eval()
  mlist = get_modules(net)
  for m in mlist:
    prune.remove(m, 'weight')
  prune_weights_reparam(net)
  return net