

from __future__ import print_function
import torch
import torch.nn as nn
from .model import*
import torch.nn.functional as F

def vq_cos_sim(embedding, input_tensor, use_dynamic_p=False,ddp=False):

    if ddp:
        embedding_weight = embedding.module.weight
    else:
        embedding_weight = embedding.weight  

    
    input_norm = F.normalize(input_tensor, p=2, dim=2)  
    embedding_norm = F.normalize(embedding_weight, p=2, dim=1) 

    similarity = torch.matmul(input_norm, embedding_norm.t())  
    cos_sim_values, indices = similarity.max(dim=2)

    if use_dynamic_p:
        
        return indices.squeeze(), cos_sim_values.squeeze()
    
    return indices.squeeze()


class RatioLossWithMSELoss(nn.Module):
    def __init__(self, total_iters, min_weight=0.001, max_weight=1,eps=torch.tensor(1e-3, dtype=torch.bfloat16)):
        super(RatioLossWithMSELoss, self).__init__()
        self.eps = eps
        self.total_iters = total_iters
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.mse=nn.MSELoss()
        self.losses={}
    def forward(self, output, target, current_iter):

        weight = self.min_weight + (self.max_weight - self.min_weight) * (current_iter / self.total_iters)
        loss = (torch.abs(target - output)) / (torch.abs(target) + self.eps)
        weighted_loss = weight * loss

        self.losses['ratio']=loss.mean()
        self.losses['mse']=self.mse(output,target)
        return weighted_loss.mean()+self.mse(output,target)
    



if __name__=='__main__':
    pass