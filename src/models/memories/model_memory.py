from copy import deepcopy

import torch
from torch import nn


class AllModelMemory:
    """Memory storing whole models (networks) for previous tasks.

    Args:
        s_max (float): max scale of mask gate function
    """

    def __init__(self):
        # stores backbones and heads
        self.backbones: nn.ModuleList = nn.ModuleList()
        self.heads: nn.ModuleList = nn.ModuleList()
        
    def get_backbone(self, task_id: int):
        """d"""
        return self.backbones[task_id]
        
    def forward(self, x: torch.Tensor, task_id: int):
        # the forward process propagates input to logits of classes of task_id
        
        with torch.no_grad():
            feature = self.backbones[task_id](x)
            logits = self.heads(feature, task_id)
        return logits
        
    def update(self, task_id: int, backbone: torch.nn.Module, heads: torch.nn.Module):
        """Store model (including backbone and heads) of self.task_id after training it."""
        self.backbones.append(deepcopy(backbone))
        self.heads = heads



class ModelMemory:
    
    def __init__(self):
        # stores backbones and heads
        self.backbone: nn.Module = None
        self.heads: nn.ModuleList = nn.ModuleList()
        
    def get_backbone(self):
        """d"""
        return self.backbone
        
    def forward(self, x: torch.Tensor, task_id: int):
        # the forward process propagates input to logits of classes of task_id
        
        with torch.no_grad():
            feature = self.backbone(x)
            logits = self.heads(feature, task_id)
        return logits
        
    def update(self, backbone: torch.nn.Module, heads: torch.nn.Module):
        """Store model (including backbone and heads) of self.task_id after training it."""

        self.backbone = deepcopy(backbone)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.heads = heads

if __name__ == "__main__":
    pass
