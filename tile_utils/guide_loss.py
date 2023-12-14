from packaging import version
import torch
from torch import nn
import math, random

class NCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        feat_k = torch.stack(feat_k, 0)
        if len(feat_q) == 0: 
            feat_q = feat_k.detach()
        else:
            feat_q = torch.stack(feat_q, 0)
        
        num_regions_q = feat_q.shape[0]
        num_regions_k = feat_k.shape[0]

        # reshape features to region size
        feat_q = feat_q.view(num_regions_q, -1)
        feat_k = feat_k.view(num_regions_k, -1)

        out = torch.mm(feat_q, feat_k.transpose(1,0)) / 0.07
        out = torch.nn.functional.normalize(out,p = 1,dim = 1)
        loss = self.cross_entropy_loss(out, torch.arange(0, out.size(0), dtype=torch.long, device=feat_q.device)).mean()

        return loss

