import torch
import torch.nn as nn 

from policy import PolicyNet
from requirement import RequirementNet

class Model(nn.Module):
    """ Model for AI fashion coordinator """
    def __init__(self, emb_size, key_size, mem_size, 
                 meta_size, hops, item_size, 
                 coordi_size, eval_node, num_rnk, 
                 use_batch_norm, use_dropout, zero_prob,
                 use_multimodal, img_feat_size):
        """
        initialize and declare variables
        """
        super().__init__()
        # class instance for requirement estimation
        self._requirement = RequirementNet(emb_size, key_size, 
                                    mem_size, meta_size, hops)
        # class instance for ranking
        self._policy = PolicyNet(emb_size, key_size, item_size, 
                                 meta_size, coordi_size, eval_node,
                                 num_rnk, use_batch_norm,
                                 use_dropout, zero_prob,
                                 use_multimodal, img_feat_size)

    def forward(self, dlg, crd):
        """
        build graph
        """
        req = self._requirement(dlg)
        logits = self._policy(req, crd)
        preds = torch.argmax(logits, 1)
        return logits, preds