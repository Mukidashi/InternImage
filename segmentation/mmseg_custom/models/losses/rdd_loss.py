import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES


@LOSSES.register_module(force=True)
class RddCELoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='rdd_ce_loss'):
        super(RddCELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)

        # Note: for BCE loss, label < 0 is invalid.
        loss_cls = self.loss_weight * self.rdd_cross_entropy(cls_score,label,**kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


    def rdd_cross_entropy(self, pred, label):

        road_weight = 1.0
        
        pred1 = pred[0]
        pred2 = pred[1]
        gt1 = label[0]
        gt2 = label[1]

        fg_mask1 = torch.any(gt1>0.01,axis=1)
        fg_mask1 = fg_mask1.to(torch.float32)
        fg_cnt1 = torch.maximum(torch.sum(fg_mask1,dim=[1,2]),torch.tensor([0.1]).to(fg_mask1.device))
        
        fg_mask2 = torch.any(gt2>0.01,axis=1)
        fg_mask2 = fg_mask2.to(torch.float32)
        fg_cnt2 = torch.maximum(torch.sum(fg_mask2,dim=[1,2]),torch.tensor([0.1]).to(fg_mask2.device))

        ce1 = torch.sum(-torch.log(pred1+1e-8)*gt1,dim=1)
        ce2 = torch.sum(-torch.log(pred2+1e-8)*gt2,dim=1)

        loss1 = torch.sum(ce1*fg_mask1,dim=[1,2])/fg_cnt1
        loss2 = torch.sum(ce2*fg_mask2,dim=[1,2])/fg_cnt2

        loss = road_weight*torch.mean(loss1) + torch.mean(loss2)
        
        return loss

