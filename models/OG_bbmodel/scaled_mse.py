import torch

class ScaledMSELoss(torch.nn.Module):

    def __init__(self):
        super(ScaledMSELoss, self).__init__()

    def _calc_loss(self, pred, target, scale):

        weights = (scale * (pred - target) ** 2)

        if len(torch.nonzero(scale)) != 0:

            mask = torch.masked_select(weights, torch.gt(scale, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(weights)

        return loss

    def forward(
            self,
            affs_prediction,
            affs_target,
            affs_scale):

        affs_loss = self._calc_loss(affs_prediction, affs_target, affs_scale)

        return affs_loss