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
            affs_scale,
            lsds_prediction,
            lsds_target,
            lsds_scale,
            ):

        affs_loss = self._calc_loss(affs_prediction, affs_target, affs_scale)
        lsds_loss = self._calc_loss(lsds_prediction, lsds_target, lsds_scale)

        return affs_loss + lsds_loss
    

class ScaledMixedLoss(torch.nn.Module):

    def __init__(self):
        super(ScaledMixedLoss, self).__init__()
        self.bce_loss = torch.nn.BCELoss(reduce=False)
        self.mse_loss = torch.nn.MSELoss(reduce=False)

    def forward(
            self,
            affs_prediction,
            affs_target,
            affs_scale,
            lsds_prediction,
            lsds_target,
            lsds_scale,
            ):

        unweighted_affs_loss = self.bce_loss(affs_prediction, affs_target)
        if len(torch.nonzero(affs_scale)) != 0:
            affs_mask = torch.masked_select(unweighted_affs_loss, torch.gt(affs_scale, 0))
            affs_loss = torch.mean(affs_mask)
        else:
            affs_loss = torch.mean(unweighted_affs_loss)

        unweighted_lsds_loss = self.mse_loss(lsds_prediction, lsds_target)
        if len(torch.nonzero(lsds_scale)) != 0:
            lsds_mask = torch.masked_select(unweighted_lsds_loss, torch.gt(lsds_scale, 0))
            lsds_loss = torch.mean(lsds_mask)
        else:
            lsds_loss = torch.mean(unweighted_lsds_loss)

        return affs_loss + lsds_loss