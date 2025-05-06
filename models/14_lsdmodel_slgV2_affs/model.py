
import torch
from funlib.learn.torch.models import UNet, ConvPass

class Model(torch.nn.Module):

    def __init__(
            self,
            in_channels=1,
            num_fmaps=12,
            fmap_inc_factor=2,
            downsample_factors=((1, 2, 2), (1, 2, 2), (1, 2, 2)),
            kernel_size_down=(
                ((3, 3, 3), (3, 3, 3)),
                ((3, 3, 3), (3, 3, 3)),
                ((1, 3, 3), (1, 3, 3)),
                ((1, 3, 3), (1, 3, 3))),
            kernel_size_up=(
                ((1, 3, 3), (1, 3, 3)),
                ((3, 3, 3), (3, 3, 3)),
                ((3, 3, 3), (3, 3, 3))),
            outputs={
                "3d_affs": {"dtype": "uint8", "dims": 6},
                "3d_lsds": {"dtype": "uint8", "dims": 10}}
        ):

        super().__init__()

        self.unet = UNet(
                in_channels=in_channels,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                constant_upsample=True,
                padding="valid")

        self.affs_head = ConvPass(num_fmaps, outputs['3d_affs']['dims'], [[1, 1, 1]], activation='Sigmoid')
        self.lsds_head = ConvPass(num_fmaps, outputs['3d_lsds']['dims'], [[1, 1, 1]], activation='Sigmoid')

    def forward(self, input):

        z = self.unet(input)

        affs = self.affs_head(z)
        lsds = self.lsds_head(z)

        return affs, lsds