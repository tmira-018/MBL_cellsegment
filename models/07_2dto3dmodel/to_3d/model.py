
import torch
from funlib.learn.torch.models import UNet, ConvPass

class Model(torch.nn.Module):

    def __init__(
            self,
            in_channels=8,
            num_fmaps=6,
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
                "3d_affs": {"dtype": "uint8", "dims": 3}}
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

    def forward(self, input_affs, input_lsds):

        z = torch.cat((input_affs,input_lsds),dim=1)
        z = self.unet(z)

        affs = self.affs_head(z)

        return affs