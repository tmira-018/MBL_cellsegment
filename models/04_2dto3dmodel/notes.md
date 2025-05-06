# 2D -> 3D MODEL

# 2D UNet
* 2D Short affinities + LSDS!
* Weighted MSE Loss 
* Weighted MSE for LSDS + BCE for affs (ScaledMixedLoss)
* No Noise Augment


# 3D UNet 
* takes stacked 2D predictions as input
* 3d affinities output
* trained using synthetic 3d labels