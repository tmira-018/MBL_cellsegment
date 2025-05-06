import torch
from model import Model
from loss import ScaledMSELoss, ScaledMixedLoss
from train_2d import train_until
import logging
import glob

'''
Run training.
'''

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    max_iterations = 30000
    training_volumes = glob.glob('/mnt/efs/dlmbl/G-3dseg/Kasturi_Data/converted_zarr/training/*.zarr')
    training_volumes = [training_volumes[1], training_volumes[-1]]
    print(training_volumes)
    log_dir='log'
    checkpoint_basename='model'
    snapshot_dir='snapshots'
    save_checkpoints_every=1000
    save_snapshots_every=1000

    model = Model()
    loss = ScaledMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # send it
    train_until(   
        max_iterations,
        training_volumes,
        model,
        optimizer,
        loss,
        log_dir,
        checkpoint_basename,
        snapshot_dir,
        save_checkpoints_every,
        save_snapshots_every)