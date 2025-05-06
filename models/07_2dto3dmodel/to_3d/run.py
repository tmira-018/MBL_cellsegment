import torch
from model import Model
from loss import ScaledMSELoss, ScaledBCELoss
from train import train_until
import logging
import glob

'''
Run training.
'''

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    max_iterations = 10000
    log_dir='log'
    checkpoint_basename='model'
    snapshot_dir='snapshots'
    save_checkpoints_every=1000
    save_snapshots_every=1000

    model = Model()
    loss = ScaledBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4)

    voxel_size = [5,1,1]

    # send it
    train_until(   
        max_iterations,
        voxel_size,
        model,
        optimizer,
        loss,
        log_dir,
        checkpoint_basename,
        snapshot_dir,
        save_checkpoints_every,
        save_snapshots_every)
