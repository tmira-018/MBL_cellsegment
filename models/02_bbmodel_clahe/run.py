# import multiprocessing as mp
# mp.set_start_method("spawn")

import torch
from model import Model
from loss import ScaledMSELoss, ScaledBCELoss
from train import train_until
import logging
import glob

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    max_iterations = 50000
    #training_volumes = glob.glob('/mnt/efs/dlmbl/G-3dseg/Mira_data/a_/volume.zarr')
    training_volumes = ['/mnt/efs/dlmbl/G-3dseg/Mira_data/a_/volume.zarr']
    log_dir='log'
    checkpoint_basename='model'
    snapshot_dir='snapshots'
    save_checkpoints_every=1000
    save_snapshots_every=1000

    model = Model()
    loss = ScaledMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

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