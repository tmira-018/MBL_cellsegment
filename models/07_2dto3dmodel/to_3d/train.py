import gunpowder as gp
import numpy as np
import zarr
from funlib.persistence.arrays import open_ds
import os
from utils import CreateLabels, CustomLSDs, IntensityAugment, SmoothAugment, CustomGrowBoundary, ObfuscateAffs
from create_mask import CreateMask
from model import Model

def train_until(
    max_iterations,
    voxel_size,
    model,
    optimizer,
    loss,
    log_dir='log',
    checkpoint_basename='model',
    snapshot_dir='snapshots',
    save_checkpoints_every=1000,
    save_snapshots_every=1000
):

    # declare arrays
    labels = gp.ArrayKey("SYNTHETIC_LABELS")
    input_affs = gp.ArrayKey("INPUT_2D_AFFS")
    input_lsds = gp.ArrayKey("INPUT_2D_LSDS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    
    # model, loss, optimizer
    model.train()
    batch_size = 1

    neighborhood = [
        [1, 0, 0], 
        [0, 1, 0], 
        [0, 0, 1],
    ]

    # i/o shapes
    input_shape = [20, 212, 212]
    output_shape = [4, 120, 120]

    # get voxel_size
    # assumes all volumes have the same voxel_size
    voxel_size = gp.Coordinate(voxel_size)

    # get i/o sizes
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size
    padding = (input_size - output_size) / 2
    
    request = gp.BatchRequest()
    request.add(labels, input_size)
    request.add(input_affs, input_size)
    request.add(input_lsds, input_size)
    request.add(gt_affs, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_weights, output_size)

    # construct pipeline
    pipeline = CreateLabels(
            labels,
            shape=input_shape,
            voxel_size=voxel_size)
            
    pipeline += gp.Pad(labels,None,mode="reflect")

    pipeline += gp.DeformAugment(
        control_point_spacing=(voxel_size[0], voxel_size[0]),
        jitter_sigma=(5.0, 5.0),
        spatial_dims=2,
        subsample=1,
        scale_interval=(0.9, 1.1),
        graph_raster_voxel_size=voxel_size[1:],
    )

    pipeline += gp.ShiftAugment(
        prob_slip=0.1,
        prob_shift=0.1,
        sigma=1)

    pipeline += gp.SimpleAugment(transpose_only=[1,2])
    
    pipeline += CustomLSDs(
        labels, input_lsds, sigma=voxel_size[-1]*10, downsample=2
    )

    pipeline += CustomGrowBoundary(labels, max_steps=1, only_xy=True)

    # that is what predicted affs will look like
    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood[1:],
        labels=labels,
        affinities=input_affs,
        dtype=np.float32,
    )

    # add missing boundaries
    pipeline += ObfuscateAffs(input_affs)
    
    # add random noise
    pipeline += gp.NoiseAugment(input_affs, mode='poisson', p=0.5)
    pipeline += gp.NoiseAugment(input_lsds, mode='gaussian', p=0.5)

    # intensity
    pipeline += IntensityAugment(input_affs, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
    pipeline += IntensityAugment(input_lsds, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)

    # smooth the batch by different sigmas to simulate noisy predictions
    pipeline += SmoothAugment(input_affs, (0.5,1.5))
    pipeline += SmoothAugment(input_lsds, (0.5,1.5))

    # add defects
    pipeline += gp.DefectAugment(
        input_lsds, 
        prob_missing=0.05,
        prob_low_contrast=0.05,
        prob_deform=0.0,
        axis=1)
    
    pipeline += gp.DefectAugment(
        input_affs, 
        prob_missing=0.05,
        prob_low_contrast=0.05,
        prob_deform=0.0,
        axis=1)
    
    # now we erode - we want the gt affs to have a pixel boundary
    pipeline += gp.GrowBoundary(labels, steps=1, only_xy=True)

    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(gt_affs, affs_weights)

    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(num_workers=4, cache_size=20)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={
            "input_lsds": input_lsds,
            "input_affs": input_affs,
        },
        loss_inputs={0: pred_affs, 1: gt_affs, 2: affs_weights},
        outputs={0: pred_affs},
        save_every=save_checkpoints_every,
        log_dir=log_dir,
        checkpoint_basename=checkpoint_basename,
    )

    pipeline += gp.Squeeze([input_affs,input_lsds,gt_affs,pred_affs,affs_weights])
    
    pipeline += gp.Snapshot(
        dataset_names={
            labels: "labels",
            input_affs: "input_affs",
            input_lsds: "input_lsds",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
            affs_weights: "affs_weights",
        },
        output_filename="batch_{iteration}.zarr",
        output_dir=snapshot_dir,
        every=save_snapshots_every,
    )

    with gp.build(pipeline):
        for i in range(max_iterations):
            pipeline.request_batch(request)
