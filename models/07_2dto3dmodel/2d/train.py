import gunpowder as gp
import numpy as np
import zarr
from funlib.persistence.arrays import open_ds
import os
from create_mask import CreateMask
from add_2d_lsds import Add2DLSDs

def train_until(
    max_iterations,
    training_volumes,
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
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    mask = gp.ArrayKey("MASK")

    gt_affs = gp.ArrayKey("GT_AFFS")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    affs_scale = gp.ArrayKey("AFFS_SCALE")
    affs_mask = gp.ArrayKey("AFFS_MASK")

    gt_lsds = gp.ArrayKey("GT_LSDS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")
    lsds_scale = gp.ArrayKey("LSDS_SCALE")

    # model, loss, optimizer
    model.train()
    batch_size = 10

    neighborhood = [
        [0, 1, 0], 
        [0, 0, 1],
    ]

    # i/o shapes
    input_shape = [1, 212, 212]
    output_shape = [1, 120, 120]

    # get voxel_size
    # assumes all volumes have the same voxel_size
    voxel_size = gp.Coordinate(zarr.open(f'{training_volumes[0]}/img_clahe','r').attrs['voxel_size'])

    # get i/o sizes
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size
    padding = (input_size - output_size) / 2

    # formulate the request for what a batch should (at least) contain
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(mask, output_size)
    request.add(gt_affs, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_scale, output_size)
    request.add(gt_lsds, output_size)
    request.add(pred_lsds, output_size)
    request.add(lsds_scale, output_size)


    # construct pipeline
    source = tuple(
        (
            gp.ArraySource(raw, open_ds(os.path.join(sample,"img_clahe")),True),
            gp.ArraySource(labels, open_ds(os.path.join(sample,"cropped_label")),False)
        ) + gp.MergeProvider()
        + gp.Normalize(raw)
        + CreateMask(labels, mask)
        + gp.Pad(raw, None)
        + gp.Pad(labels, padding)
        + gp.Pad(mask, padding)
        + gp.RandomLocation()
        + gp.Reject(mask=mask, min_masked=0.005)
        for sample in training_volumes
    )

    pipeline = source + gp.RandomProvider()

    # augmentations
    pipeline += gp.SimpleAugment(transpose_only=[1,2])

    pipeline += gp.DeformAugment(
        control_point_spacing=(voxel_size[-1] * 10, voxel_size[-1] * 10),
        jitter_sigma=(2.0, 2.0),
        spatial_dims=2,
        subsample=1,
        scale_interval=(0.9, 1.1),
        graph_raster_voxel_size=voxel_size[1:],
        p=0.5
    )

    pipeline += gp.IntensityAugment(raw, scale_min=0.9, scale_max=1.1, shift_min=-0.1, shift_max=0.1, z_section_wise=True, p=0.5)

    #pipeline += gp.NoiseAugment(raw, p=0.5)

    pipeline += gp.DefectAugment(raw, prob_low_contrast=0.05, prob_missing=0.0)

    # targets
    pipeline += Add2DLSDs(
        labels,
        gt_lsds,
        unlabelled=mask,
        lsds_mask=lsds_scale,
        sigma=(0,10*voxel_size[-1],10*voxel_size[-1]),
        downsample=2,
)

    pipeline += gp.GrowBoundary(
        labels, mask, steps=1, only_xy=True
    )

    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        unlabelled=mask,
        affinities_mask=affs_mask,
        dtype=np.float32,
    )
    
    pipeline += gp.BalanceLabels(gt_affs, affs_scale, affs_mask)

    pipeline += gp.Stack(batch_size)
    pipeline += gp.PreCache(num_workers=8, cache_size=20)
    
    # train
    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={0: raw},
        loss_inputs={
            0: pred_affs,
            1: gt_affs,
            2: affs_scale,
            3: pred_lsds,
            4: gt_lsds,
            5: lsds_scale,
        },
        outputs={
            0: pred_affs,
            1: pred_lsds
        },
        log_dir=log_dir,
        checkpoint_basename=checkpoint_basename,
        save_every=save_checkpoints_every,
    )

    pipeline += gp.Snapshot(
        dataset_names={
            raw: "raw",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
            affs_scale: "affs_scale",
            gt_lsds: "gt_lsds",
            pred_lsds: "pred_lsds",
            lsds_scale: "lsds_scaleee",
        },
        output_filename="batch_{iteration}.zarr",
        output_dir=snapshot_dir,
        every=save_snapshots_every,
    )

    #pipeline += gp.PrintProfilingStats(every=100)

    with gp.build(pipeline):
        for i in range(max_iterations):
            pipeline.request_batch(request)
