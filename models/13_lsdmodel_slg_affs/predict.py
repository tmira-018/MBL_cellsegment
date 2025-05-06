import logging
import json
import zarr
import gunpowder as gp
import math
import numpy as np
import os
import sys
import torch
import daisy
from funlib.geometry import Coordinate, Roi
from funlib.persistence import prepare_ds, open_ds

from model import Model

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def predict(iteration, raw_file, raw_dataset):

    # out file name, dataset
    out_file = raw_file
    out_affs_dataset = f'predictions/3d_slg-affs{iteration}'
    out_lsds_dataset = f'predictions/3d_slg-lsds{iteration}'

    # i/o shapes
    shape_increase = [20, 200, 200]
    input_shape = [20, 212, 212]
    output_shape = [4, 120, 120]
    input_shape = [x + y for x,y in zip(shape_increase, input_shape)]
    output_shape = [x + y for x,y in zip(shape_increase, output_shape)]
   
    # i/o sizes
    voxel_size = gp.Coordinate(zarr.open(f'{raw_file}/{raw_dataset}','r').attrs['voxel_size'])
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = (input_size - output_size) // 2
    
    # model, ckpt
    model = Model()
    model.eval()
    checkpoint = f'model_checkpoint_{iteration}'

    # declare array keys
    raw = gp.ArrayKey('RAW')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    pred_lsds = gp.ArrayKey('PRED_LSDS')

    # these are the small chunk requests that tile over the big volume
    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(pred_affs, output_size)
    scan_request.add(pred_lsds, output_size)

    # construct pipeline
    in_ds = open_ds(os.path.join(raw_file, raw_dataset))
    source = gp.ArraySource(raw, in_ds, True)
    
    # get total roi
    with gp.build(source):
        raw_roi = source.spec[raw].roi

    # prepare output dataset
    out_affs_ds = prepare_ds(
        store = f'{out_file}/{out_affs_dataset}',
        shape = (6, *in_ds.shape),
        offset = raw_roi.offset,
        voxel_size = voxel_size,
        dtype = np.uint8,
        axis_names= ['c^','z','y','x'],
        units = ['nm','nm','nm']
    )

    out_lsds_ds = prepare_ds(
        store = f'{out_file}/{out_lsds_dataset}',
        shape = (10, *in_ds.shape),
        offset = raw_roi.offset,
        voxel_size = voxel_size,
        dtype = np.uint8,
        axis_names= ['c^','z','y','x'],
        units = ['nm','nm','nm']
    )

    predict = gp.torch.Predict(
            model,
            checkpoint=checkpoint,
            inputs = {
                0: raw
            },
            outputs = {
                0: pred_affs,
                1: pred_lsds
            },
            array_specs = {
                pred_affs: gp.ArraySpec(roi=raw_roi.grow(context, context)),
                pred_lsds: gp.ArraySpec(roi=raw_roi.grow(context, context)),
            })

    scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
            dataset_names={
                pred_affs: out_affs_dataset,
                pred_lsds: out_lsds_dataset
            },
            store=out_file)

    pipeline = (
            source +
            gp.Normalize(raw) +
            gp.Pad(raw, None, mode="reflect") +
            gp.Unsqueeze([raw]) +
            gp.Unsqueeze([raw]) +
            predict +
            gp.Squeeze([pred_affs, pred_lsds]) +
            gp.IntensityScaleShift(pred_affs, 255, 0) +
            gp.IntensityScaleShift(pred_lsds, 255, 0) +
            write+
            scan)

    predict_request = gp.BatchRequest()

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    iteration = int(sys.argv[1]) # checkpoint number
    raw_file = sys.argv[2] # path to input zarr
    raw_dataset = sys.argv[3] # name of image dataset inside input zarr container (make sure it is normalized)

    predict(iteration, raw_file, raw_dataset)