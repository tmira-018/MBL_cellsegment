from lsd.train.gp import AddLocalShapeDescriptor
from lsd.train import LsdExtractor
from gunpowder.batch import Batch
from gunpowder.array import Array
import numpy as np

class Add2DLSDs(AddLocalShapeDescriptor):
    def __init__(self, segmentation, descriptor, *args, **kwargs):

        super().__init__(segmentation, descriptor, *args, **kwargs)

        self.extractor = LsdExtractor(
                self.sigma[1:], self.mode, self.downsample
        )


    def process(self, batch, request):

        labels = batch[self.segmentation].data

        spec = batch[self.segmentation].spec.copy()

        spec.dtype = np.float32

        descriptor = np.zeros(shape=(6, *labels.shape))

        for z in range(labels.shape[0]):
            labels_sec = labels[z]

            descriptor_sec = self.extractor.get_descriptors(
                segmentation=labels_sec, voxel_size=spec.voxel_size[1:]
            )

            descriptor[:, z] = descriptor_sec

        old_batch = batch
        batch = Batch()
        
        # create lsds mask array
        if self.lsds_mask and self.lsds_mask in request:

            if self.labels_mask:

                mask = self._create_mask(old_batch, self.labels_mask, descriptor)#, crop)

            else:

                mask = (labels != 0).astype(
                    np.float32
                )

                mask_shape = len(mask.shape)

                assert mask.shape[-mask_shape:] == descriptor.shape[-mask_shape:]

                mask = np.array([mask] * descriptor.shape[0])

            if self.unlabelled:

                unlabelled_mask = self._create_mask(
                    old_batch, self.unlabelled, descriptor#, crop
                )

                mask = mask * unlabelled_mask

            batch[self.lsds_mask] = Array(
                mask.astype(spec.dtype), spec.copy()
            )

        batch[self.descriptor] = Array(descriptor.astype(spec.dtype), spec)

        return batch


    def _create_mask(self, batch, mask, lsds):#, #crop):

        mask = batch.arrays[mask].data

        mask = np.array([mask] * lsds.shape[0])

        #mask = mask[(slice(None),) + crop]

        return mask