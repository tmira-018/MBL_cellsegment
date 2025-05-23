import gunpowder as gp
import numpy as np
import random
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt, gaussian_filter, maximum_filter, generate_binary_structure
from skimage.measure import label
from skimage.morphology import disk, star, ellipse
from skimage.segmentation import expand_labels, watershed
from skimage.util import random_noise

from lsd.train.gp import AddLocalShapeDescriptor
from lsd.train import LsdExtractor


class CreateLabels(gp.BatchProvider):
    def __init__(self, array_key, anisotropy_range=None, shape=(20, 20, 20), dtype=np.uint64, voxel_size=None):
        self.array_key = array_key
        self.anisotropy_range = anisotropy_range
        self.shape = shape
        self.dtype = dtype
        self.voxel_size = voxel_size
        self.ndims = None

    def setup(self):
        spec = gp.ArraySpec()

        if self.voxel_size is None:
            voxel_size = gp.Coordinate((1,) * len(self.shape))
        else:
            voxel_size = gp.Coordinate(self.voxel_size)

        spec.voxel_size = voxel_size
        self.ndims = len(spec.voxel_size)

        if self.anisotropy_range is None:
            self.anisotropy_range = (4,int(voxel_size[0]/voxel_size[1]))

        offset = gp.Coordinate((0,) * self.ndims)
        spec.roi = gp.Roi(offset, gp.Coordinate(self.shape) * spec.voxel_size)
        spec.dtype = self.dtype
        spec.interpolatable = False

        self.provides(self.array_key, spec)

    def provide(self, request):
        batch = gp.Batch()

        request_spec = request.array_specs[self.array_key]
        voxel_size = self.spec[self.array_key].voxel_size
        
        # scale request roi to voxel units
        dataset_roi = request_spec.roi / voxel_size

        # shift request roi into dataset
        dataset_roi = dataset_roi - self.spec[self.array_key].roi.get_offset() / voxel_size

        # create array spec
        array_spec = self.spec[self.array_key].copy()
        array_spec.roi = request_spec.roi

        labels = self._generate_labels(dataset_roi.to_slices())

        batch.arrays[self.array_key] = gp.Array(labels, array_spec)

        return batch

    def _generate_labels(self, slices):
        shape = tuple(s.stop - s.start for s in slices)
        labels = np.zeros(shape, self.dtype)
        anisotropy = random.randint(*self.anisotropy_range)
        labels = np.concatenate([labels] * anisotropy)
        shape = labels.shape

        choice = random.choice(["tubes", "random"])

        if choice == "tubes":
            num_points = random.randint(5, 5 * anisotropy)
            for n in range(num_points):
                z = random.randint(1, labels.shape[0] - 1)
                y = random.randint(1, labels.shape[1] - 1)
                x = random.randint(1, labels.shape[2] - 1)
                labels[z, y, x] = 1

            for z in range(labels.shape[0]):
                dilations = random.randint(1, 10)
                structs = [
                    generate_binary_structure(2, 2),
                    disk(random.randint(1, 4)),
                    star(random.randint(2, 4)),
                    ellipse(random.randint(2, 4), random.randint(2, 4))
                ]
                dilated = binary_dilation(
                    labels[z], structure=random.choice(structs), iterations=dilations
                )
                labels[z] = dilated.astype(labels.dtype)

            labels = label(labels)

            distance = labels.shape[0]
            distances, indices = distance_transform_edt(labels == 0, return_indices=True)
            expanded_labels = np.zeros_like(labels)
            dilate_mask = distances <= distance
            masked_indices = [dimension_indices[dilate_mask] for dimension_indices in indices]
            nearest_labels = labels[tuple(masked_indices)]
            expanded_labels[dilate_mask] = nearest_labels
            labels = expanded_labels

            labels[labels == 0] = np.max(labels) + 1
            labels = label(labels)[::anisotropy].astype(np.uint64)

        elif choice == "random":
            np.random.seed()
            peaks = np.random.random(shape).astype(np.float32)
            peaks = gaussian_filter(peaks, sigma=10.0)
            max_filtered = maximum_filter(peaks, 15)
            maxima = max_filtered == peaks
            seeds = label(maxima, connectivity=1)
            labels = watershed(1.0 - peaks, seeds)[::anisotropy].astype(np.uint64)

        else:
            raise AssertionError("invalid choice")

        return labels


class SmoothAugment(gp.BatchFilter):
    def __init__(self, array, blur_range=(0.0, 1.0)):
        self.array = array
        self.range = blur_range

    def process(self, batch, request):

        array = batch[self.array].data

        # different numbers will simulate noisier or cleaner array

        if len(array.shape) == 3:
            for z in range(array.shape[0]):
                sigma = random.uniform(self.range[0], self.range[1])
                array_sec = array[z]

                array[z] = np.array(
                        gaussian_filter(array_sec, sigma=sigma)
                ).astype(array_sec.dtype)
        
        elif len(array.shape) == 4:
            for z in range(array.shape[1]):
                sigma = random.uniform(self.range[0], self.range[1])
                array_sec = array[:, z]

                array[:, z] = np.array(
                    [
                        gaussian_filter(array_sec[i], sigma=sigma)
                        for i in range(array_sec.shape[0])
                    ]
                ).astype(array_sec.dtype)
        
        elif len(array.shape) == 2:                
            sigma = random.uniform(self.range[0], self.range[1])
            array = np.array(
                        gaussian_filter(array, sigma=sigma)
            ).astype(array.dtype)

        else:
            raise AssertionError("array shape is not 2d, 3d, or multi-channel 3d")

        batch[self.array].data = array


class CustomLSDs(AddLocalShapeDescriptor):
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
            labels_sec = np.copy(labels[z])

            if np.random.random() > 0.2:
                labels_sec = self._random_merge(labels_sec)

            descriptor_sec = self.extractor.get_descriptors(
                segmentation=labels_sec, voxel_size=spec.voxel_size[1:]
            )

            descriptor[:, z] = descriptor_sec

        batch = gp.Batch()
        batch[self.descriptor] = gp.Array(descriptor.astype(spec.dtype), spec)

        return batch

    def _random_merge(self, array, num_pairs_to_merge=4):
        
        unique_ids = np.unique(array)

        if len(unique_ids) < 2:
            raise ValueError("Not enough unique_ids to merge.")

        np.random.shuffle(unique_ids)

        # Determine the number of pairs we can merge
        max_pairs = len(unique_ids) // 2
        pairs_to_merge = min(num_pairs_to_merge, max_pairs)

        for _ in range(random.randrange(pairs_to_merge)):
            label1, label2 = np.random.choice(unique_ids, 2, replace=False)
            array[array == label2] = label1
            unique_ids = unique_ids[unique_ids != label2]

        return array


class IntensityAugment(gp.BatchFilter):
    """Randomly scale and shift the values of an intensity array.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify.

        scale_min (``float``):
        scale_max (``float``):
        shift_min (``float``):
        shift_max (``float``):

            The min and max of the uniformly randomly drawn scaling and
            shifting values for the intensity augmentation. Intensities are
            changed as::

                a = a.mean() + (a-a.mean())*scale + shift

        z_section_wise (``bool``):

            Perform the augmentation z-section wise. Requires 3D arrays and
            assumes that z is the first dimension.

        clip (``bool``):

            Set to False if modified values should not be clipped to [0, 1]
            Disables range check!
    """

    def __init__(
        self,
        array,
        scale_min,
        scale_max,
        shift_min,
        shift_max,
        z_section_wise=False,
        clip=True,
    ):
        self.array = array
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shift_min = shift_min
        self.shift_max = shift_max
        self.z_section_wise = z_section_wise
        self.clip = clip

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):
        raw = batch.arrays[self.array]

        assert (
            not self.z_section_wise or raw.spec.roi.dims == 3
        ), "If you specify 'z_section_wise', I expect 3D data."
        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, (
            "Intensity augmentation requires float types for the raw array (not "
            + str(raw.data.dtype)
            + "). Consider using Normalize before."
        )
        if self.clip:
            assert (
                raw.data.min() >= 0 and raw.data.max() <= 1
            ), "Intensity augmentation expects raw values in [0,1]. Consider using Normalize before."

        if self.z_section_wise:
            for z in range((raw.spec.roi / self.spec[self.array].voxel_size).shape[0]):
                if len(raw.data.shape) == 3:
                    raw.data[z] = self.__augment(
                        raw.data[z],
                        np.random.uniform(low=self.scale_min, high=self.scale_max),
                        np.random.uniform(low=self.shift_min, high=self.shift_max)
                    )
                else:
                    raw.data[:, z, :, :] = self.__augment(
                        raw.data[:, z, :, :],
                        np.random.uniform(low=self.scale_min, high=self.scale_max),
                        np.random.uniform(low=self.shift_min, high=self.shift_max)
                    )

        else:
            raw.data = self.__augment(
                raw.data,
                np.random.uniform(low=self.scale_min, high=self.scale_max),
                np.random.uniform(low=self.shift_min, high=self.shift_max),
            )

        # clip values, we might have pushed them out of [0,1]
        if self.clip:
            raw.data[raw.data > 1] = 1
            raw.data[raw.data < 0] = 0

    def __augment(self, a, scale, shift):
        return a.mean() + (a - a.mean()) * scale + shift

class CustomGrowBoundary(gp.BatchFilter):
    """Grow a boundary between regions in a label array. Does not grow at the
    border of the batch or an optionally provided mask.

    Args:

        labels (:class:`ArrayKey`):

            The array containing labels.

        mask (:class:`ArrayKey`, optional):

            A mask indicating unknown regions. This is to avoid boundaries to
            grow between labelled and unknown regions.

        max_steps (``int``, optional):

            Number of voxels (not world units!) to grow.

        background (``int``, optional):

            The label to assign to the boundary voxels.

        only_xy (``bool``, optional):

            Do not grow a boundary in the z direction.
    """

    def __init__(self, labels, mask=None, max_steps=1, background=0, only_xy=False):
        self.labels = labels
        self.mask = mask
        self.steps = max_steps
        self.background = background
        self.only_xy = only_xy

    def process(self, batch, request):
        gt = batch.arrays[self.labels]
        gt_mask = None if not self.mask else batch.arrays[self.mask]

        if gt_mask is not None:
            # grow only in area where mask and gt are defined
            crop = gt_mask.spec.roi.intersect(gt.spec.roi)

            if crop is None:
                raise RuntimeError(
                    "GT_LABELS %s and GT_MASK %s ROIs don't intersect."
                    % (gt.spec.roi, gt_mask.spec.roi)
                )
            voxel_size = self.spec[self.labels].voxel_size
            crop_in_gt = (
                crop.shift(-gt.spec.roi.offset) / voxel_size
            ).get_bounding_box()
            crop_in_gt_mask = (
                crop.shift(-gt_mask.spec.roi.offset) / voxel_size
            ).get_bounding_box()

            self.__grow(
                gt.data[crop_in_gt], gt_mask.data[crop_in_gt_mask], self.only_xy
            )

        else:
            self.__grow(gt.data, only_xy=self.only_xy)

    def __grow(self, gt, gt_mask=None, only_xy=False):
        if gt_mask is not None:
            assert (
                gt.shape == gt_mask.shape
            ), "GT_LABELS and GT_MASK do not have the same size."

        if only_xy:
            assert len(gt.shape) == 3
            for z in range(gt.shape[0]):
                self.__grow(gt[z], None if gt_mask is None else gt_mask[z])
            return

        # get all foreground voxels by erosion of each component
        foreground = np.zeros(shape=gt.shape, dtype=bool)
        masked = None
        if gt_mask is not None:
            masked = np.equal(gt_mask, 0)
        for label in np.unique(gt):
            if label == self.background:
                continue
            label_mask = gt == label
            # Assume that masked out values are the same as the label we are
            # eroding in this iteration. This ensures that at the boundary to
            # a masked region the value blob is not shrinking.
            if masked is not None:
                label_mask = np.logical_or(label_mask, masked)
            
            steps = random.choice(range(self.steps + 1))
            
            if steps > 0:
                eroded_label_mask = binary_erosion(
                    label_mask, iterations=steps, border_value=1
                )
            else:
                eroded_label_mask = label_mask
            foreground = np.logical_or(eroded_label_mask, foreground)

        # label new background
        background = np.logical_not(foreground)
        gt[background] = self.background

class ObfuscateAffs(gp.BatchFilter):
    """
    Modifies 2D affinity arrays by creating random blobs of 
    positive affinities in areas that were previously negative.
    Parameters:
        affinity_array (str): The name of the array in the batch to modify.
        blob_size_range (tuple): The range of possible blob sizes (min, max) pixels.
        num_blobs_range (tuple): The range for the number of blobs to create (min, max).
        probability (float): The probability of applying the modification to a batch.
    """

    def __init__(self, affinity_array, blob_size_range=(40, 60), num_blobs_range=(5, 20), probability=0.8):
        self.affinity_array = affinity_array
        self.blob_size_range = blob_size_range
        self.num_blobs_range = num_blobs_range
        self.probability = probability

    def process(self, batch, request):

        if np.random.random() > self.probability:
            return

        affinities = batch[self.affinity_array].data
        assert affinities.shape[0] == 2, "Expected 2 channels for 2D affinities"

        # Find boundary regions (where both channels are 0)
        boundary_mask = np.all(affinities == 0, axis=0)

        # Generate random blobs
        num_blobs = np.random.randint(*self.num_blobs_range)
        for _ in range(num_blobs):
            self._create_and_place_blob(affinities, boundary_mask)

        # Update the batch with modified affinities
        batch[self.affinity_array].data = affinities

    def _create_and_place_blob(self, affinities, boundary_mask):

        blob_size = np.random.randint(*self.blob_size_range)

        # Create a random 2D blob
        blob = np.ones((1, blob_size, blob_size), dtype=bool)
        blob = binary_dilation(blob, iterations=2)

        # Find a random position to place the blob
        valid_positions = np.where(boundary_mask)

        if len(valid_positions[0]) > 0:
            idx = np.random.randint(len(valid_positions[0]))
            z, y, x = valid_positions[0][idx], valid_positions[1][idx], valid_positions[2][idx]

            # Place the blob
            z_start, y_start, x_start = max(0, z - 1), max(0, y - blob_size//2), max(0, x - blob_size//2)
            z_end, y_end, x_end = min(affinities.shape[1], z_start + 1), min(affinities.shape[2], y_start + blob_size), min(affinities.shape[3], x_start + blob_size)
            blob_slice = blob[:z_end-z_start, :y_end-y_start, :x_end-x_start]

            # Apply the blob to both channels
            affinities[0, z_start:z_end, y_start:y_end, x_start:x_end][blob_slice] = 1
            affinities[1, z_start:z_end, y_start:y_end, x_start:x_end][blob_slice] = 1