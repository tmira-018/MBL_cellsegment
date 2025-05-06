import gunpowder as gp
import numpy as np

class CreateMask(gp.BatchFilter):

  def __init__(self, in_array, out_array):
    self.in_array = in_array
    self.out_array = out_array

  def setup(self):

    # tell downstream nodes about the new array

    spec = self.spec[self.in_array].copy()
    spec.dtype = np.uint8

    self.provides(
      self.out_array,
      spec)

  def prepare(self, request):

    # to deliver mask, we need labels data in the same ROI
    deps = gp.BatchRequest()

    request_spec = request[self.out_array].copy()
    request_spec.dtype = np.uint32

    deps[self.in_array] = request_spec
 
    return deps

  def process(self, batch, request):

    # get the data from in_array and mask it
    data = batch[self.in_array].data

    # mask and convert to uint8
    data = (data > 0).astype(np.uint8)
    #data = np.ones_like(data).astype(np.uint8)

    # create the array spec for the new array
    spec = batch[self.in_array].spec.copy()
    spec.roi = request[self.out_array].roi.copy()
    spec.dtype = np.uint8

    # create a new batch to hold the new array
    batch = gp.Batch()

    # create a new array
    masked = gp.Array(data, spec)

    # store it in the batch
    batch[self.out_array] = masked

    # return the new batch
    return batch