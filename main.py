import h5py
import numpy as np

filename = 'filename.hdf5'

with h5py.File(filename, 'r') as file:
    print(file.keys())
    print(type(file.keys()))
    print(type(file.values()))
    print(type(file.items()))

np.zeros(10);
# get slice of relevant data
# grouping average
# normalize
# bandpass
# finally, ai overview
