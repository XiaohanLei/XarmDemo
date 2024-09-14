import numpy as np
from glob import glob

file_names = glob('D:\\Codes\\XarmDemo\\data\\lift the orange block\\*.npz')
extrinsics = np.load('calibration/extrinsic.npy')
for file_name in file_names:
    data = np.load(file_name, allow_pickle=True)
    frames = data['frames']
    for frame in frames:
        
    
