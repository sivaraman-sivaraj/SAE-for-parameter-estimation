import numpy as np 
import torch

data = np.load("input_cw_ccw.npy",allow_pickle = True)

print(len(data))


print(torch.tensor(data[0]))