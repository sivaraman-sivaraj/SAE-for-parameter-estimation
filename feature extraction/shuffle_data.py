import os
import numpy as np
import pandas as pd
from random import shuffle

os.chdir("C:/Users/srama/OneDrive/Desktop/Parameter Estimation by SAE")
path = "C:/Users/srama/OneDrive/Desktop/Parameter Estimation by SAE/Data_set"
class_names = os.listdir(path)

# To store all the data_points
data = []

for i in class_names:
    
    # Load the corresponding class_file
    load_name = os.path.join(path, i)
    extracted_features = pd.read_csv(load_name,header=None)
    extracted_features = np.array(extracted_features)
    
    for j in extracted_features[2501:]:
        
        data.append(j)# store it in the array
        
        
# sdata = shuffle(data)

# sdata = np.asanyarray(sdata)



np.save("input_cw_ccw", data)
# csv = np.array(sdata)
# np.savetxt("input_cw_ccw.csv",csv)


