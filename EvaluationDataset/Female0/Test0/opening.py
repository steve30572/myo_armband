import numpy as np

data_read_from_file = np.fromfile('classe_18.dat',
                                              dtype=np.int16)
data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
print(data_read_from_file.shape)