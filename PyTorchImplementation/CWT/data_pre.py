import numpy as np
import calculate_wavelet
import numpy as np
import torch.nn.functional as F
#from Pytorch_implementation.CWT import Wavelet_CNN_Source_Network
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import time
from scipy.stats import mode
import load_pre_training_dataset
import load_evaluation_dataset
import pickle
import os
# past_path=os.getcwd()
#
# path='/Users/heesoo/PycharmProjects/pythonProject6/MyoArmbandDataset/PreTrainingDataset'
# load_pre_training_dataset.read_data(path)
examples, labels = load_evaluation_dataset.read_data('/Users/heesoo/PycharmProjects/pythonProject6/MyoArmbandDataset/EvaluationDataset',
                                                     type='Test1')

datasets = [examples, labels]
print(examples.shape)
np.save("../formatted_datasets/test1_evaluation_example",examples,allow_pickle=True)
np.save("../formatted_datasets/test1_evaluation_labels",labels,allow_pickle=True)