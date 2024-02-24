import time
import matplotlib.pyplot as plt
import cv2
from typing import List
import numpy as np

import torch
import torch.nn.functional as F

from estimator import Estimator

class DataMover(Estimator):
    def __init__(self, method):
        super().__init__(method)
        
        self.switcher = {
            'MoveToCPU': self._moveToCPU,
            'MoveToGPU': self._moveToGPU
        }

    def _moveToCPU(self, data):
        data_cpu = data.cpu()
        out_cpu = np.array(data_cpu)
        # if isinstance(data, List):
        #     out_cpu = []
        #     for item in data_cpu:
        #         item_np = np.array(item)
        #         out_cpu.append(item_np)
        #     return out_cpu
        # else:
        #     out_cpu = np.array(data_cpu)
        return out_cpu

    def _moveToGPU(self, data):
        # if isinstance(data, List):
        #     tensors = []
        #     for item in data:
        #         item_tensor = torch.tensor(item)
        #         tensors.append(item_tensor)
        #     tensors = torch.stack(tensors)
        #     out_gpu = tensors.cuda()
        # else:
        #     item_tensor = torch.tensor(data)
        #     out_gpu = item_tensor.cuda()
        out_tensor = torch.tensor(data)
        out_gpu = out_tensor.cuda()
        return out_gpu
