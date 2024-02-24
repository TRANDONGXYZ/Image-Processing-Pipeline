import cv2
import bm3d
import sys
import numpy as np

from utils import Utils
import torch

# Add new path to system for importing `Estimator` class
sys.path.append('../')
from estimator import Estimator

class Denoising(Estimator):
    def __init__(self, method, filter_size=3, strength=0.8, sigma=1.0):
        super().__init__(method)

        self.filter_size = filter_size
        self.strength = strength
        self.sigma = sigma
        
        self.switcher = {
            'MeanFilter': self._meanFilter,
            'GaussianFilter': self._gaussianFilter,
            'MedianFilter': self._medianFilter,
            'UnsharpFilter': self._unsharpFilter,
            'BilateralFilter': self._bilateralFilter,
            'MeanFilterGPU': self._meanFilterGPU,
            'GaussianFilterGPU': self._gaussianFilterGPU
        }

    def _meanFilter(self, noisy_image):
        new_image = cv2.blur(noisy_image, (self.filter_size, self.filter_size))
        return new_image

    def _gaussianFilter(self, noisy_image):
        new_image = cv2.GaussianBlur(noisy_image, (self.filter_size, self.filter_size), 0)
        return new_image

    def _medianFilter(self, noisy_image):
        new_image = cv2.medianBlur(noisy_image, self.filter_size)
        return new_image

    def _sharpGrayImage(self, gray_image):
        # Median filtering
        image_mf = self._medianFilter(gray_image, self.filter_size)

        # Calculate the Laplacian
        lap = cv2.Laplacian(image_mf, cv2.CV_64F)

        # Calculate the sharpened image
        sharpen = gray_image - self.strength * lap

        # Saturate the pixels in either direction
        sharpen[sharpen > 255] = 255
        sharpen[sharpen < 0] = 0

        return sharpen
    
    def _unsharpFilter(self, noisy_image):
        new_image = np.zeros_like(noisy_image)
        for channel in range(3):
            new_image[:, :, channel] = self._sharpGrayImage(noisy_image[:, :, channel], self.filter_size, self.strength)
        return new_image

    def _bilateralFilter(self, noisy_image):
        new_image = cv2.bilateralFilter(noisy_image, 9, 75, 75)
        return new_image

    # Use GPU
    def _meanFilterGPU(self, noisy_image):
        nb_channels = 1
        kernel = torch.ones(size=(self.filter_size, self.filter_size)) / (self.filter_size * self.filter_size)
        kernel = kernel.view(1, 1, self.filter_size, self.filter_size).repeat(1, nb_channels, 1, 1)
        kernel = kernel.type(torch.cuda.FloatTensor).cuda()
        
        denoised_image = Utils.conv2d(noisy_image, kernel)
        return denoised_image
    
    def _generateGaussianFilter(self):
        """
        creates gaussian kernel with side length `filter_size` and a sigma of `sigma`
        """
        ax = np.linspace(-(self.filter_size - 1) / 2., (self.filter_size - 1) / 2., self.filter_size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(self.sigma))
        kernel = np.outer(gauss, gauss)
        return torch.tensor(kernel / np.sum(kernel))

    def _gaussianFilterGPU(self, noisy_image):
        nb_channels = 1
        kernel = self._generateGaussianFilter()
        kernel = kernel.view(1, 1, self.filter_size, self.filter_size).repeat(1, nb_channels, 1, 1)
        kernel = kernel.type(torch.cuda.FloatTensor).cuda()
        
        denoised_image = Utils.conv2d(noisy_image, kernel)
        return denoised_image

# ---------------------------
# import cv2
# import os

# images = []
# list_paths = ['../input_images/' + path for path in os.listdir('../input_images') if '.jpg' in path]
# for image_path in list_paths:
#     image = cv2.imread(image_path)
#     images.append(image)

# denoise = Denoising()
# out_images = denoise.forward(images, 'MeanFilter', {'filter_size': 9})

# for i, image in enumerate(out_images):
#     cv2.imwrite(f'../output_images/out_image_{i + 1}.jpg', image)
