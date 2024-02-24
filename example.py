import time
import multiprocessing
import cv2
import numpy as np
import os

from pipeline import NormalizationPipeline
from normalization.colorspace import ColorSpaceTransformer
from normalization.denoising import Denoising
from normalization.brightness import Brightness
from normalization.contrast import Contrast
from normalization.deblurring import Deblurring
from datamover import DataMover

from utils import Utils

data_path = '../../../../denoise_dataset/bilateral_filter/'
# data_path = './input_images/'
image_paths = [data_path + file_name for file_name in os.listdir(data_path) if '.jpg' in file_name]

images = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    images.append(image)
print(f'Total {len(images)} images in {data_path} folder')

pipeline = NormalizationPipeline([
    ('Increase brightness', Brightness(method='AdaptiveGammaCorrection', init_gamma=0.3)),
    ('Increase constrast', Contrast(method='CLAHE')),
    ('Deblur', Deblurring(method='UnsharpMasking')),
    # ('Move to GPU', DataMover(method='MoveToGPU')),
    ('Denoise', Denoising(method='GaussianFilter', filter_size=3)),
    # ('Move to GPU', DataMover(method='MoveToCPU'))
], measure_time=True)

images_out, execution_time = pipeline.forward(images)
for i, image in enumerate(images_out[:3]):
    Utils.saveImage(image, f'./output_images/image{i + 1}.jpg')
print(f'The pipeline takes {execution_time}s')
