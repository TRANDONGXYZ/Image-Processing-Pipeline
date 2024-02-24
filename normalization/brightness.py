import cv2
import numpy as np

import sys

# Add new path to system for importing `Estimator` class
sys.path.append('../')
from estimator import Estimator

class Brightness(Estimator):
    def __init__(self, method, init_gamma=0.3):
        super().__init__(method)

        self.init_gamma = init_gamma
        
        self.switcher = {
            'AdaptiveGammaCorrection': self._adaptiveGammaCorrection
        }

    def _mean_brightness(self, bgr_image):
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        return np.mean(hsv_image[:, :, 2])

    def _mean_hsv_brightness(self, hsv_image):
        return np.mean(hsv_image[:, :, 2])

    def _adaptiveGammaCorrection(self, image):
        """
        Adaptive gamma correction
        :param image: an image in BRG color space
        """
        # Convert to HSV color space
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Calculate mean brightness of the image thorugh V channel
        mean_brightness_value = self._mean_brightness(image)
        # Calculate the gamma value
        gamma = np.log2(self.init_gamma) / np.log2(mean_brightness_value / 255 + 1e-5)
        # print("Gamma value: ", gamma)
        # print("Mean brightness value: ", mean_brightness_value)
        # Apply gamma correction
        img_gamma_corrected = np.power(img_hsv[:, :, 2] / 255, gamma) * 255
        img_hsv[:, :, 2] = img_gamma_corrected

        # Convert back to BGR color space
        img_gamma_corrected = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return img_gamma_corrected

# image = cv2.imread('../../sample_images/test.jpg')
# new_image = Brighness.AdaptiveGammaCorrection([image])[0]
# cv2.imwrite('../../sample_images/hahaha.jpg', new_image)



# ---------------------------
# import cv2
# import os

# images = []
# list_paths = ['../input_images/' + path for path in os.listdir('../input_images') if '.jpg' in path]
# for image_path in list_paths:
#     image = cv2.imread(image_path)
#     images.append(image)

# deblur = Brightness()
# out_images = deblur.forward(images, 'AdaptiveGammaCorrection')

# for i, image in enumerate(out_images):
#     cv2.imwrite(f'../output_images/out_image_{i + 5}.jpg', image)
