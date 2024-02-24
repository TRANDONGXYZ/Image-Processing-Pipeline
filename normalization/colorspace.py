import cv2
import sys

# Add new path to system for importing `Estimator` class
sys.path.append('../')
from estimator import Estimator

class ColorSpaceTransformer(Estimator):
    def __init__(self, method):
        super().__init__(method)
        
        self.switcher = {
            'BGR2HSV': self._convertBGR2HSV,
            'HSV2BGR': self._convertHSV2BGR,
            'BGR2RGB': self._convertBGR2RGB
        }

    def _convertBGR2HSV(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv
    
    def _convertHSV2BGR(self, image):
        bgr = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return bgr
    
    def _convertBGR2RGB(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb








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
