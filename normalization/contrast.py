import cv2
import sys

# Add new path to system for importing `Estimator` class
sys.path.append('../')
from estimator import Estimator

class Contrast(Estimator):
    def __init__(self, method):
        super().__init__(method)

        self.switcher = {
            'GHE': self._GHE,
            'CLAHE': self._CLAHE
        }

    def _GHE(self, image):
        """
        Global histogram equalization
        :param image: input image in BGR color space
        :return: equalized image
        """
        # Convert from BGR to HSV
        new_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Equalize the histogram of the Y channel
        new_img[:, :, 2] = cv2.equalizeHist(new_img[:, :, 2])
        # Convert back to BGR
        new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
        return new_img

    def _CLAHE(self, image):
        """
        Contrast limited adaptive histogram equalization of image
        :param image: input image in BGR color space
        :return: equalized image
        """
        # Convert from BGR to HSV
        new_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Equalize the histogram of the V channel
        new_img[:, :, 2] = clahe.apply(new_img[:, :, 2])
        # Convert back to BGR
        new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
        return new_img

# ---------------------------
# import cv2
# import os

# images = []
# list_paths = ['../input_images/' + path for path in os.listdir('../input_images') if '.jpg' in path]
# for image_path in list_paths:
#     image = cv2.imread(image_path)
#     images.append(image)

# deblur = Contrast()
# out_images = deblur.forward(images, 'CLAHE')

# for i, image in enumerate(out_images):
#     cv2.imwrite(f'../output_images/out_image_{i + 5}.jpg', image)
