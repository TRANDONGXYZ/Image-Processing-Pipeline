import cv2
import numpy as np
from scipy.signal import convolve2d
import sys

# Add new path to system for importing `Estimator` class
sys.path.append('../')
from estimator import Estimator

class Deblurring(Estimator):
    def __init__(self, method, threshold=1000, sigma=1.2, strength=1.5, num_iterations=30):
        super().__init__(method)

        self.threshold = threshold
        self.sigma = sigma
        self.strength = strength
        self.num_iterations = num_iterations
        
        self.switcher = {
            'VarianceOfLaplacian': self._varianceOfLaplacian,
            'SharpenFilter': self._sharpenFilter,
            'AdaptiveSharpenFilter': self._adaptiveSharpenFilter,
            'UnsharpMasking': self._unsharpMasking,
        }

    def _varianceOfLaplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def _sharpenFilter(self, image):
        # Create our shapening kernel, it must equal to one eventually
        kernel_sharpening = np.array([[0,-1,0],
                                    [-1, 5,-1],
                                    [0,-1, 0]])
        # applying the sharpening kernel to the input image & displaying it.
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)
        return sharpened

    def _adaptiveSharpenFilter(self, image):
        sharpen_score = self._varianceOfLaplacian(image)
        # print("Sharpen score: ", sharpen_score)
        if sharpen_score < self.threshold:
            # magic = 1 -  sharpen_score / threshold
            # print(sharpen_score / threshold)
            magic = -np.log10(sharpen_score / self.threshold + 0.1)
            # base kernel
            kernel_sharpening = np.array([[0, -1, 0],
                                        [-1, 0, -1],
                                        [0, -1, 0]])
            # scale the kernel with magic
            kernel_sharpening = magic * kernel_sharpening

            # set the center value
            kernel_sharpening[1, 1] = 1 - np.sum(kernel_sharpening)
            # print(kernel_sharpening)

            # applying the sharpening kernel to the input image.
            sharpened = cv2.filter2D(image, -1, kernel_sharpening)
            return sharpened
        else:
            return image

    def _unsharpMasking(self, image):
        # Step 1: Create a blurred image (mask)
        blurred = cv2.GaussianBlur(image, (0, 0), self.sigma)

        # Step 2: Subtract the blurred image from the original
        sharpened = cv2.addWeighted(image, 1 + self.strength, blurred, -self.strength, 0)
        return sharpened


# ---------------------------
# import cv2
# import os

# images = []
# list_paths = ['../output_images/' + path for path in os.listdir('../output_images') if '.jpg' in path]
# for image_path in list_paths:
#     image = cv2.imread(image_path)
#     images.append(image)

# deblur = Deblurring()
# out_images = deblur.forward(images, 'UnsharpMasking', {'strength': 9})

# for i, image in enumerate(out_images):
#     cv2.imwrite(f'../output_images/out_image_{i + 5}.jpg', image)
