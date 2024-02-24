import time
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn.functional as F

class Utils(object):
    @staticmethod
    def getCurrentTime():
        return time.time()

    @staticmethod
    def printMessage(message):
        print(f'{message}')

    @staticmethod
    def showImage(image, title='Image'):
        plt.axis('off')
        plt.title(title)
        plt.imshow(image)

    @staticmethod
    def saveImage(image, file_name):
        cv2.imwrite(file_name, image)

    @staticmethod
    def conv2d(noisy_image, kernel):
        image = torch.tensor(noisy_image).permute(2, 0, 1).unsqueeze(0)
        
        b = image[:, 0, :, :].unsqueeze(0).type(torch.cuda.FloatTensor)
        g = image[:, 1, :, :].unsqueeze(0).type(torch.cuda.FloatTensor)
        r = image[:, 2, :, :].unsqueeze(0).type(torch.cuda.FloatTensor)
        
        stride_size = 1
        b_out = F.conv2d(b, kernel, padding='same', stride=stride_size).type(torch.cuda.IntTensor)
        g_out = F.conv2d(g, kernel, padding='same', stride=stride_size).type(torch.cuda.IntTensor)
        r_out = F.conv2d(r, kernel, padding='same', stride=stride_size).type(torch.cuda.IntTensor)
        
        image_out = torch.zeros_like(image).cuda()
        image_out[:, 0, :, :] = b_out
        image_out[:, 1, :, :] = g_out
        image_out[:, 2, :, :] = r_out
        image_out = image_out.squeeze().permute(1, 2, 0)
        
        return image_out
