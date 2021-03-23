import numpy as np
from PIL import Image

# Salt and Peper noise
class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), 
        p=[self.density/2.0, self.density/2.0, 1-self.density])
        mask = np.repeat(mask, c, axis=2)
        img[mask == 0] = 0
        img[mask == 1] = 255
        img= Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

# Gaussian noise
class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img