#
# The functions to measure performance
#

from PIL import Image
import numpy as np
import math

# Mean Square Error
# img1 is the path to the image1
# img2 is the path to the image2
def MSE(img1, img2):
  image1 = Image.open(img1).convert('RGB')
  image2 = Image.open(img2).convert('RGB')
  
  image1 = np.asarray(image1)
  image2 = np.asarray(image2)
  
  mse = np.mean((image1 - image2) ** 2)

  return mse


# Peak Signal-to-Noise Ratio
# img1 is the path to the image1
# img2 is the path to the image2
def PSNR(img1, img2):
  image1 = Image.open(img1).convert('RGB')
  image2 = Image.open(img2).convert('RGB')
  
  image1 = np.asarray(image1)
  image2 = np.asarray(image2)
  
  mse = np.mean((image1 - image2) ** 2)
  
  if mse == 0:
    return 100

  maxPixel = 255.0
  psnr = 20 * math.log10(maxPixel / math.sqrt(mse))
  
  return psnr 
