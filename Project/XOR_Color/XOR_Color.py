import numpy as np
from PIL import Image
import sys, getopt

sys.path.insert(0, "../../utility")
import performance

import os,errno
import glob
import cv2 as cv
from matplotlib import pyplot as plt


def encrypt(input_image, share_size):
  image = np.asarray(input_image)
  (row, column, depth) = image.shape
  size =(row, column, depth, share_size)
  shares = np.random.randint(0, 256, size)
  shares[:,:,:,-1] = image.copy()
  for i in range(share_size-1):
    shares[:,:,:,-1] = shares[:,:,:,-1] ^ shares[:,:,:,i]

  return shares


def decrypt(shares):
  (row, column, depth, share_size) = shares.shape
  shares_image = shares.copy()
  for i in range(share_size-1):
    shares_image[:,:,:,-1] = shares_image[:,:,:,-1] ^ shares_image[:,:,:,i]

  final_output = shares_image[:,:,:,share_size-1]
  output_image = Image.fromarray(final_output.astype(np.uint8))
  return output_image

    
if __name__ == "__main__":

  share_size = int(sys.argv[1])
  try:
    if share_size < 2 or share_size > 8:
      print("Share size must be between 2 and 8")
      raise ValueError
  except ValueError:
    print("Input is not a valid integer!")
    exit(0)

  # create a folder to store output images
  if not os.path.isdir("Output"):
    os.makedirs("Output")

  # remove existing files in the outputs folder
  else:
    files = glob.glob("./Output/*.jpg")
    for f in files:
      os.remove(f)
      
  # create a folder to store shares of images
  if not os.path.isdir("Shares"):
    os.makedirs("Shares")

  # remove existing files in the shares folder
  else:
    files = glob.glob("./Shares/*.jpg")
    for f in files:
      os.remove(f)

  try:
    inputfile = sys.argv[2]
    input_image = Image.open(inputfile)
  except FileNotFoundError:
    print("Input file not found!")
    exit(0)
  print("Image has been uploaded successfully!")
  print("Number of shares image = ", share_size)

  shares = encrypt(input_image, share_size)

  for idx in range(share_size):
    image = Image.fromarray(shares[:,:,:,idx].astype(np.uint8))
    name = "./Shares/XOR_Share_" + str(idx+1) + ".jpg"
    image.save(name)

  output_image = decrypt(shares)

  output_image.save("./Output/Output_XOR.jpg")
  print("Decrypted image has been saved in the output folder")
  input_image.save("./Output/Input_XOR.jpg")
  
  print("Evaluation metrics : ")
  MSE = performance.MSE(inputfile, "./Output/Output_XOR.jpg")
  print("MSE = " + str(MSE))
  PSNR = performance.PSNR(inputfile, "./Output/Output_XOR.jpg")
  print("PSNR = " + str(PSNR))
  
  img=cv.imread("Output/Input_XOR.jpg")
  img1=cv.imread('Output/Output_XOR.jpg')
    
  fig = plt.figure()
  blue_histogram = cv.calcHist([img], [0], None, [256], [0, 256])
  red_histogram = cv.calcHist([img], [1], None, [256], [0, 256])
  green_histogram = cv.calcHist([img], [2], None, [256], [0, 256]) 
    
  # show original image
  fig.add_subplot(221)
  plt.title('Original Image')
  plt.axis("off")
  plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

  fig.add_subplot(222)
  plt.title("Histogram")
  plt.plot(blue_histogram,color="darkblue")
  plt.plot(green_histogram,color="green")
  plt.plot(red_histogram,color="red")

  fig.add_subplot(223)
  plt.title('Decrypted Image')
  plt.axis("off")
  plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    
  blue_histogram = cv.calcHist([img1], [0], None, [256], [0, 256])
  red_histogram = cv.calcHist([img1], [1], None, [256], [0, 256])
  green_histogram = cv.calcHist([img1], [2], None, [256], [0, 256]) 

  fig.add_subplot(224)
  plt.title('Histogram')
  plt.plot(blue_histogram,color="darkblue")
  plt.plot(green_histogram,color="green")
  plt.plot(red_histogram,color="red")
  
  plt.tight_layout()
      
  plt.show()
      # histogram of both images
    #hist1 = cv.calcHist([img],[0],None,[256],[0,256])
    #hist2 = cv.calcHist([img1],[0],None,[256],[0,256])

    # plot both the histograms and mention colors of your choice
    #plt.plot(hist1,color='red',label='Original Image')
    #plt.plot(hist2, color='green',label='Decrypted Image')
    #plt.legend()

    #plt.show()
    
  
