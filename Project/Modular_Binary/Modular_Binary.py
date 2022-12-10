import numpy as np
from PIL import Image
from qr_code import qr
import sys, getopt
import performance

import os,errno
import glob
import cv2 as cv
from matplotlib import pyplot as plt

def encrypt(input_image, share_size):
    image = np.asarray(input_image).astype(np.uint8)
    (row, column) = image.shape
    shares = np.random.randint(0, 2, size=(row, column, share_size))
    shares[:,:,-1] = image.copy()
    for i in range(share_size-1):
        shares[:,:,-1] = (shares[:,:,-1] + shares[:,:,i])%2


    return shares, image

def decrypt(shares):
    (row, column, share_size) = shares.shape
    shares_image = shares.copy()
    for i in range(share_size-1):
        shares_image[:,:,-1] = (shares_image[:,:,-1] - shares_image[:,:,i] + 2)%2

    final_output = shares_image[:,:,share_size-1]
    output_image = Image.fromarray(final_output.astype(np.uint8)*255)
    return output_image, final_output


if __name__ == "__main__":
    
  qr()
    
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
    files = glob.glob("./Output/*.png")
    for f in files:
      os.remove(f)
      
  # create a folder to store shares of images
  if not os.path.isdir("Shares"):
    os.makedirs("Shares")

  # remove existing files in the shares folder
  else:
    files = glob.glob("./Shares/*.png")
    for f in files:
      os.remove(f)

    try:
        input_image = Image.open('qrcode.png')

    except FileNotFoundError:
        print("Input file not found!")
        exit(0)

    print("Image uploaded successfully!")
    print("Input image size (in pixels) : ", input_image.size)   
    print("Number of shares image = ", share_size)

    shares, input_matrix = encrypt(input_image, share_size)

    for ind in range(share_size):
        image = Image.fromarray(shares[:,:,ind].astype(np.uint8) * 255)
        name = "Shares/MA_Share_" + str(ind+1) + ".png"
        image.save(name)

    output_image, output_matrix = decrypt(shares)
    output_image.save('Output/Output_MA.png', mode = '1')
    print("Decrypted image is saved in the output folder as Output_MA.png")
    input_image.save('Output/Input_MA.png')

    print("Evaluation metrics : ")
    MSE = performance.MSE("./Output/Input_MA.png", "./Output/Output_MA.png")
    print("MSE = " + str(MSE))
    PSNR = performance.PSNR("./Output/Input_MA.png", "./Output/Output_MA.png")
    print("PSNR = " + str(PSNR))
    
    img=cv.imread("Output/Input_MA.png",0)
    img1=cv.imread('Output/Output_MA.png',0)
    
    fig = plt.figure()

    # show original image
    fig.add_subplot(221)
    plt.title('Original Image')
    plt.set_cmap('gray')
    plt.imshow(img)

    fig.add_subplot(222)
    plt.title('Histogram ')
    plt.hist(img,10)

    fig.add_subplot(223)
    plt.title('Decrypted Image')
    plt.set_cmap('gray')
    plt.imshow(img1)

    fig.add_subplot(224)
    plt.title('Histogram')
    plt.hist(img1,10)

    plt.show()
