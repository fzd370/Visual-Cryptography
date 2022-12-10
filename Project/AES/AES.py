import base64
from PIL import Image
import numpy as np
import math
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import new as Random
from hashlib import sha256
from base64 import b64encode,b64decode
import cv2
import sys, getopt
import pandas as pd
import os
import glob
from matplotlib import pyplot as plt

sys.path.insert(0, "../../utility")
import performance


class AESCipher:
  def __init__(self, data, key):
    self.block_size = AES.block_size
    self.data = data
    self.key = sha256(key.encode()).digest()
    
    self.pad = lambda s : s + (self.block_size - len(s) % self.block_size) * chr (self.block_size - len(s) % self.block_size)
    self.unpad = lambda s : s[:-ord(s[len(s) - 1:])]

  # Encrypt
  def Encrypt(self):
    plain_text = self.pad(self.data)
    nounce = Random().read(AES.block_size)
    cipher = AES.new(self.key, AES.MODE_OFB, nounce)
    return b64encode(nounce + cipher.encrypt(plain_text.encode())).decode()

  # Decrypt
  def Decrypt(self):
    cipher_text = b64decode(self.data.encode())
    nounce = cipher_text[:self.block_size]
    cipher = AES.new(self.key, AES.MODE_OFB, nounce)
    return self.unpad(cipher.decrypt(cipher_text[self.block_size:])).decode()



def encryption(inputimage):
  K = "This is cryptography paper implementation"
  SK = hashlib.sha256(K.encode())

  with open(inputimage, "rb") as image:
    BI = base64.b64encode(image.read())
  BI = BI.decode("utf-8")

  # cipher image
  CI = AESCipher(BI, SK.hexdigest()).Encrypt()

  # generate KI1 and KI2
  w = 255
  h = len(K)
  
  C = np.ones((h, w, 1), dtype = "uint8")

  for i in range(h):
    j = ord(K[i])
    for k in range(w):
      if k < j:
        C[i][k][0] = 0
      else:
        break

  # divide C into R and P
  R = np.ones((h, w, 1), dtype = "uint8")
  P = np.ones((h, w, 1), dtype = "uint8")

  # fill pixel for R
  for i in range(h): 
    for j in range(w):
      r = np.random.normal(0, 1, 1)
      R[i][j][0] = r

  # fill pixel for P
  for i in range(h): 
    for j in range(w):
      p = R[i][j][0] ^ C[i][j][0]
      P[i][j][0] = p

  # generate R share
  filename = "./Shares/R.png"
  cv2.imwrite(filename, R)

  # generate P share
  filename = "./Shares/P.png"
  cv2.imwrite(filename, P)
  
  input_image = Image.open(inputfile)
  input_image.save("./Output/Input_AES.jpg")
    
  txt = []
  for ci in CI:
    ch = ord(ci)
    txt.append(int(ch))
   
  text = ""
  for t in txt:
    text += chr(t) + " "
   
  # write ciphertext to a file  
  f = open("./Output/cipher.txt", "w", encoding = "utf-8")
  f.write(text)
  f.close() 



def decryption():
  # read in the ciphertext 
  f = open("./Output/cipher.txt", "r", encoding = "utf-8")
  cipher = f.read()
  f.close()
  
  cipher = cipher.split(' ')
 
  # read in two shares 
  P = cv2.imread("./Shares/P.png")
  R = cv2.imread("./Shares/R.png")

  h = np.shape(P)[0]
  w = np.shape(P)[1]

  CK = np.ones((h, w, 1), dtype = "uint8")

  for i in range(h):
    for j in range(w):
      ck = P[i][j][0] ^ R[i][j][0]
      CK[i][j][0] = ck

  K1 = []
  for i in range(len(CK)):
    K1.append(0)

  for i in range(len(CK)):
    count = 0
    for j in range(len(CK[i])):
      if CK[i][j][0] == 0:
        count += 1
    K1[i] = chr(count)    

  K1 = "".join(K1)

  SK1 = hashlib.sha256(K1.encode())

  txt = []
  for c in cipher:
    try:
      ch = ord(c)
      txt.append(int(ch))
    except:
      print(c)

  text = ""
  for t in txt:
    text += chr(t)

  de = AESCipher(text, SK1.hexdigest()).Decrypt()
  de = de.encode("utf-8")

  with open("./Output/Output_AES.jpg", "wb") as f:
    f.write(base64.decodebytes(de))

if __name__ == "__main__":
  
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

  inputfile = sys.argv[1]
  
  encryption(inputfile)
  decryption()

  MSE = performance.MSE(inputfile, "./Output/Output_AES.jpg")
  print("MSE = " + str(MSE))
  PSNR = performance.PSNR(inputfile, "./Output/Output_AES.jpg")
  print("PSNR = " + str(PSNR))
  
  img=cv2.imread("Output/Input_AES.jpg")
  img1=cv2.imread('Output/Output_AES.jpg')
    
  fig = plt.figure()
  blue_histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
  red_histogram = cv2.calcHist([img], [1], None, [256], [0, 256])
  green_histogram = cv2.calcHist([img], [2], None, [256], [0, 256]) 
    
  # show original image
  fig.add_subplot(221)
  plt.title('Original Image')
  plt.axis("off")
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  fig.add_subplot(222)
  plt.title("Histogram")
  plt.plot(blue_histogram,color="darkblue")
  plt.plot(green_histogram,color="green")
  plt.plot(red_histogram,color="red")

  fig.add_subplot(223)
  plt.title('Decrypted Image')
  plt.axis("off")
  plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    
  blue_histogram = cv2.calcHist([img1], [0], None, [256], [0, 256])
  red_histogram = cv2.calcHist([img1], [1], None, [256], [0, 256])
  green_histogram = cv2.calcHist([img1], [2], None, [256], [0, 256]) 

  fig.add_subplot(224)
  plt.title('Histogram')
  plt.plot(blue_histogram,color="darkblue")
  plt.plot(green_histogram,color="green")
  plt.plot(red_histogram,color="red")
  
  plt.tight_layout()
      
  plt.show()
