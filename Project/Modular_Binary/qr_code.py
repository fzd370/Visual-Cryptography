import pyqrcode

def qr():
  a=input("Enter the number to be converted to qrcode - ")
  q_img=pyqrcode.create(a)
  q_img.png("qrcode.png",scale=5)
  
