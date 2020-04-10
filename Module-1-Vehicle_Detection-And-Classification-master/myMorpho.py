import numpy as np
import cv2



def mymorpho(image):
  #opening
  k1 = np.ones((3,3),np.uint8)
  erosion = cv2.erode(image,k1,iterations = 1)

  k1 = np.ones((5,5),np.uint8)
  dilate= cv2.dilate(erosion,k1,iterations = 2)


  #closing
  k1 = np.ones((5,5),np.uint8)
  dilate1= cv2.dilate(dilate,k1,iterations = 1)

  k1 = np.ones((13,13),np.uint8)
  erosion1= cv2.erode(dilate1,k1,iterations = 1)
  return erosion1



