import numpy as np
import cv2

# Load an color image in grayscale and make all the obstacles 255
img = cv2.imread('map.pgm',0)
ret,imgProcessed = cv2.threshold(img,165,255,cv2.THRESH_BINARY)

# imgProcessed is the array you are working with
print imgProcessed

#drawing of line, (img, pt1,pt2,colour,thickness)
cv2.line(imgProcessed,(0,0),(511,511),(0,0,0),1)
#a way to show the img
cv2.imshow('image',imgProcessed)
cv2.waitKey(0)
cv2.destroyAllWindows()
