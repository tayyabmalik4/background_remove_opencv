import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

segmentor = SelfiSegmentation()

path = 'image/7.jpg'
img= cv2.imread(path,-1)
resize = cv2.resize(img,(600,400))
backremove=segmentor.removeBG(resize,(255,255,255),threshold=0.95)
cv2.imshow('remove background color',backremove)
key = cv2.waitKey(0)
cv2.destroyAllWindows()