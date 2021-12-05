import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 648)
cap.set(4,488)
cap.set(cv2.CAP_PROP_FPS,60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
# imgBg = cv2.imread("image/1.jpg")
width = 640
height = 480
dim = (width, height)
  
# resize background image
# resized = cv2.resize(imgBg, (640,480), interpolation = cv2.INTER_AREA)
# resized = cv2.resize(imgBg, dim, interpolation = cv2.INTER_AREA)

# for handling multiple images
listimg = os.listdir("image")
imglist = []
for imgpath in listimg:
    img=cv2.imread(f"image/{imgpath}")
    resizeimg = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    imglist.append(resizeimg)

indeximg = 0

while True:
    success, img = cap.read()
    # imgout = segmentor.removeBG(img, (255,255,255),threshold=0.8)
    imgout = segmentor.removeBG(img,imglist[indeximg],threshold=0.5)

    imgstacked = cvzone.stackImages([img,imgout],2,1)
    _, imgstack = fpsReader.update(imgstacked,color = (0,0,255))

    # cv2.imshow("Image", img)
    # cv2.imshow("Image", imgout)
    cv2.imshow("Image", imgstacked)
    key = cv2.waitKey(1)
    if key==ord('a'):
        if indeximg>0:
            indeximg -=1
    elif key==ord('d'):
        if indeximg<len(imglist)-1:
            indeximg +=1
    elif key==ord('q'):
        break

cv2.destroyAllWindows()
