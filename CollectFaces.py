import urllib.request
import cv2
import numpy as np
import time
from cv2 import rectangle



URL = "http://192.168.1.203:8080/shot.jpg"

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

scale_percent = 20 # percent of original size
width = int(1920 * scale_percent / 100)
height = int(1080 * scale_percent / 100)
dim = (width, height)

name = input('Enter your name: ')

for i in range(0,10000):
    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()),dtype=np.uint8)
    img = cv2.imdecode(img_arr,-1)

    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    bboxes = classifier.detectMultiScale(img)
    # print bounding box for each detected face
    for box in bboxes:
        global cropImage
        x, y, width, height = box
        x2, y2 = x + width, y + height
        rectangle(img, (x, y), (x2, y2), (0,0,255), 1)
        cropImage = np.array(img[y:y2, x:x2])

    cv2.imwrite('faces/' +  name + '/' + name + str(i) + '.jpg', cropImage)   
    cv2.imshow('IPWebcam', cropImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

