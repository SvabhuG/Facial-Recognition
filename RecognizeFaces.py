import urllib.request
import cv2
import numpy as np
import time
from cv2 import rectangle
import tensorflow as tf
from tensorflow import keras

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = keras.models.load_model('faceRecognize.h5')

URL = "http://192.168.1.203:8080/shot.jpg"

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



for i in range(0,10000):
    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()),dtype=np.uint8)
    img = cv2.imdecode(img_arr,-1)
    
    bboxes = classifier.detectMultiScale(img)
    # print bounding box for each detected face
    for box in bboxes:
        global cropImage
        x, y, width, height = box
        x2, y2 = x + width, y + height
        rectangle(img, (x, y), (x2, y2), (0,0,255), 1)
        cropImage = np.array(img[y:y2, x:x2])

    face = cv2.resize(cropImage, dsize=(60, 60), interpolation=cv2.INTER_CUBIC)   
    cv2.imshow('IPWebcam', img)
    val = model.predict_classes(face.reshape(1,60,60,3))

    if val == 0:
        print("Svabhu")
    else:
        print("Praveena")
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

