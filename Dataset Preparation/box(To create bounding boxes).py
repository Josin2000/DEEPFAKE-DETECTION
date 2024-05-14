"""import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, join
import numpy
import cv2
import mediapipe
import pandas as pd
import os
path='frames/fake/'
files=sorted(os.listdir(path))
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for i in files:
    try:
        img = cv2.imread(path+i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, 1.1, 4)
        for (x, y , w ,h) in faces:
            cv2.rectangle(img, (x-23,y-50), (x+w+25, y+h+45), (255, 0 , 0), 3)
            cropped_image=img[y-45:y+h+43,x-20:x+w+23]
            #cv2.imwrite('For Prediction/zpred/b.jpg',cropped_image)
        cv2.imwrite('box/fake/'+str(i),cropped_image)
    except:
        print('hi')"""
from os import listdir
from os.path import isfile, join
import numpy
import cv2
import mediapipe
import pandas as pd
import os
path='4f/'
files=sorted(os.listdir(path))
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for i in files:
    try:
        img = cv2.imread(path+i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, 1.1, 4)
        for (x, y , w ,h) in faces:
            cv2.rectangle(img, (x-23,y-50), (x+w+25, y+h+45), (255, 0 , 0), 3)
            cropped_image=img[y-45:y+h+43,x-20:x+w+23]
            #cv2.imwrite('For Prediction/zpred/b.jpg',cropped_image)
        cv2.imwrite('gg/'+str(i),cropped_image)
    except:
        print('hi')