"""import cv2
import os
path='video/real/'
files=sorted(os.listdir(path))
c=1
for i in files:
    cap = cv2.VideoCapture('video/real/'+i)
    for j in range(0,20):
        ret,frame=cap.read()
        cv2.imwrite('frames/real/'+str(c)+'.jpg',frame)
        c=c+1"""
import cv2
import os
path='video/fake/'
files=sorted(os.listdir(path))
c=1
for i in files:
    cap = cv2.VideoCapture('video/fake/'+i)
    for j in range(0,20):
        ret,frame=cap.read()
        cv2.imwrite('frames/fake/'+str(c)+'.jpg',frame)
        c=c+1




