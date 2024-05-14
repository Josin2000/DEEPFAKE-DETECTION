import cv2
import os
path='2nd step/fakeb/'
files=sorted(os.listdir(path))
a=[]
for i in files:
	a.append(i)
c=47
for j in range(0,46):
	h=a[j]
	h=str(h)
	img = cv2.imread(path+h)
	print(img)
	cv2.imwrite('bounding box/fake/'+str(c)+'.jpg',img)
	c=c+1
	
