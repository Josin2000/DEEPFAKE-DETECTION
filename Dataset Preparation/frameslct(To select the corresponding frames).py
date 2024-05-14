import cv2
import os
path1='2nd step/realb'
#path2='selected frames/real'
path3='2nd step/frames/real/'
files1=sorted(os.listdir(path1))
files2=sorted(os.listdir(path3))
a=[]
b=[]
for i in files1:
	a.append(i)
for j in files2:
	b.append(j)
print(len(a))
print(len(b))
for o in range(0,46):
	for k in range(0,2080):
		if(a[o]==b[k]):
			h=b[k]
			h=str(h)
			img = cv2.imread(path3+h)
			print(img)
			cv2.imwrite('selected frames/real2/'+h,img)
