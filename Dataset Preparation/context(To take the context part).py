import mediapipe
import cv2
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
mypath='bounding box/fake/'
onlyfiles = sorted([ f for f in listdir(mypath) if isfile(join(mypath,f)) ])
#print(onlyfiles)
#print(len(onlyfiles))
images = np.empty(len(onlyfiles), dtype=object)
print(images)
for n in range(0, len(onlyfiles)):
	images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
for o in range(0,100):
	a=images[o]
	mp_face_mesh = mediapipe.solutions.face_mesh
	face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
	results = face_mesh.process(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
	landmarks = results.multi_face_landmarks[0]
	face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
	import pandas as pd
	df = pd.DataFrame(list(face_oval), columns = ['p1','p2'])
	routes_idx = []
	 
	p1 = df.iloc[0]['p1']
	p2 = df.iloc[0]['p2']
	 
	for i in range(0, df.shape[0]):
 
			print(p1, p2)
 
			obj = df[df['p1'] == p2]
			p1 = obj['p1'].values[0]
			p2 = obj['p2'].values[0]
			 
			route_idx = []
			route_idx.append(p1)
			route_idx.append(p2)
			routes_idx.append(route_idx)
 
	# -------------------------------
	 
	#for route_idx in routes_idx:
			#print(f'Draw a line between {route_idx[0]}th landmark point to {route_idx[1]}th landmark point')
	routes = []
	 
	for source_idx, target_idx in routes_idx:
 
			source = landmarks.landmark[source_idx]
			target = landmarks.landmark[target_idx]
 
			relative_source = (int(a.shape[1] * source.x), int(a.shape[0] * source.y))
			relative_target = (int(a.shape[1] * target.x), int(a.shape[0] * target.y))
	
			#cv2.line(a, relative_source, relative_target, (255, 255, 255), thickness = 2)
 
			routes.append(relative_source)
			routes.append(relative_target)
	mask = np.zeros((a.shape[0], a.shape[1]))
	color = (0, 0, 0)
	mask = cv2.fillConvexPoly(a, np.array(routes),color)#######
	mask = mask.astype(bool)

	out = np.zeros_like(a)
	#a=np.zeros_like(a)
	out[mask] = a[mask]
	w=str(onlyfiles[o])
	cv2.imwrite('context/fake/'+w,out)