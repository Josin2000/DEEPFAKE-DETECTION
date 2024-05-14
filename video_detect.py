from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input
from os import listdir
from os.path import isfile, join
import cv2
import mediapipe
import pandas as pd
import os
from tensorflow.keras.backend import concatenate
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential

model1=load_model('Saved Models/bound_model.h5')
model2=load_model('Saved Models/face_model.h5')
model3=load_model('Saved Models/context_model.h5')
model4=load_model('Saved Models/final_model.h5')

def detectvid(filename):
	det=[]
	video_path = filename
	cap = cv2.VideoCapture(video_path)
	while True:
		# Read a frame from the video
		ret, frame = cap.read()

		if not ret:
			print("End of video.")
			break

		try:
			#################################################(Bounding Box)
			face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
			img = frame
			a=img
			gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
			faces = face.detectMultiScale(gray, 1.1, 4)
			for (x, y , w ,h) in faces:
				cv2.rectangle(a, (x-23,y-50), (x+w+25, y+h+45), (255, 0 , 0), 3)
				cropped_image=a[y-45:y+h+43,x-20:x+w+23]
				cv2.imwrite('Prediction Image/b.jpg',cropped_image)
			################################################(Face)	
			img1 = cv2.imread('Prediction Image/b.jpg')
			mp_face_mesh = mediapipe.solutions.face_mesh
			face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
			results = face_mesh.process(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
			landmarks = results.multi_face_landmarks[0]
			face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
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
			 
			for route_idx in routes_idx:
					print(f'Draw a line between {route_idx[0]}th landmark point to {route_idx[1]}th landmark point')
			routes = []
			 
			for source_idx, target_idx in routes_idx:
					 
					source = landmarks.landmark[source_idx]
					target = landmarks.landmark[target_idx]
							 
					relative_source = (int(img1.shape[1] * source.x), int(img1.shape[0] * source.y))
					relative_target = (int(img1.shape[1] * target.x), int(img1.shape[0] * target.y))
			 
					#cv2.line(a, relative_source, relative_target, (255, 255, 255), thickness = 2)
					 
					routes.append(relative_source)
					routes.append(relative_target)
			mask = np.zeros((img1.shape[0], img1.shape[1]))
			mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
			mask = mask.astype(bool)
				
			out = np.zeros_like(img1)
			#a=np.zeros_like(a)
			out[mask] = img1[mask]
			cv2.imwrite('Prediction Image/f.jpg',out)
			##############################################(Context)
			img1 = cv2.imread('Prediction Image/b.jpg')
			mp_face_mesh = mediapipe.solutions.face_mesh
			face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
			results = face_mesh.process(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
			landmarks = results.multi_face_landmarks[0]
			face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
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
			 
			for route_idx in routes_idx:
					print(f'Draw a line between {route_idx[0]}th landmark point to {route_idx[1]}th landmark point')
			routes = []
			 
			for source_idx, target_idx in routes_idx:
					 
					source = landmarks.landmark[source_idx]
					target = landmarks.landmark[target_idx]
							 
					relative_source = (int(img1.shape[1] * source.x), int(img1.shape[0] * source.y))
					relative_target = (int(img1.shape[1] * target.x), int(img1.shape[0] * target.y))
			 
					#cv2.line(a, relative_source, relative_target, (255, 255, 255), thickness = 2)
					 
					routes.append(relative_source)
					routes.append(relative_target)
			mask = np.zeros((img1.shape[0], img1.shape[1]))
			color = (0, 0, 0)
			mask = cv2.fillConvexPoly(img1, np.array(routes),color)
			mask = mask.astype(bool)

			out = np.zeros_like(img1)
			#a=np.zeros_like(a)
			out[mask] = img1[mask]
			cv2.imwrite('Prediction Image/c.jpg',out)
			#############################################
			height=128
			width=128
			path1='Prediction Image/b.jpg'
			img1=cv2.imread(path1)
			img1=cv2.resize(img1,(height,width))
			img1=np.array(img1)
			img1=np.expand_dims(img1, axis=0)
			path2='Prediction Image/f.jpg'
			img2=cv2.imread(path2)
			img2=cv2.resize(img2,(height,width))
			img2=np.array(img2)
			img2=np.expand_dims(img2, axis=0)
			path3='Prediction Image/c.jpg'
			img3=cv2.imread(path3)
			img3=cv2.resize(img3,(height,width))
			img3=np.array(img3)
			img3=np.expand_dims(img3, axis=0)
			#######################################################
							#FIRST MODEL(Bounding Box)    
			#######################################################
			pred1 = model1.predict(img1)
			#######################################################
							#SECOND MODEL(Face)   
			#######################################################
			pred2 = model2.predict(img2)
			#######################################################
							#THIRD MODEL(Context)    
			#######################################################
			pred3 = model3.predict(img3)
			#####################################################
			print(pred1,pred2,pred3)
			
			#####################################################
				  #CONCATENATION OF THREE MODELS OUTPUT    
			#####################################################
			combined=np.concatenate([pred1,pred2,pred3])
			print(combined)
		
			#######################################################
							#FInal MODEL(Bounding Box)    
			#######################################################
			p=np.expand_dims(combined, axis=0)
			pred4=model4.predict(p)
			print(pred4)
			#######################################################
			pred4=pred4[0]
			print(pred4)
			pred4=pred4[0]
			print(type(pred4))
			#######################################################
			if pred4>=0.5:  #Printing the prediction of model.
				result='Fake'
				det.append(result)
			else:
				result='Real'
				det.append(result)
			if os.path.exists('Prediction Image/b.jpg'):
				os.remove('Prediction Image/b.jpg')
			if os.path.exists('Prediction Image/f.jpg'):
				os.remove('Prediction Image/f.jpg')
			if os.path.exists('Prediction Image/c.jpg'):
				os.remove('Prediction Image/c.jpg')
		except:
			print("#")

	cap.release()
	return det