import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from tkinter import Label,Canvas,Entry,PhotoImage,Button,SUNKEN,FLAT,GROOVE,Toplevel,messagebox
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
from video_detect import detectvid
from tensorflow.keras import Sequential
customtkinter.set_appearance_mode("Light")
global filename,label1
filename=0
label1=0
t=customtkinter.CTk()
t.title("Deep Fake Detection")
t.minsize(900,550)
t.maxsize(900,550)

frame1 = customtkinter.CTkFrame(t,
							   width=465,
							   height=550,
							   corner_radius=0,fg_color='#ffc14d')
frame1.place(x=0,y=0)
frame1i = customtkinter.CTkFrame(frame1,width=415,
							   height=470,corner_radius=20,fg_color='#ffdb99')
frame1i.place(x=25,y=40)
im=ImageTk.PhotoImage(Image.open("Extra/r.png").resize((380,380)))
icon= Label(frame1i, image=im, bg='#ffdb99')
icon.image = im
icon.place(x=8, y=45)
frame2= customtkinter.CTkFrame(t,width=300,
							   height=350,corner_radius=20)
frame2.place(x=531,y=119)
frame2i=customtkinter.CTkFrame(frame2,width=250,
							   height=320,corner_radius=20)
frame2i.place(x=25,y=10)
frame2i1=customtkinter.CTkFrame(frame2i,width=230,
							   height=170,corner_radius=20)
frame2i1.place(x=10,y=70)
label = Label(t, text='DEEP FAKE DETECTION', font=('Arial', 26, 'bold'), bg='#ebebec',fg="#3e3e42")
label.place(x=477, y=35)
imw=ImageTk.PhotoImage(Image.open("Extra/fg.png").resize((45, 45)))
iconww= Label(frame2i1, image=imw, bg='#d1d5d8')
iconww.image = imw
iconww.place(x=91, y=60)
global model1,model2,model3,model4
model1=load_model('Saved Models/bound_model.h5')
model2=load_model('Saved Models/face_model.h5')
model3=load_model('Saved Models/context_model.h5')
model4=load_model('Saved Models/final_model.h5')
def pred():
	global filename,model1,model2,model3,model4
	try:
		file_extension = os.path.splitext(filename)[1]
		
	except:
		print("#")
	if(filename==0):
		messagebox.showinfo("","Please select an image first.")
	elif(file_extension==".mp4"):
		ab=detectvid(filename)
		print("Predictions:",ab)
		max_occurred_element = max(set(ab), key=ab.count)
		messagebox.showinfo("Prediction",max_occurred_element)
	else:
		f=str(filename)
		#################################################(Bounding Box)
		face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		img = cv2.imread(f)
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
		#print("hi")
		#####################################################
			  #CONCATENATION OF THREE MODELS OUTPUT    
		#####################################################
		combined=np.concatenate([pred1,pred2,pred3])
		print(combined)
		#print("hello")
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
			messagebox.showinfo("Prediction","Fake")
		else:
			result='Real'
			messagebox.showinfo("Prediction","Real")
			#details()
		if(result=='Fake'): 
			#details()
			pass
		else:
			pass

def refresh():
	global label1
	if os.path.exists('Prediction Image/b.jpg'):
		os.remove('Prediction Image/b.jpg')
	if os.path.exists('Prediction Image/f.jpg'):
		os.remove('Prediction Image/f.jpg')
	if os.path.exists('Prediction Image/c.jpg'):
		os.remove('Prediction Image/c.jpg')
	if(label1==0):
		messagebox.showinfo("","Please select an image first.")	
	else:
		label1.destroy()
		label1=0

def browseFiles(fr):
	global filename,label1
	filename = filedialog.askopenfilename(initialdir ="D:/Machine_Learning_and_Deep_Learning/Deep Fake Detection/For Testing",title = "Select Image",
		filetypes=[('Image Files', ['.jpeg', '.jpg', '.png', '.gif','.tiff', '.tif', '.bmp'])])
	filename=str(filename)
	image = Image.open(filename)
	resize_image = image.resize((215, 135))
 
	img = ImageTk.PhotoImage(resize_image)

	label1 = Label(fr,image=img)
	label1.image = img
	label1.place(x=5,y=15)

def browseFilesV(fr):
	global filename,label1
	filename = filedialog.askopenfilename(initialdir ="D:/Machine_Learning_and_Deep_Learning/Deep Fake Detection/For Testing",title = "Select Image",
		filetypes=[('Video Files', ['.mp4'])])
	img=ImageTk.PhotoImage(Image.open("Extra/vid.png").resize((215, 135)))
	label1 = Label(fr,image=img)
	label1.image = img
	label1.place(x=5,y=15)
button = customtkinter.CTkButton(frame2i,text="SELECT IMAGE",border_width=3,fg_color="#94b8b8",hover_color="#cce6ff",cursor='hand2',command=lambda:browseFiles(frame2i1))
button.place(x=50, y=5)
button = customtkinter.CTkButton(frame2i,text="SELECT VIDEO",border_width=3,fg_color="#94b8b8",hover_color="#cce6ff",cursor='hand2',command=lambda:browseFilesV(frame2i1))
button.place(x=50, y=35)
button1 = customtkinter.CTkButton(frame2i, text="Predict",border_width=2,text_color="white",corner_radius=20,cursor='hand2',width=20,height=35,command=lambda:pred())
button1.place(x=83,y=260)
rimg=ImageTk.PhotoImage(Image.open("Extra/refresh.png").resize((20,20)))
refreshb= customtkinter.CTkButton(frame2i, image=rimg, text="", width=30, height=30,
                                                corner_radius=10, fg_color="gray40", hover_color="gray25",
                                                cursor='hand2',command=lambda:refresh())
refreshb.place(x=196,y=19)
t.mainloop()