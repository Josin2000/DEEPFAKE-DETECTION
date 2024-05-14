DEEPFAKE DETECTION FOR HUMAN FACE IMAGES AND VIDEOS
Deepfake is a technique for fake media synthesis based on AI. Deepfakes are created by combining and superimposing existing images and videos onto source images or videos using a deeplearning technique , GAN.
This project explores the technology behind deepfakes, the growing threat they pose, and the innovative methods and tools developed to combat this formidable challenge.
The growing computation power has made creating an indistinguishable synthesized video called as deepfakes very simple.
Scenarios where deepfakes are used to create political distress, fake news, revenge porn, financial fraud are becoming common. 
In response to the growing concern over deepfake technology's impact on media credibility, the project presents a comprehensive DFD system.


Objective
To build a deep fake detection model to detect deepfake videos and deepfake 
images.

GUI
Python Tkinter will be used as GUI
GUI will contain a page for to upload video and image

Notes
We will use publically available Faceforensic++ dataset for our project implementation.

Steps
1) Load Dataset
2) Preprocessing & Face Detection
- Loading pretrained Face detection model
- Perform face detection
- Cropping of bounding boxes 
- Perform Face and Context Segmentation
- Applying resizing
- Split dataset into training and testing set
3) Customized Deep Learning Models
a) Face Network 
- Create Customized Face Network model
- Input segmented faces and get its feature vectors
b) Context Recognition Network 
- Create Customized Context Recognition Network Model
- Input segmented contexts and get its feature vectors
c) Bounding Box Network
- Create Customized Bounding Box Network Model
- Input segmented bouding boxes and get its feature vectors
d)Perform Concatenation of outputs from the 3 Networks
e) Classification Network
-Create Customized Classification Network Model
- Training the model
- Calculate Accuracy of trained model
- Save Trained model
4) Prediction Process
- Input video or image
- Loading pretrained Face detection model
- Load Trained model
- Read frames from the inputted video(Only for video)
- Perform face detection
- Cropping Bounding box
- Perform Face and Context Segmentation
- Load Bounding box network model, Face and Context network model 
- Input segmented parts to the 3 networks and get its feature vectors
- Perform Concatenation
- Prediction using the loaded model
- View results(fake or real)

  Hardware Specification
• Processor: i5 or i7 
• RAM: 8GB (Minimum)
• Hard Disk: 500GB or above

Software Specification
• Tool: Python IDLE
• Python: version3
• Operating System: Windows 10
• Front End: Python Tkinter

contact via 
: linkedin.com/in/josin-jose-721324214   
: josinkottayil123@gmail.com
