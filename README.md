# Smile_Detection
We'll be build a model that can detect smiles in a video stream in real-time using deep learning along with traditional computer vision
techniques.
To accomplish this task, we’ll be training the LetNet architecture on a dataset of images that contain faces of people who are smiling and not smiling.
Once our network is trained, we’ll detect faces in images via OpenCV’s built-in Haar cascade face detector, extract the face region of interest (ROI) from the image,
and then pass the ROI through LeNet for smile detection.
