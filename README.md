# Real-time Handgun Detection with YOLOv8
## About
This application is used to compare three handgun detection models trained with different training datasets on YOLOv8. The custom model is trained on a custom dataset that was created by collecting YouTube videos of real-life shootings caught on surveillance which is manually annotated after pre-processing. The dataset for the hybrid model was created by adding 500 images from the Roboflow dataset to our created custom dataset.

![image](https://github.com/user-attachments/assets/a15ac54c-d660-4271-b9b0-9ae8fc4885f0)
![image](https://github.com/user-attachments/assets/d74c3d4d-8e1a-46de-984d-6944dba819a1)

## Prerequisites
1. Python (3.9)
2. YOLOv8 (Ultralytics)
3. OpenCV (cv2)
4. PyQT5

# Getting Started
To get a local copy up and running, kindly follow these steps.
## Installation
For packages compatability, Python version 3.9 is required. Setting up a virtual environment using Anaconda is recommended.
### Anaconda 
To install Anaconda, follow the guide <a href="https://docs.anaconda.com/anaconda/install/windows/">here.</a> Be sure to follow the guide closely, especially step 9.
### Steps: 
1. Install the python packages required
   ```
   pip install cv2
   pip install PyQt5
   pip install ultralytics
   ```
2. Clone the repo
   ```
   git clone https://github.com/mrtnrbld/handgun-detection-YOLOv8 
   ```
3. Go to root directory
   ```
   cd handgun-detection-YOLOv8
   ```
5. Go to app directory
   ```
   cd app
   ```
7. Run the program
   ```
   python app.py
   ```
