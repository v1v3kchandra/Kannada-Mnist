# Kannada-Mnist
A CNN model is used to train and predict the handwritten digits of kannada language. A simple GUI is created to test this model using tkinter. 
## Requirements
- Python version-3.7	  
- Keras version-2.2.4	
- Opencv(cv2)	
- Tkinter	
- Pandas and numpy
## Understanding the Data
- The numericals in kannada is as follows
!["Numericals"](/demo/kannada_digits.png)
## Directory Structure
- data
  -contains the train.csv ,val.csv and test.csv file
- model
  -contains the pretrained model.
- src
  -contains train_model.py- to train the model and main.py- to open gui to draw kannada digits and predict.
- image dump
  -every image drawn is stored here( future use- could be used as dataset for training)
## Demo
  !["Demo"](/demo/Kannada_canvas.gif)
## Note
- The train.csv is compressed to .zip folder in the data folder, download and extract this folder to get the train.csv.(kindly oblige for the inconvenience caused)
- For those who want to train the model from scratch please mention the path of the 'models' folder to save the model after training, in the train_model.py
- Kindly mention the 'image_dump' folder path in the main.py to dump the images drawn on the canvas to this folder.
All queries are appreciated regarding the project.
