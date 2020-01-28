import os
import numpy as np
import cv2
from PIL import Image,ImageDraw
from tkinter import *
from tkinter import messagebox
from keras.models import load_model
import PIL
import tensorflow as tf

width = 250
height = 250
center = height//2
white = (255, 255, 255)
green = (0,128,0)
dim=(28,28)
 

image_dump="/home/vivek/Kannada_mnist/image_dump/"     #path to dump the image
model=load_model("/home/vivek/Kannada_mnist/model/Kan_mnist_model.hdf5")      #load the trained model

def paint(event):   
     
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)

def clear():
    cv.delete("all")
    draw_img()

def predict1():
    global count
    filename = image_dump+"{}".format(count)+".png"    #path to dump the image
    image1.save(filename)
    
    x=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    
    x = cv2.bitwise_not(x)

    x=cv2.resize(x,(28,28),interpolation=cv2.INTER_AREA)    
    x=np.expand_dims(x,axis=0)
    x=np.expand_dims(x,axis=3)
    ans=model.predict(x)
    ans=np.argmax(ans)
    ans=str(ans)      
    messagebox.showerror("Number in english: ",ans)
    count+=1
    
def draw_img():
    global draw
    global image1
    
    image1 = PIL.Image.new("RGB", (width, height),white)
    draw = ImageDraw.Draw(image1)
    

root = Tk()
root.title("Kannada Canvas")
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

draw_img()
count=0
cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

button=Button(text="clear",command=clear)
button.pack()

button1=Button(text="predict",command=predict1)
button1.pack()


root.mainloop()
























