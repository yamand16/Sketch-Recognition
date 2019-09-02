import cv2
import numpy as np
from tkinter import *
import tkinter as tk
import io
from PIL import Image
from PredictImage_class import PredictImage
from pynput.mouse import Button

canvas_width = 500
canvas_height = 500
master = Tk()
master.title("Draw something!")
w = Canvas(master, width = canvas_width, height = canvas_height)
predictor = PredictImage()

label_list = []

label_1 = tk.Label(master, text = "", fg="dark blue")
mouse_release = False

def read_file():
    """
    Read labels from file
    """
    f_read = open('labels_images.txt')
    all_data = f_read.readlines()
    [label_list.append(line.split(' ')[0]) for line in all_data]
    
def paint(event):
    python_green = "#476042"
    x1 = (event.x - 1)
    y1 = (event.y - 1)
    x2 = (event.x + 1)
    y2 = (event.y + 1)
    w.create_oval(x1, y1, x2, y2, fill = python_green, width = 5)

def convert_image_from_PIL_to_opencv(pil_image):
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)
    return opencv_image

def recognize():
    """
    Classify drawn object and print top-5 predictions
    """
    global label_1
    label_1.destroy()
    retval = w.postscript(colormode = 'color')
    img = Image.open(io.BytesIO(retval.encode('utf-8')))
    new_size = 256, 256
    img.thumbnail(new_size, Image.ANTIALIAS)
    results = predictor.Predict(convert_image_from_PIL_to_opencv(img))
    predictions = "/ "
    for i in range(0, 5):
        print(label_list[int(results.indices[0][0][i])])
        predictions = predictions + label_list[int(results.indices[0][0][i])] + " / "
    label_1 = tk.Label(master, text = predictions, fg="dark blue")
    label_1.pack(side = "top", padx = 20, pady = 20)

def deleteScreen():
    """
    Clear screen
    """
    w.delete("all")
    label_1.destroy()

def key(event):
    clicked = True

def callback(event):
    global mouse_release
    mouse_release = True

read_file()
button_clear = tk.Button(master, text = "Clear Screen", command = deleteScreen)

w.pack(expand = YES, fill=BOTH)
button_clear.pack(side = "bottom", fill = "both", expand = "yes", padx = 10, pady = 10)
w.bind("<B1-Motion>", paint)
w.bind("<Key>", key)
w.bind("<ButtonRelease-1>", callback)
message = Label(master, text = "Press and drag the mouse to draw")
message.pack(side = BOTTOM)

while True:
    master.update()
        
    if mouse_release == True:
        recognize()
        mouse_release = False