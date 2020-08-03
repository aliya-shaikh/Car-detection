from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from random import randrange

root = Tk()
root.title("Face Detector")
root.geometry("1350x700+0+0")

def open():
	label.config(text = "You selected : " + var.get())
	label.pack()
	if var.get() == "Image":
		button1 = Button(root,text="Browse a file",command=imagebutton)
		button1.pack()
	if var.get() == "Video":
		button2 = Button(root, text='start' command=videobutton)
		button2.pack()
	if var.get() == "Webcam":
		button3 = Button(root,text='Start',command=webacambutton)
		button3.pack()

def imagebutton():
	imagefile = filedialog.askopenfilename(initialdir ='/',title='Select A File',filetypes=(("jpeg", "*.jpg",".png"),("All Files","*.*")))
	trained_face_data = cv2.CascadeClassifier('car_detection.xml')
	img = cv2.imread(imagefile)
	grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cars = trained_face_data.detectMultiScale(grayscaled_img)
	for (x,y,w,h) in cars:
		cv2.rectangle(img, (x,y),(x+w, y+h),(randrange(256),randrange(256),randrange(256)),2)
	cv2.imshow('Face Detector',img)
	cv2.waitKey()

def videobutton():
	trained_face_data = cv2.CascadeClassifier('car_detection.xml')
	video = cv2.VideoCapture('car.mp4')
	while True:
		successful_frame_read, frame = video.read()
		grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		cars = trained_face_data.detectMultiScale(grayscaled_img)
		for (x,y,w,h) in cars:
			cv2.rectangle(frame,(x,y),(x+w, y+h),(randrange(256),randrange(256),randrange(256)),2)
		cv2.imshow("Face Detector",frame)
		cv2.waitKey()

def webacambutton():
	trained_face_data = cv2.CascadeClassifier('car_detection.xml')
	webcam = cv2.VideoCapture(0)
	while True:
		successful_frame_read, frame = webcam.read()
		grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cars = trained_face_data.detectMultiScale(grayscaled_img)
		for (x,y,w,h) in cars:
			cv2.rectangle(frame,(x,y),(x+w, y+h),(randrange(256),randrange(256),randrange(256)),2)
		cv2.imshow('Face Detector',frame)
		key = cv2.waitKey(1)
		if key==81 or key==113:
			break


name = Label(root,text ="CHOOSE ONE")
name.pack()
var = StringVar(root)
var.set("Image")

option = OptionMenu(root, var, "Image", "Video", "Webcam")
option.pack()

button = Button(root, text="Ok",command=open)
button.pack()

label = Label(root,text="You selected")
label.pack()

root.mainloop()
