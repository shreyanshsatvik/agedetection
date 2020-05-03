# -*- coding: utf-8 -*-
"""
Created on Sun May  3 01:47:10 2020

@author: Shreyansh Satvik
"""
import numpy as np
import pandas as pd
import argparse
import cv2
import os


#Loading image file and face model
imagepath="shreyansh.jpg"
faceweights= "res10_300x300_ssd_iter_140000.caffemodel"
facecfg=     "deploy.prototxt"
facenet=cv2.dnn.readNet(facecfg,faceweights)

#defines the age bucket

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
	"(38-43)", "(48-53)", "(60-100)"]


#Loading our age model

ageweights= "age_net.caffemodel"
agecfg="age_deploy.prototxt"
agenet=cv2.dnn.readNet(agecfg,ageweights)

#load input image and construct imput blob for the image
#h--height w--width
image=cv2.imread(imagepath)
(h,w)=image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))

#passing the input as a blob in our deep neural network

facenet.setInput(blob)
detections= facenet.forward()
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the confidence is
	# greater than the minimum confidence
	if confidence > 0.5:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# extract the ROI of the face and then construct a blob from
		# *only* the face ROI
		face = image[startY:endY, startX:endX]
		faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
			(78.426, 87.76, 114.79),
			swapRB=False)

		# make predictions on the age and find the age bucket with
		# the largest corresponding probability
		agenet.setInput(faceBlob)
		preds = agenet.forward()
		i = preds[0].argmax()
		age = AGE_BUCKETS[i]
		ageConfidence = preds[0][i]

		# display the predicted age to our terminal
		text = "{}: {:.2f}%".format(age, ageConfidence * 100)
		print("[INFO] {}".format(text))

		# draw the bounding box of the face along with the associated
		# predicted age
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.imwrite('age1.jpg',image)
"""
for i in range(0,detections.shape[2]):
    #extracting the probability
    confidence=detections[0,0,i,2]
    if confidence > 0.5:
        box=detections[0,0,i,3:7]*np.array(w,h,w,h)
        (x,y,ex,ey)=box.astype("int")
        
        face=image[y:ey,x:ex]
        faceblob=cv2.dnn.blobFromImage(face, 1.0, (227, 227),(78.42, 87.76, 114.89),swapRB=False)
        #makeprediction on age
        agenet.setInput(faceblob)
        preds=agenet.forward()
        i=preds[0].argmax()
        age=AGE_BUCKETS[i]
        ageConfidence=preds[0][i]
       
		text = "{}: {:.2f}%".format(age, ageConfidence * 100)
		print("[INFO] {}".format(text))
        #draw boundingbox
        if y-10>10:
            y=y-10
        else:
            y=y+10
        cv2.rectangle(image, (x,y), (ex, ey),(0, 0, 255), 2)
		cv2.putText(image, text, (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        
    