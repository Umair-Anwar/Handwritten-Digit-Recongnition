import cv2
import numpy as np
import csv
Class = 9
newImageCount=1
ExampleCount = 67

while (newImageCount <= ExampleCount):
	print newImageCount
	name = 'Nine_Image'+str(newImageCount)+'.jpg'
	newImageCount = newImageCount+1
	path = 'Dataset Images - Segmented/9_Nine/'
	file = path + name
	print file
	img = cv2.imread(file, 0)
	#Get Rows, Cols of Image
	rows,cols = img.shape

	data=[]
	data.append(name)
	#Add pixel one-by-one into data Array.
	for i in range(rows):
	    for j in range(cols):
	        k = img[i,j]
	        data.append(k)
	#Class Variable
	data.append(Class)
	#print data

	with open('Dataset.csv', 'ab') as f:
		writer = csv.writer(f)
		writer.writerow(data)