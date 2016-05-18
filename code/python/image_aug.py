#Author: Tianchu Liang
#Purpose:
'''
	This script reads a directory of images and : 
		1. Save all images to another directory and among 
		the images in the new directory, randomly sampled 50
		percent will be picked and mirrored; the mirrored images
		are additional images in the directory

		2. In the new directory, all images will cropped with 
		two thirds upper portion kept. This is particularly useful
		for getting sunset information. 
'''

from PIL import Image  
from PIL import ImageOps
from PIL import ImageEnhance
import os
import sys 
import numpy as np 
import cv2

def copy(FROM_DIRECTORY,TO_DIRECTORY):
	# Use python os command here to make directory, 
	os.system('mkdir '+TO_DIRECTORY)
	# and copy images
	os.system('cp -r '+FROM_DIRECTORY+' '+TO_DIRECTORY)

def mirror(FROM_DIRECTORY,DIRECTORY):
	# Mirror each such image and save as jpg into the 
	os.system('mkdir '+DIRECTORY+'/temp')
	filenames = os.listdir(FROM_DIRECTORY)
	tempDIRECTORY=DIRECTORY+'/temp'
	for i,filename in enumerate(filenames):
		if i%100==0:
			print "Mirroring images"
		if filename.split(".")[-1]=='jpg' or filename.split(".")[-1]=='JPG' or filename.split(".")[-1]=='jpeg':
			im = Image.open(FROM_DIRECTORY+"/"+filename)
			new_im = ImageOps.mirror(im)
			new_im.save(tempDIRECTORY+'/mirrored_'+str(i)+'.jpg')
		else:
			continue
	os.system('cp -r '+DIRECTORY+'/temp/'+' '+DIRECTORY+'/'+FROM_DIRECTORY.split("/")[-1])
	os.system('rm -r '+DIRECTORY+'/temp')

def center_crop(FROM_DIRECTORY,DIRECTORY):
	# For all images, 
		# CROP each image by taking pixels away from four sides; 
		# the number of pixels is determined by 10% of the shorter side of width and height.
		# width and height
	os.system('mkdir '+DIRECTORY+'/temp')
	filenames = os.listdir(FROM_DIRECTORY)
	tempDIRECTORY=DIRECTORY+'/temp'
	for i,filename in enumerate(filenames):
		if i%100==0:
			print "center cropping images"
		if filename.split(".")[-1]=='jpg' or filename.split(".")[-1]=='JPG' or filename.split(".")[-1]=='jpeg':
			im = Image.open(FROM_DIRECTORY+"/"+filename)
			size = im.size
			border=0
			if size[0]<size[1]:
				border=int(0.1*size[0])
			else:
				border=int(0.1*size[1])
			new_im = ImageOps.crop(im,border=border)

			new_im.save(tempDIRECTORY+'/center_cropped_'+str(i)+'.jpg')
		else:
			continue
	os.system('cp -r '+DIRECTORY+'/temp/'+' '+DIRECTORY+'/'+FROM_DIRECTORY.split("/")[-1])
	os.system('rm -r '+DIRECTORY+'/temp')

def color_enhance(FROM_DIRECTORY,DIRECTORY):
	# For all images, 
		# enhance each image by a factor of 1.5
	os.system('mkdir '+DIRECTORY+'/temp')
	filenames = os.listdir(FROM_DIRECTORY)
	tempDIRECTORY=DIRECTORY+'/temp'
	for i,filename in enumerate(filenames):
		if i%100==0:
			print "enhancing images by a factor of 1.25 in color and contrast"
		if filename.split(".")[-1]=='jpg' or filename.split(".")[-1]=='JPG' or filename.split(".")[-1]=='jpeg':
			im = Image.open(FROM_DIRECTORY+"/"+filename)
			color_enh = ImageEnhance.Color(im)
			new_im = color_enh.enhance(1.25)
			contr_enh = ImageEnhance.Contrast(new_im)
			new_im = contr_enh.enhance(1.25)
			new_im.save(tempDIRECTORY+'/enhanced_'+str(i)+'.jpg')
		else:
			continue
	os.system('cp -r '+DIRECTORY+'/temp/'+' '+DIRECTORY+'/'+FROM_DIRECTORY.split("/")[-1])
	os.system('rm -r '+DIRECTORY+'/temp')	


def change_hue(FROM_DIRECTORY,DIRECTORY):
	os.system('mkdir '+DIRECTORY+'/temp')
	filenames = os.listdir(FROM_DIRECTORY)
	tempDIRECTORY=DIRECTORY+'/temp'

	for i,filename in enumerate(filenames):
		if i%100==0:
			print "Changing HUE"

		if filename.split(".")[-1]=='jpg' or filename.split(".")[-1]=='JPG' or filename.split(".")[-1]=='jpeg':
			
			im = cv2.imread(FROM_DIRECTORY+"/"+filename,1)
			new_im = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
			cv2.imwrite(tempDIRECTORY+'/hue_converted_'+str(i)+'.jpg', new_im)
		else:
			continue
	os.system('cp -r '+DIRECTORY+'/temp/'+' '+DIRECTORY+'/'+FROM_DIRECTORY.split("/")[-1])
	os.system('rm -r '+DIRECTORY+'/temp')

if __name__=="__main__":

	if len(sys.argv)>1:
		FROM_DIRECTORY=sys.argv[1]
		TO_DIRECTORY=sys.argv[2]
		copy(FROM_DIRECTORY,TO_DIRECTORY)
		mirror(FROM_DIRECTORY,TO_DIRECTORY)
		center_crop(FROM_DIRECTORY,TO_DIRECTORY)
		color_enhance(FROM_DIRECTORY,TO_DIRECTORY)
		change_hue(FROM_DIRECTORY,TO_DIRECTORY)