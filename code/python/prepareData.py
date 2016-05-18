import os
from os import listdir
import sys

if len(sys.argv)>1:
	MAINPATH=sys.argv[1] # Path for input and output
	if MAINPATH[-1]!="/":
		MAINPATH=MAINPATH+"/"

	INFO = MAINPATH.split("/")
	
	TYPE=INFO[-2]
	
	OUTFILENAME=TYPE+".txt"
	IGNOREFILE='.DS_Store'

	os.system("touch "+MAINPATH+OUTFILENAME)
	file_to_w = open(MAINPATH+OUTFILENAME,'w')
	count=0
	for name in listdir(MAINPATH):
		if name!=OUTFILENAME and name!=IGNOREFILE and name!='.txt':
			folderPath = MAINPATH+name
			print "Reading folder: ",folderPath, "..."
			for filename in listdir(folderPath):
				fileInfo = filename.split(".")

				if fileInfo[-1]=='DS_Store':
					continue
				else:
					count=count+1
					print "Reading file: ",filename, "..."
					label=0
					if name=='sunset':
						label=1
					else:
						label=0

					line = "data/Sunset/"+TYPE+"/"+name+"/"+filename+" "+str(label)+"\n"
	    			file_to_w.write(line)
	print "Total Number of images: ", count
	file_to_w.flush()
	# print listdir("/Users/tianchuliang/Documents/gt_acad/7616Spring16/hwsoln/hw4/data/Sunset/Train")
