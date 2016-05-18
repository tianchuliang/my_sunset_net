import os
import sys 
import numpy as np 

def trim(dat):

	i = 1
	result =[]
	dat[0]=np.array([float(a) for a in dat[0]])

	result.append(dat[0])

	while i<dat.shape[0]:
		prev = int(float(dat[i-1][0]))
		cur = int(float(dat[i][0]))
		if cur!=prev:
			dat[i]=np.array([float(a) for a in dat[i]])
			result.append(dat[i])
			i=i+1
		else:
			i=i+1
	return np.array(result)
	
def parse_train(filepath):
	f = open(filepath,'r').readlines()[1:]
	f = [row.split(",") for row in f]

	iters =[int(float(row[0])) for row in f]
	loss = [float(row[-1]) for row in f]
	acc = [float(row[-2]) for row in f]

	return iters,loss,acc

def parse_test(filepath):
	f = open(filepath,'r').readlines()[1:]
	f = [row.split(",") for row in f]
	# f = [[float(a) for a in row] for row in f]

	f = trim(np.array(f))

	iters =[int(float(row[0])) for row in f]
	loss = [float(row[-2]) for row in f]
	acc = [float(row[-3]) for row in f]
	
	return iters,loss,acc



