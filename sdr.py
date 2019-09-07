#!/usr/bin/env python


import warnings
warnings.filterwarnings("ignore")

import sys
import matplotlib
import numpy as np
import random
import itertools
import socket
from sklearn import preprocessing

sys.path.append('./src')
sys.path.append('./src/algorithms')
sys.path.append('./src/optimizer')
sys.path.append('./src/kernels')
sys.path.append('./src/tools')

from gaussian import *
from ism import *
from linear_supv_dim_reduction import *
from kernel_lib import *
from classifiers import *

np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


class sdr():
	def __init__(self, X, Y, q=None):	#	X=data, Y=label, q=reduced dimension
		#	automated variables
		self.db = {}
		self.db['X'] = X
		self.db['Y'] = Allocation_2_Y(Y)
		self.db['N'] = N = X.shape[0]
		self.db['d'] = d = X.shape[1]
		self.db['H'] = np.eye(N) - (1.0/N)*np.ones((N, N))
		self.db['q'] = rank_by_variance(X, q)	#if q is not set, then we keep 99% of variance
		self.db['c'] = self.db['Y'].shape[1]	#c is the number of classes

		#	adjustable variables
		self.db['convergence_method'] = 'use_eigen_values'	# use_eigen_values is faster but gradient might not = 0 and use_W is slower but more accurate with gradient = 0
		self.db['algorithm'] = linear_supv_dim_reduction(self.db)
		self.db['kernel'] = gaussian(self.db)
		self.db['optimizer'] = ism(self.db)


	def __del__(self):
		del self.db['algorithm']
		del self.db['kernel']
		del self.db['optimizer']
		self.db.clear()
	
	def train(self):
		db = self.db
		Alg = db['algorithm']

		Alg.initialize_U()
		Alg.initialize_W()

		start_time = time.time() 
		while True:				# for supervised DR, this only runs once, however, it enables unsupervised extensions
			Alg.update_f()
			Alg.update_U()
			if Alg.outer_converge(): break;

		Alg.verify_result(start_time)
		
	def get_projection_matrix(self):
		return self.db['W']

	def get_reduced_dim_data(self, X):
		return X.dot(self.db['W'])
	

if __name__ == "__main__":
	X = np.loadtxt('data/wine.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('data/wine_label.csv', delimiter=',', dtype=np.int32)			
	X_test = np.loadtxt('data/wine_test.csv', delimiter=',', dtype=np.float64)			
	Y_test = np.loadtxt('data/wine_label_test.csv', delimiter=',', dtype=np.int32)			


	X = preprocessing.scale(X)


	s = sdr(X,Y)
	s.train()
	W = s.get_projection_matrix()
	Xsmall = s.get_reduced_dim_data(X)

	[out_allocation, nmi, svm_object] = use_svm(Xsmall, Y, k='rbf')
	print(nmi)

	del s
	

	#db = {}
	#fin = open(sys.argv[1],'r')
	#cmds = fin.readlines()

