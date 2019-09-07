#!/usr/bin/env python3

from algorithm import *
import kernel_lib as klib

#import sys
#import debug

#	max Tr[(DKuD)Kx]
#	W -> Kx -> D -> γ -> Σ ψA_i,j -> W
class linear_supv_dim_reduction(algorithm):
	def __init__(self, db):
		self.db = db
		db['Γ'] = db['H'].dot(db['Y']).dot(db['Y'].T).dot(db['H'])
		db['compute_cost'] = self.compute_HSIC
		db['compute_gradient'] = self.compute_gradient


	def __del__(self):
		pass	

	def initialize_U(self):
		pass

	def initialize_W(self):
		init_HSIC = self.compute_HSIC(np.eye(13))

		db = self.db
		Φ0 = self.db['kernel'].get_Φ0()
		[db['W'], eigs] = klib.eig_solver(Φ0, db['q'])

	def update_U(self):
		pass

	def update_f(self):
		self.db['W'] = self.db['optimizer'].run(self.db['W'])

	def outer_converge(self):
		return True

	def verify_result(self, start_time):
		pass


	def compute_HSIC(self, W):
		db = self.db
		Kx = klib.rbk_sklearn(db['X'].dot(W), db['kernel'].σ)
		HSIC_val = np.sum(db['Γ']*Kx)
		#print('HSIC : %.4f'%HSIC_val)
		return HSIC_val

	def compute_gradient(self, W, Λ, Φ=None):
		db = self.db 

		σ = db['kernel'].σ
		if Φ is None: Φ = db['kernel'].get_Φ(W)
		gradient = Φ.dot(W) - W.dot(np.diag(Λ))
		#print(gradient)
		#print('Gradient : %.4f'%gradient)
		return gradient			


		return g	


#	def compute_Lagrangian_gradient(self):
#		Φ = self.compute_Φ()
#		[new_W, W_λ] = eig_solver(Φ, db['q'], mode='smallest')
#		gradient = Φ.dot(db['W']) - db['W'].dot(np.diag(W_λ))
#		print('Gradient :\n')
#		print(gradient)
#		return gradient			
#
#
#	def verify_result(self, start_time):
#		db = self.db	
#		if 'ignore_verification' in db: return
#
#		final_cost = self.compute_cost()
#
#		db['data'].load_validation()
#		outstr = '\nExperiment : linear supervised dimensionality reduction : %s, final cost : %.3f\n'%(db['data_name'],final_cost)
#
#		Y = db['data'].Y
#		X = db['data'].X
#
#		X_valid = db['data'].X_valid
#		Y_valid = db['data'].Y_valid
#
#		outstr += self.verification_basic_info(start_time)
#			
#		[out_allocation, nmi, svm_time, svmO] = use_svm(X,Y)
#		acc = accuracy_score(Y, out_allocation)
#		outstr += '\t\tTraining SVM NMI without dimension reduction : %.3f, acc : %.3f, time : %.4f'%(nmi, acc, svm_time) + '\n'
#
#		
#		[out_allocation, nmi, svm_time, svm_object] = use_svm(X,Y, W=db['W'])
#		acc = accuracy_score(Y, out_allocation)
#		outstr += '\t\tTraining SVM NMI with dimension reduction : %.3f , acc : %.3f'%(nmi, acc) + '\n'
#
#		if db['separate_data_for_validation']:
#			[out_allocation, nmi, svm_time] = predict_with_svm(svmO, X_valid, Y_valid)
#			acc = accuracy_score(Y_valid, out_allocation)
#			outstr += '\t\tTest Set SVM NMI without dimension reduction : %.3f, acc : %.3f, time : %.4f'%(nmi, acc, svm_time) + '\n'
#	
#			[out_allocation, nmi_valid, svm_time] = predict_with_svm(svm_object, X_valid, Y_valid, db['W'])
#			acc_valid = accuracy_score(Y_valid, out_allocation)
#			outstr += '\t\tTest Set SVM NMI with dimension reduction : %.3f, acc : %.3f '%(nmi_valid, acc_valid) + '\n'
#
#
#
#		#	relative kernel	
#		#Kx = rbk_relative_σ(db, X_valid.dot(db['W']))
#		#Kx = rbk_sklearn(X_valid.dot(db['W']), db['data'].σ)
#		#Kx = rbk_sklearn(X_valid.dot(db['W']), 1)
#		#[out_allocation, nmi, svm_time, svm_object] = use_svm(X_valid,Y_valid)
#		#acc = accuracy_score(Y_valid, out_allocation)
#		#outstr += '\t\tTraining SVM NMI without dimension reduction : %.3f, acc : %.3f, time : %.4f'%(nmi, acc, svm_time) + '\n'
#		##[out_allocation, nmi, svm_time, svm_object] = use_svm(X_valid,Y_valid, W=db['W'], k='precomputed',K=Kx)
#		#[out_allocation, nmi, svm_time, svm_object] = use_svm(X_valid,Y_valid, W=db['W'])
#		#acc = accuracy_score(Y, out_allocation)
#		#outstr += '\t\tTraining SVM NMI with dimension reduction : %.3f , acc : %.3f'%(nmi, acc) + '\n'
#
#
#		print(db['inner_convergence_cost_list'])
#
#
#		start_time = time.time() 
#		clf = LinearDiscriminantAnalysis(n_components=db['q'])
#		clf.fit(X, Y)
#		lda_labels = clf.predict(X)
#		lda_time = time.time() - start_time
#		nmi = normalized_mutual_info_score(lda_labels, Y)
#		acc = accuracy_score(Y, lda_labels)
#
#		outstr += '\tLDA\n'
#		outstr += '\t\tTraining NMI with LDA : %.3f, acc : %.3f'%(nmi, acc) + '\n'
#		outstr += '\t\tLDA Run time : %.3f'%lda_time + '\n'
#
#		start_time = time.time() 
#		pca = PCA(n_components=db['q'])
#		Xpca1 = pca.fit_transform(X)
#		Xpca = pca.transform(X_valid)
#		pca_time = time.time() - start_time
#
#
#		[out_allocation, nmi, svm_time, svm_object] = use_svm(Xpca1,Y)
#		acc = accuracy_score(Y, out_allocation)
#		outstr += '\tPCA\n'
#		outstr += '\t\tTraining SVM NMI with PCA dimension reduction : %.3f, acc : %.3f'%(nmi, acc) + '\n'
#
#		if db['separate_data_for_validation']:
#			[out_allocation, nmi, svm_time] = predict_with_svm(svm_object, Xpca, Y_valid)
#			acc = accuracy_score(Y_valid, out_allocation)
#			outstr += '\t\tTest Set SVM NMI with PCA dimension reduction : %.3f, acc : %.3f'%(nmi, acc) + '\n'
#
#		outstr += '\t\tPCA training time : %.3f'%pca_time + '\n\n\n\n\n'
#		if db['separate_data_for_validation']:
#			outstr += 'NMI : %.3f\n'%nmi_valid
#			outstr += 'ACC : %.3f\n'%acc_valid
#			outstr += 'TIME : %.5f\n'%db['run_time']
#			outstr += 'COST : %.3f\n'%final_cost
#
#		print(outstr)
#
#		fin = open('./results/LSDR_' + db['data_name']  + '_' + db['kernel_type'] + '_' +  db['W_optimize_technique'].__name__ + '.txt', 'w') 
#		fin.write(outstr)
#		fin.close()
#
