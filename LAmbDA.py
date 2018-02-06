#!/bin/python3
#******************************************************
# LAmbDA
# This file is a self-contained version of the LAmbDA
# algorithm. A user can imput their data and be returned
# a trained LAmbDA NN model. The user can also import a 
# trained LAmbDA model and return the predicted values 
# with that model.
# Travis S Johnson, Yi Wu, Yatong Han, Kun Huang

#******************************************************
# Loading requirred libraries
import tensorflow as tf
import numpy as np
import math
import scipy.io as sio
import sys

print('Loaded required libraries')
#******************************************************
# Loading command line args
function = str(sys.argv[1]);
Xfile = str(sys.argv[2]);
Yfile = str(sys.argv[3]);
if function == 'train':
	Gfile = str(sys.argv[4]);
	gamma = float(sys.argv[5]);
	delta = float(sys.argv[6]);
	tau = float(sys.argv[7]);
	lambda1 = float(sys.argv[8]);
	hidden_feats = int(sys.argv[9]);
	cutoff_prc = int(sys.argv[10]);
	iterations = int(sys.argv[11]);
	model_file = str(sys.argv[12]);
elif function == 'test':
	Yfile = str(sys.argv[3]);
	model_file = str(sys.argv[4]);
	preds_file = str(sys.argv[5]);
else:
	print('Incorrect function argument, try train or test')

print('Loaded command line args')
#******************************************************
# Importing data files
if Xfile[-3:] == 'mat':
	X = sio.loadmat(Xfile);
	X = np.array(X['X']);
elif Xfile[-3:] == 'csv':
	X = np.genfromtxt(Xfile,delimiter=',');
elif Xfile[-3:] == 'txt':
	X = np.genfromtxt(Xfile,delimiter=',');
else:
	print('Error incorrect Xfile format');

if Yfile[-3:] == 'mat':
	Y = sio.loadmat(Yfile);
	Y = np.array(Y['Y']);
elif Yfile[-3:] == 'csv':
	Y = np.genfromtxt(Yfile,delimiter=',');
elif Yfile[-3:] == 'txt':
	Y = np.genfromtxt(Yfile,delimiter=',');
else:
	print('Error incorrect Yfile format')

if function == 'train':	
	if Gfile[-3:] == 'mat':
		G = sio.loadmat(Gfile);
		G = np.array(G['G']);
		G = tf.cast(G, tf.float32);
	elif Gfile[-3:] == 'csv':
		G = np.genfromtxt(Gfile,delimiter=',');
	elif Gfile[-3:] == 'txt':
		G = np.genfromtxt(Gfile,delimiter=',');
	else:
		print('Error incorrect Gfile format')

print('Loaded data files')
#******************************************************
# Functions for adding layers
def get_weight(shape, lambda1):  
	var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)  
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))  
	return var

def add_layer(input,in_size,out_size,activation_function=None,dropout_function=False,lambda1=0):
	Weights= get_weight([in_size,out_size], lambda1) 
	biases=tf.Variable(tf.zeros([1,out_size])+0.1)
	Wx_plus_b=tf.matmul(input,Weights)+biases
	if dropout_function==True:
		Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob=0.5)
	else:
		pass
	if activation_function is None:
		outputs=Wx_plus_b
	else:
		outputs=activation_function(Wx_plus_b)
	return outputs

print('Created required functions')

if function == 'train':
	#******************************************************
	# Oversampling
	num_samps = int(X.shape[0]);
	input_feats = int(X.shape[1]);
	num_labls = int(G.shape[0]);
	output_feats = int(G.shape[1]);
	add = list();
	rem = list();
	colsums = np.sum(Y,axis=0);
	rowsums = np.sum(tf.Session().run(G),axis=1);
	cutoff = math.ceil(np.percentile(colsums,cutoff_prc));
	for i in range(len(colsums)):
		if colsums[i] < math.ceil(cutoff*(math.log((max(rowsums)/rowsums[i])+1,2)**gamma)):
			idx = np.where(Y[:,i]);
			for j in np.random.choice(int(colsums[i]),int(math.ceil(cutoff*(math.log((max(rowsums)/rowsums[i])+1,2)**gamma))-colsums[i])):
				add.append(idx[0][j]);
		elif colsums[i] == math.ceil(cutoff*(math.log((max(rowsums)/rowsums[i])+1,2)**gamma)):
			pass
		else:
			idx = np.where(Y[:,i]);
			for j in np.random.choice(int(colsums[i]),int(colsums[i]-math.ceil(cutoff*(math.log((max(rowsums)/rowsums[i])+1,2)**gamma)))):
				rem.append(idx[0][j]);
	
	train = np.concatenate((list([val for val in range(num_samps) if val not in rem]),add));
	print('Finished resampling')
	#************************************************************
	# Building neural network with regularization term and first layer dropout
	xs = tf.placeholder(tf.float32, [None,input_feats])
	ys = tf.placeholder(tf.float32, [None,num_labls])
	layer1=add_layer(xs,input_feats,hidden_feats,activation_function=tf.sigmoid,dropout_function=True,lambda1=lambda1)
	fx=add_layer(layer1,hidden_feats,output_feats,activation_function=tf.nn.softmax,dropout_function=False,lambda1=lambda1)
	print('Added layers to network')
	#****************************************************************
	# Creating new label matrix using known labels (ys), correspondence matrix (G) and predictions (predict)
	# The predicted values are reweighted by column means using a constant tau to increase or decrease the weights
	fxd = tf.multiply(fx+0.1,tf.pow(tf.reduce_mean(tf.reduce_mean(fx+0.1,0))/tf.reduce_mean(fx+0.1,0),tau));
	# The input labels are converted to the output labels using the correspondence matrix and multiplied by the reweighted predicted values
	fxmd = tf.multiply(tf.matmul(ys,G),fxd);
	# The mean across the output labels is calculated for each of the input labels
	fxs = tf.matmul(ys,tf.matmul(tf.transpose(ys),fxmd));
	# The predicted values are reweighted by the input label means to keep the same input labels in the same output label
	fxsmd = (delta*fxs)+((1-delta)*fxmd)
	# The column mean reweighted and input type mean reweighted values are used to generate the final labels
	yhat = tf.one_hot(tf.argmax(fxsmd,axis=1),output_feats)
	print('Generated new labels')
	#***********************************************************************
	# Cost function with new label matrix
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(yhat-fx),reduction_indices=[1]))
	train_step= tf.train.AdamOptimizer(0.01).minimize(loss)
	init=tf.global_variables_initializer()
	saver = tf.train.Saver()
	sess=tf.Session()
	sess.run(init)
	print('Starting optimization')
	for i in range(iterations+1):
		sess.run(train_step,feed_dict={xs: X[train,:],ys: Y[train,:]})
		if i%5==0:
			print(str(sess.run(loss,feed_dict={xs:X[train,:],ys:Y[train,:]})));
	saver.save(sess,model_file);

if function == 'test':
	saver = tf.train.Saver();
	saver.restore(sess, model_file);
	preds = sess.run(fx,feed_dict={xs:X,ys:Y});
	if preds_file[end-3:end] == 'mat':
		sio.savemat(preds_file,{'preds':preds});
	elif preds_file[end-3:end] == 'csv':
		np.savetxt(preds_file,preds,delimiter=',');
	elif Xfile[end-3:end] == 'txt':
		np.savetxt(preds_file,preds,delimiter='\t');
	else:
		print('Error incorrect preds_file format');



