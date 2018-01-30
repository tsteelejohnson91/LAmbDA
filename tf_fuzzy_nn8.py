#*******************************************************
# tf_fuzzy_nn: Trains using fuzzy graph cost function scRNA-Seq data
# This code was written by Yatong Han (tf initialization and training)
# and editted by Travis Johnson (loss functions and optimization)
# The loss funcitons use the graph structure encode in G to optimize the
# neural network.

import tensorflow as tf
import numpy as np
import math
import scipy.io as sio
X = sio.loadmat('ZeiselLakeDarm_expr2.mat');
X = np.array(X['expr2']);
Y = sio.loadmat('ZeiselLakeDarm_labels.mat');
Y = np.array(Y['y']);
G = sio.loadmat('label_mask3.mat');
G = np.array(G['G']);
#G= np.identity(70);
G = tf.cast(G, tf.float32);

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

for cv in range(0,10):
	# Oversampling
	gamma = 1;
	delta1 = 0.7;
	delta2 = 2.0;
	num_samps = 5700;
	input_feats = 12797;
	hidden_feats = 200;
	num_labls = 70;
	output_feats = 48;
	perm = np.random.permutation(6376);
	train = perm[0:num_samps];
	test = perm[num_samps:6376];
	add = list();
	rem = list();
	colsums = np.sum(Y[train,:],axis=0);
	rowsums = np.sum(tf.Session().run(G),axis=1);
	cutoff = math.ceil(np.percentile(colsums,90));
	for i in range(len(colsums)):
		if colsums[i] < math.ceil(cutoff*(math.log((max(rowsums)/rowsums[i])+1,2)**gamma)):
			idx = np.where(Y[train,i]);
			for j in np.random.choice(int(colsums[i]),int(math.ceil(cutoff*(math.log((max(rowsums)/rowsums[i])+1,2)**gamma))-colsums[i])):
				add.append(idx[0][j]);
		elif colsums[i] == math.ceil(cutoff*(math.log((max(rowsums)/rowsums[i])+1,2)**gamma)):
			pass
		else:
			idx = np.where(Y[train,i]);
			for j in np.random.choice(int(colsums[i]),int(colsums[i]-math.ceil(cutoff*(math.log((max(rowsums)/rowsums[i])+1,2)**gamma)))):
				rem.append(idx[0][j]);
	#print rem;
	#Y2 = np.concatenate((Y,Y[add,:]));
	#X2 = np.concatenate((X,X[add,:]));
	train2 = np.concatenate((list([val for val in train if val not in rem]),add));
	#************************************************************
	# Building neural network with regularization term
	xs = tf.placeholder(tf.float32, [None,input_feats])
	ys = tf.placeholder(tf.float32, [None,num_labls])
	layer1=add_layer(xs,input_feats,hidden_feats,activation_function=tf.sigmoid,dropout_function=True,lambda1=1.00)
	predict=add_layer(layer1,hidden_feats,output_feats,activation_function=tf.nn.softmax,dropout_function=False,lambda1=1.00)
	
	#****************************************************************
	# Creating new label matrix using known labels (ys), correspondence matrix (G) and predictions (predict)
	# The predicted values are reweighted by column means using a constant delta2 to increase or decrease the weights
	yw = tf.multiply(predict+0.1,tf.pow(tf.reduce_mean(tf.reduce_mean(predict+0.1,0))/tf.reduce_mean(predict+0.1,0),delta2));
	# The 70 input labels are converted to the 48 output labels using the correspondence matrix and multiplied by the reweighted predicted values
	ye = tf.multiply(tf.matmul(ys,G),yw);
	# The mean across the 48 output labels is calculated for each of the 70 input labels
	yt = tf.matmul(ys,tf.matmul(tf.transpose(ys),ye));
	# The predicted values are reweighted by the 70 input label means to keep the same labels in the same category
	ya = (delta1*yt)+((1-delta1)*ye)
	# The column mean reweighted and input type mean reweighted values are used to generate the final labels
	yn = tf.one_hot(tf.argmax(ya,axis=1),output_feats)
	
	#***********************************************************************
	# Cost function with new label matrix
	loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(yn-predict),reduction_indices=[1]))
	#loss2 = tf.losses.softmax_cross_entropy(onehot_labels=yn, logits=predict)			# Cross Entropy (Does not work)
	train_step2= tf.train.AdamOptimizer(0.01).minimize(loss2)
	init=tf.global_variables_initializer()
	sess=tf.Session()
	sess.run(init)
	iter = 1000;
	for i in range(iter+1):
		sess.run(train_step2,feed_dict={xs: X[train2,:],ys: Y[train2,:]})
		if i==iter:
			blah = sess.run(predict,feed_dict={xs:X[test,:],ys:Y[test,]});
			sio.savemat('preds13_cv'+str(cv)+'.mat',{'preds':blah});
			sio.savemat('truth13_cv'+str(cv)+'.mat',{'labels':Y[test,:]});
		elif i%5==0:
			print(str(sess.run(loss2,feed_dict={xs:X[train2,:],ys:Y[train2,:]}))+' '+str(sess.run(loss2,feed_dict={xs:X[test,:],ys:Y[test,:]})));

