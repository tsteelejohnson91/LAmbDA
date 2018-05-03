# Fig 5A
for cv in range(0,10):
	import tensorflow as tf
	import numpy as np
	import math
	import scipy.io as sio
	X = sio.loadmat('LAmbDA/LAmbDA_data/ZeiselLakeDarm_expr2.mat');
	X = np.array(X['expr2']);
	Y = sio.loadmat('LAmbDA/LAmbDA_data/ZeiselLakeDarm_labels.mat');
	Y = np.array(Y['y']);
	G = sio.loadmat('LAmbDA/LAmbDA_data/label_mask3.mat');
	G = np.array(G['G']);
	G = tf.cast(G, tf.float32);
	D = sio.loadmat('LAmbDA/LAmbDA_data/dat_labels.mat');
	D = np.array(D['D']);
	D = tf.cast(D, tf.float32);
	
	def get_weight(shape, lambda1):  
		var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)  
		tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))  
		return var
	
	def add_layer(input,in_size,out_size,activation_function=None,dropout_function=False,lambda1=0):
		Weights= get_weight([in_size,out_size], lambda1) 
		biases=tf.Variable(tf.zeros([1,out_size])+0.1)
		Wx_plus_b=tf.matmul(input,Weights)+biases
		if dropout_function==True:
			Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob=0.7)
		else:
			pass
		if activation_function is None:
			outputs=Wx_plus_b
		else:
			outputs=activation_function(Wx_plus_b)
		return outputs
	
	# Oversampling
	gamma = 1;
	delta = 0.6;
	tau = 1.5;
	lambda1 = 3.5;
	lambda2 = 1;
	lambda3 = 1;
	bs_prc = 0.6;
	num_samps = 5700;
	input_feats = 12797;
	hidden_feats = 720;
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
	
	train2 = np.concatenate((list([val for val in train if val not in rem]),add));
	bs = int(np.ceil(bs_prc*train2.size))
	#************************************************************
	# Building neural network with regularization term
	xs = tf.placeholder(tf.float32, [None,input_feats])
	ys = tf.placeholder(tf.float32, [None,num_labls])
	layer1=add_layer(xs,input_feats,hidden_feats,activation_function=tf.sigmoid,dropout_function=True,lambda1=lambda1)
	predict=add_layer(layer1,hidden_feats,output_feats,activation_function=tf.nn.softmax,dropout_function=False,lambda1=lambda1)
	
	#****************************************************************
	# Creating new label matrix using known labels (ys), correspondence matrix (G) and predictions (predict)
	# The predicted values are reweighted by column means using a constant tau to increase or decrease the weights
	Cm = tf.matmul(tf.transpose(tf.matmul(ys,D)),predict+0.1)/tf.reshape(tf.reduce_sum(tf.transpose(tf.matmul(ys,D)),1),(-1,1));
	mCm = tf.reshape(tf.reduce_mean(tf.cast(tf.matmul(tf.transpose(D),G)>0,tf.float32)*Cm,1),(-1,1));
	yw = tf.multiply(predict+0.1,tf.matmul(tf.matmul(ys,D),tf.pow(mCm/Cm,tau)));
	# The 70 input labels are converted to the 48 output labels using the correspondence matrix and multiplied by the reweighted predicted values
	ye = tf.multiply(tf.matmul(ys,G),yw);
	# The mean across the 48 output labels is calculated for each of the 70 input labels
	yt = tf.matmul(ys,tf.matmul(tf.transpose(ys),ye));
	# The predicted values are reweighted by the 70 input label means to keep the same labels in the same category
	ya = (delta*yt)+((1-delta)*ye)
	# The column mean reweighted and input type mean reweighted values are used to generate the final labels
	yn = tf.one_hot(tf.argmax(ya,axis=1),output_feats)
	# Subtype centroid optimization in hidden layer
	Ct = tf.transpose(tf.matmul(tf.transpose(layer1),ys))/tf.reshape(tf.reduce_sum(tf.transpose(ys),1),(-1,1));
	E = tf.multiply(tf.matmul(tf.reshape(tf.reduce_sum(tf.square(Ct),1),(-1,1)),tf.ones([1,num_labls])) + tf.matmul(tf.ones([num_labls,1]),tf.reshape(tf.reduce_sum(tf.square(Ct),1),(1,-1))) - tf.multiply(tf.cast(2,tf.float32),tf.matmul(Ct,tf.transpose(Ct))),tf.ones([num_labls,num_labls])-tf.eye(tf.cast(num_labls,tf.int32)))
	M1 = (tf.cast(tf.matmul(G,tf.transpose(G))>0,tf.float32) * (tf.ones([num_labls,num_labls])-tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32))) / (tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(G,1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0) + tf.transpose(tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(G,1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0)))
	M2 = tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32) / (tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32),1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0) + tf.transpose(tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32),1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0)))
	#***********************************************************************
	# Cost function with new label matrix
	iter = 1000;
	loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(yn-predict),reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(loss1+lambda2*tf.reduce_mean(E*M1)-lambda3*tf.reduce_mean(E*M2))
	init=tf.global_variables_initializer()
	sess=tf.Session()
	sess.run(init)
	for i in range(iter+1):
		sess.run(train_step,feed_dict={xs: X[train2[0:bs],:],ys: Y[train2[0:bs],:]})
		if i==iter:
			blah = sess.run(predict,feed_dict={xs:X[test,:],ys:Y[test,]});
			blah2 = sess.run(layer1,feed_dict={xs:X[test,:],ys:Y[test,]});
			sio.savemat('preds1_cv'+str(cv)+'.mat',{'preds':blah});
			sio.savemat('truth1_cv'+str(cv)+'.mat',{'labels':Y[test,:]});
			sio.savemat('hidden1_cv'+str(cv)+'.mat',{'hidden':blah2});
		elif i%5==0:
			print(str(sess.run(loss1,feed_dict={xs:X[train2[0:bs],:],ys:Y[train2[0:bs],:]}))+' '+str(sess.run(loss1,feed_dict={xs:X[test,:],ys:Y[test,:]}))+' '+str(sess.run(tf.reduce_mean(E*M1),feed_dict={xs:X[train2[0:bs],:],ys:Y[train2[0:bs],:]}))+' '+str(sess.run(tf.reduce_mean(E*M2),feed_dict={xs:X[train2[0:bs],:],ys:Y[train2[0:bs],:]})));
		elif i%50==0:
			train2 = tf.random_shuffle(train2);
	tf.reset_default_graph();

# Fig 5B
for cv in range(0,10):
	import tensorflow as tf
	import numpy as np
	import math
	import scipy.io as sio
	X = sio.loadmat('LAmbDA/LAmbDA_data/ZeiselLakeDarm_expr2.mat');
	X = np.array(X['expr2']);
	Y = sio.loadmat('LAmbDA/LAmbDA_data/ZeiselLakeDarm_labels.mat');
	Y = np.array(Y['y']);
	G = sio.loadmat('LAmbDA/LAmbDA_data/label_mask2.mat');
	G = np.array(G['G']);
	G = tf.cast(G, tf.float32);
	D = sio.loadmat('LAmbDA/LAmbDA_data/dat_labels.mat');
	D = np.array(D['D']);
	D = tf.cast(D, tf.float32);
	
	def get_weight(shape, lambda1):  
		var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)  
		tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))  
		return var
	
	def add_layer(input,in_size,out_size,activation_function=None,dropout_function=False,lambda1=0):
		Weights= get_weight([in_size,out_size], lambda1) 
		biases=tf.Variable(tf.zeros([1,out_size])+0.1)
		Wx_plus_b=tf.matmul(input,Weights)+biases
		if dropout_function==True:
			Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob=0.7)
		else:
			pass
		if activation_function is None:
			outputs=Wx_plus_b
		else:
			outputs=activation_function(Wx_plus_b)
		return outputs
	
	# Oversampling
	gamma = 1;
	delta = 0.6;
	tau = 1.5;
	lambda1 = 3.5;
	lambda2 = 1;
	lambda3 = 1;
	bs_prc = 0.6;
	num_samps = 5700;
	input_feats = 12797;
	hidden_feats = 720;
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
	
	train2 = np.concatenate((list([val for val in train if val not in rem]),add));
	bs = int(np.ceil(bs_prc*train2.size))
	#************************************************************
	# Building neural network with regularization term
	xs = tf.placeholder(tf.float32, [None,input_feats])
	ys = tf.placeholder(tf.float32, [None,num_labls])
	layer1=add_layer(xs,input_feats,hidden_feats,activation_function=tf.sigmoid,dropout_function=True,lambda1=lambda1)
	predict=add_layer(layer1,hidden_feats,output_feats,activation_function=tf.nn.softmax,dropout_function=False,lambda1=lambda1)
	
	#****************************************************************
	# Creating new label matrix using known labels (ys), correspondence matrix (G) and predictions (predict)
	# The predicted values are reweighted by column means using a constant tau to increase or decrease the weights
	Cm = tf.matmul(tf.transpose(tf.matmul(ys,D)),predict+0.1)/tf.reshape(tf.reduce_sum(tf.transpose(tf.matmul(ys,D)),1),(-1,1));
	mCm = tf.reshape(tf.reduce_mean(tf.cast(tf.matmul(tf.transpose(D),G)>0,tf.float32)*Cm,1),(-1,1));
	yw = tf.multiply(predict+0.1,tf.matmul(tf.matmul(ys,D),tf.pow(mCm/Cm,tau)));
	# The 70 input labels are converted to the 48 output labels using the correspondence matrix and multiplied by the reweighted predicted values
	ye = tf.multiply(tf.matmul(ys,G),yw);
	# The mean across the 48 output labels is calculated for each of the 70 input labels
	yt = tf.matmul(ys,tf.matmul(tf.transpose(ys),ye));
	# The predicted values are reweighted by the 70 input label means to keep the same labels in the same category
	ya = (delta*yt)+((1-delta)*ye)
	# The column mean reweighted and input type mean reweighted values are used to generate the final labels
	yn = tf.one_hot(tf.argmax(ya,axis=1),output_feats)
	# Subtype centroid optimization in hidden layer
	Ct = tf.transpose(tf.matmul(tf.transpose(layer1),ys))/tf.reshape(tf.reduce_sum(tf.transpose(ys),1),(-1,1));
	E = tf.multiply(tf.matmul(tf.reshape(tf.reduce_sum(tf.square(Ct),1),(-1,1)),tf.ones([1,num_labls])) + tf.matmul(tf.ones([num_labls,1]),tf.reshape(tf.reduce_sum(tf.square(Ct),1),(1,-1))) - tf.multiply(tf.cast(2,tf.float32),tf.matmul(Ct,tf.transpose(Ct))),tf.ones([num_labls,num_labls])-tf.eye(tf.cast(num_labls,tf.int32)))
	M1 = (tf.cast(tf.matmul(G,tf.transpose(G))>0,tf.float32) * (tf.ones([num_labls,num_labls])-tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32))) / (tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(G,1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0) + tf.transpose(tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(G,1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0)))
	M2 = tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32) / (tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32),1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0) + tf.transpose(tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32),1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0)))
	#***********************************************************************
	# Cost function with new label matrix
	iter = 1000;
	loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(yn-predict),reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(loss1+lambda2*tf.reduce_mean(E*M1)-lambda3*tf.reduce_mean(E*M2))
	init=tf.global_variables_initializer()
	sess=tf.Session()
	sess.run(init)
	for i in range(iter+1):
		sess.run(train_step,feed_dict={xs: X[train2[0:bs],:],ys: Y[train2[0:bs],:]})
		if i==iter:
			blah = sess.run(predict,feed_dict={xs:X[test,:],ys:Y[test,]});
			blah2 = sess.run(layer1,feed_dict={xs:X[test,:],ys:Y[test,]});
			sio.savemat('preds2_cv'+str(cv)+'.mat',{'preds':blah});
			sio.savemat('truth2_cv'+str(cv)+'.mat',{'labels':Y[test,:]});
			sio.savemat('hidden2_cv'+str(cv)+'.mat',{'hidden':blah2});
		elif i%5==0:
			print(str(sess.run(loss1,feed_dict={xs:X[train2[0:bs],:],ys:Y[train2[0:bs],:]}))+' '+str(sess.run(loss1,feed_dict={xs:X[test,:],ys:Y[test,:]}))+' '+str(sess.run(tf.reduce_mean(E*M1),feed_dict={xs:X[train2[0:bs],:],ys:Y[train2[0:bs],:]}))+' '+str(sess.run(tf.reduce_mean(E*M2),feed_dict={xs:X[train2[0:bs],:],ys:Y[train2[0:bs],:]})));
		elif i%50==0:
			train2 = tf.random_shuffle(train2);
	tf.reset_default_graph();

# Fig 5G 
for cv in range(0,10):
	import tensorflow as tf
	import numpy as np
	import math
	import scipy.io as sio
	X = sio.loadmat('LAmbDA/LAmbDA_data/ZeiselLakeDarm_expr2.mat');
	X = np.array(X['expr2']);
	Y = sio.loadmat('LAmbDA/LAmbDA_data/ZeiselLakeDarm_labels.mat');
	Y = np.array(Y['y']);
	G = sio.loadmat('LAmbDA/LAmbDA_data/label_mask3.mat');
	G = np.array(G['G']);
	G = tf.cast(G, tf.float32);
	D = sio.loadmat('LAmbDA/LAmbDA_data/dat_labels.mat');
	D = np.array(D['D']);
	D = tf.cast(D, tf.float32);
	
	def get_weight(shape, lambda1):  
		var = tf.Variable(tf.random_normal(shape), dtype=tf.float32) 
		tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))  
		return var
	
	def add_layer(input,in_size,out_size,activation_function=None,dropout_function=False,lambda1=0):
		Weights= get_weight([in_size,out_size], lambda1) 
		biases=tf.Variable(tf.zeros([1,out_size])+0.1)
		Wx_plus_b=tf.matmul(input,Weights)+biases
		if dropout_function==True:
			Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob=0.7)
		else:
			pass
		if activation_function is None:
			outputs=Wx_plus_b
		else:
			outputs=activation_function(Wx_plus_b)
		return outputs
	
	# Oversampling
	gamma = 1;
	delta = 0.6;
	tau = 1.5;
	lambda1 = 3.5;
	lambda2 = 1;
	lambda3 = 2;
	bs_prc = 0.6;						#Added new
	num_samps = 5700;
	input_feats = 12797;
	hidden_feats = 720;
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
	
	train2 = np.concatenate((list([val for val in train if val not in rem]),add));
	bs = int(np.ceil(bs_prc*train2.size))
	#************************************************************
	# Building neural network with regularization term
	xs = tf.placeholder(tf.float32, [None,input_feats])
	ys = tf.placeholder(tf.float32, [None,num_labls])
	layer1=add_layer(xs,input_feats,hidden_feats,activation_function=tf.sigmoid,dropout_function=True,lambda1=lambda1)
	predict=add_layer(layer1,hidden_feats,output_feats,activation_function=tf.nn.softmax,dropout_function=False,lambda1=lambda1)
	#****************************************************************
	# Creating new label matrix using known labels (ys), correspondence matrix (G) and predictions (predict)
	# The predicted values are reweighted by column means using a constant tau to increase or decrease the weights
	Cm = tf.matmul(tf.transpose(tf.matmul(ys,D)),predict+0.1)/tf.reshape(tf.reduce_sum(tf.transpose(tf.matmul(ys,D)),1),(-1,1));
	mCm = tf.reshape(tf.reduce_mean(tf.cast(tf.matmul(tf.transpose(D),G)>0,tf.float32)*Cm,1),(-1,1));
	yw = tf.multiply(predict+0.1,tf.matmul(tf.matmul(ys,D),tf.pow(mCm/Cm,tau)));
	# The 70 input labels are converted to the 48 output labels using the correspondence matrix and multiplied by the reweighted predicted values
	ye = tf.multiply(tf.matmul(ys,G),yw);
	# The mean across the 48 output labels is calculated for each of the 70 input labels
	yt = tf.matmul(ys,tf.matmul(tf.transpose(ys),ye));
	# The predicted values are reweighted by the 70 input label means to keep the same labels in the same category
	ya = (delta*yt)+((1-delta)*ye)
	# The column mean reweighted and input type mean reweighted values are used to generate the final labels
	yn = tf.one_hot(tf.argmax(ya,axis=1),output_feats)
	# Subtype centroid optimization in hidden layer
	Ct = tf.transpose(tf.matmul(tf.transpose(layer1),ys))/tf.reshape(tf.reduce_sum(tf.transpose(ys),1),(-1,1));
	E = tf.multiply(tf.matmul(tf.reshape(tf.reduce_sum(tf.square(Ct),1),(-1,1)),tf.ones([1,num_labls])) + tf.matmul(tf.ones([num_labls,1]),tf.reshape(tf.reduce_sum(tf.square(Ct),1),(1,-1))) - tf.multiply(tf.cast(2,tf.float32),tf.matmul(Ct,tf.transpose(Ct))),tf.ones([num_labls,num_labls])-tf.eye(tf.cast(num_labls,tf.int32)))
	M1 = (tf.cast(tf.matmul(G,tf.transpose(G))>0,tf.float32) * (tf.ones([num_labls,num_labls])-tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32))) / (tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(G,1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0) + tf.transpose(tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(G,1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0)))
	M2 = tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32) / (tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32),1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0) + tf.transpose(tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32),1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0)))
	#***********************************************************************
	# Cost function with new label matrix
	iter = 1500;
	# Initialize to unambiguous labels
	trainI = train2[np.in1d(train2,np.where(np.sum(np.matrix(Y[:,np.where(rowsums==1)]),1)==1))];
	testI = test[np.in1d(test,np.where(np.sum(np.matrix(Y[:,np.where(rowsums==1)]),1)==1))];
	lossI = tf.reduce_mean(tf.reduce_sum(tf.square(ys[:,0:48]-predict),reduction_indices=[1]))
	train_stepI = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(lossI)
	# LAmbDA optimization
	loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(yn-predict),reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(loss1+lambda2*tf.reduce_mean(E*M1)-lambda3*tf.reduce_mean(E*M2))
	init=tf.global_variables_initializer()
	sess=tf.Session()
	sess.run(init)
	for i in range(iter+1):
		if i <= 500:
			sess.run(train_stepI,feed_dict={xs: X[trainI,:],ys: Y[trainI,:]})
			if i%5==0:
				print(str(sess.run(lossI,feed_dict={xs:X[trainI,:],ys:Y[trainI,:]}))+' '+str(sess.run(lossI,feed_dict={xs:X[testI,:],ys:Y[testI,:]})));
		else:
			sess.run(train_step,feed_dict={xs: X[train2[0:bs],:],ys: Y[train2[0:bs],:]})
			if i==iter:
				blah = sess.run(predict,feed_dict={xs:X[test,:],ys:Y[test,]});
				blah2 = sess.run(layer1,feed_dict={xs:X[test,:],ys:Y[test,]});
				sio.savemat('preds3_cv'+str(cv)+'.mat',{'preds':blah});
				sio.savemat('truth3_cv'+str(cv)+'.mat',{'labels':Y[test,:]});
				sio.savemat('hidden3_cv'+str(cv)+'.mat',{'hidden':blah2});
			elif i%5==0:
				print(str(sess.run(loss1,feed_dict={xs:X[train2[0:bs],:],ys:Y[train2[0:bs],:]}))+' '+str(sess.run(loss1,feed_dict={xs:X[test,:],ys:Y[test,:]}))+' '+str(sess.run(tf.reduce_mean(E*M1),feed_dict={xs:X[train2[0:bs],:],ys:Y[train2[0:bs],:]}))+' '+str(sess.run(tf.reduce_mean(E*M2),feed_dict={xs:X[train2[0:bs],:],ys:Y[train2[0:bs],:]})));
			elif i%50==0:
				train2 = tf.random_shuffle(train2);
	tf.reset_default_graph();

