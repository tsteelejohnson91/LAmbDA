import tensorflow as tf
import numpy as np
import math
import scipy.io as sio
import optunity as opt

def get_weight(shape, lambda1): 
	var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))  
	return var

def add_layer(input,in_size,out_size,activation_function=None,dropout_function=False,lambda1=0, keep_prob1=1):
	#print([in_size,out_size])											# Testing
	Weights= get_weight([in_size,out_size], lambda1) 
	biases=tf.Variable(tf.zeros([1,out_size])+0.1)
	Wx_plus_b=tf.matmul(input,Weights)+biases
	if dropout_function==True:
		Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob=keep_prob1)
	else:
		pass
	if activation_function is None:
		outputs=Wx_plus_b
	else:
		outputs=activation_function(Wx_plus_b)
	return outputs

def wt_cutoff(colnum,cutoff,Gtmp,gamma):
	rowsums = np.sum(Gtmp,axis=1);
	return(math.ceil(cutoff*(math.log((max(rowsums)/rowsums[colnum])+1,2)**gamma)))

def resample(prc_cut,Y,Gtmp,train,gamma):
	add = list()
	rem = list()
	colsums = np.sum(Y[train,:],axis=0);
	cutoff = math.ceil(np.percentile(colsums,prc_cut));
	for i in range(len(colsums)):
		if colsums[i] == 0:
			pass
		elif colsums[i] < wt_cutoff(i,cutoff,Gtmp,gamma):
			idx = np.squeeze(np.array(np.where(Y[train,i]>=1)));
			choice = np.random.choice(train[idx],int(wt_cutoff(i,cutoff,Gtmp,gamma)-colsums[i]))
			add = add + choice.tolist();
		elif colsums[i] == wt_cutoff(i,cutoff,Gtmp,gamma):
			pass
		else:
			idx = np.squeeze(np.array(np.where(Y[train,i]>=1)));
			choice = np.random.choice(train[idx],int(colsums[i]-wt_cutoff(i,cutoff,Gtmp,gamma)),replace=False)
			rem = rem + choice.tolist()
	return np.concatenate((list([val for val in train if val not in rem]),add));

def select_feats(Xtmp,num_zero_prc_cut,var_prc_cut):
	#*********************************************************************
	# remove features with many zeros
	num_feat_zeros = np.sum(Xtmp==0,axis=1);
	Xtmp = Xtmp[num_feat_zeros<num_zero_prc_cut*Xtmp.shape[1],:]
	#*********************************************************************
	# remove features with low variance
	feat_vars = np.var(Xtmp,axis=1)
	Xtmp = Xtmp[feat_vars>np.percentile(feat_vars,var_prc_cut),:]
	return(Xtmp)

def get_yn(predict,ys,delta,tau,output_feats):
	D = tf.cast(Dnp, tf.float32);
	G = tf.cast(Gnp, tf.float32);
	ys = tf.cast(ys, tf.float32);
	#print("start")
	Cm = tf.matmul(tf.transpose(tf.matmul(ys,D)),predict+0.1)/tf.reshape(tf.reduce_sum(tf.transpose(tf.matmul(ys,D)),1),(-1,1));
	#print("1")
	mCm = tf.reshape(tf.reduce_mean(tf.cast(tf.matmul(tf.transpose(D),G)>0,tf.float32)*Cm,1),(-1,1));
	#print("2")
	yw = tf.multiply(predict+0.1,tf.matmul(tf.matmul(ys,D),tf.pow(mCm/Cm,tau)));
	#print("3")
	ye = tf.multiply(tf.matmul(ys,G),yw);
	#print("4")
	yt = tf.matmul(ys,tf.matmul(tf.transpose(ys),ye));
	#print("5")
	ya = (delta*yt)+((1-delta)*ye)
	#print("6")
	yn = tf.cast(tf.one_hot(tf.argmax(ya,axis=1),output_feats), dtype=tf.float32)
	#print("7")
	return(yn)

def get_Eucl(layer1,ys,num_labls):
	Ct = tf.transpose(tf.matmul(tf.transpose(layer1),ys))/tf.reshape(tf.reduce_sum(tf.transpose(ys+0.01),1),(-1,1));
	E = tf.multiply(tf.matmul(tf.reshape(tf.reduce_sum(tf.square(Ct),1),(-1,1)),tf.ones([1,num_labls])) + tf.matmul(tf.ones([num_labls,1]),tf.reshape(tf.reduce_sum(tf.square(Ct),1),(1,-1))) - tf.multiply(tf.cast(2,tf.float32),tf.matmul(Ct,tf.transpose(Ct))),tf.ones([num_labls,num_labls])-tf.eye(tf.cast(num_labls,tf.int32)))
	return(E)

def run_LAmbDA(gamma, delta, tau, prc_cut, bs_prc, do_prc, hidden_feats, lambda1, lambda2, lambda3):
	print("gamma=%.4f, delta=%.4f, tau=%.4f, prc_cut=%.4f, bs_prc=%.4f, do_prc=%.4f, hidden_feats=%.4f, lambda1= %.4f, lambda2= %.4f, lambda3= %.4f" % (gamma, delta, tau, prc_cut, bs_prc, do_prc, hidden_feats, lambda1, lambda2, lambda3))
	global X, Y, Gnp, Dnp, train, test, prt
	D = tf.cast(Dnp, tf.float32);
	G = tf.cast(Gnp, tf.float32);
	hidden_feats = int(math.floor(hidden_feats))
	input_feats = X.shape[1];
	num_labls = G.shape.as_list();
	output_feats = num_labls[1];
	num_labls = num_labls[0];
	rowsums = np.sum(Gnp,axis=1);
	#print(G.shape)
	train2 = resample(prc_cut, Y, Gnp, train, gamma);
	bs = int(np.ceil(bs_prc*train2.size))
	xs = tf.placeholder(tf.float32, [None,input_feats])
	ys = tf.placeholder(tf.float32, [None,num_labls])
	#print(input_feats)													# Testing
	#print(hidden_feats)													# Testing
	layer1_1=add_layer(xs,input_feats,hidden_feats,activation_function=tf.sigmoid,dropout_function=True,lambda1=lambda1, keep_prob1=do_prc)
	predict_1=add_layer(layer1_1,hidden_feats,output_feats,activation_function=tf.nn.softmax,dropout_function=False,lambda1=lambda1)
	layer1_2=add_layer(xs,input_feats,hidden_feats,activation_function=tf.sigmoid,dropout_function=True,lambda1=lambda1, keep_prob1=do_prc)
	predict_2=add_layer(layer1_2,hidden_feats,output_feats,activation_function=tf.nn.softmax,dropout_function=False,lambda1=lambda1)
	layer1_3=add_layer(xs,input_feats,hidden_feats,activation_function=tf.sigmoid,dropout_function=True,lambda1=lambda1, keep_prob1=do_prc)
	predict_3=add_layer(layer1_3,hidden_feats,output_feats,activation_function=tf.nn.softmax,dropout_function=False,lambda1=lambda1)
	layer1_4=add_layer(xs,input_feats,hidden_feats,activation_function=tf.sigmoid,dropout_function=True,lambda1=lambda1, keep_prob1=do_prc)
	predict_4=add_layer(layer1_4,hidden_feats,output_feats,activation_function=tf.nn.softmax,dropout_function=False,lambda1=lambda1)
	layer1_5=add_layer(xs,input_feats,hidden_feats,activation_function=tf.sigmoid,dropout_function=True,lambda1=lambda1, keep_prob1=do_prc)
	predict_5=add_layer(layer1_5,hidden_feats,output_feats,activation_function=tf.nn.softmax,dropout_function=False,lambda1=lambda1)
	yn_1 = get_yn(predict_1, ys,delta,tau,output_feats)
	yn_2 = get_yn(predict_2, ys,delta,tau,output_feats)
	yn_3 = get_yn(predict_3, ys,delta,tau,output_feats)
	yn_4 = get_yn(predict_4, ys,delta,tau,output_feats)
	yn_5 = get_yn(predict_5, ys,delta,tau,output_feats)
	E_1 = get_Eucl(layer1_1,ys,num_labls)
	E_2 = get_Eucl(layer1_2,ys,num_labls)
	E_3 = get_Eucl(layer1_3,ys,num_labls)
	E_4 = get_Eucl(layer1_4,ys,num_labls)
	E_5 = get_Eucl(layer1_5,ys,num_labls)
	M1 = (tf.cast(tf.matmul(G,tf.transpose(G))>0,tf.float32) * (tf.ones([num_labls,num_labls])-tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32))) / (tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(G,1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0) + tf.transpose(tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(G,1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0)))
	M2 = tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32) / (tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32),1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0) + tf.transpose(tf.matrix_band_part(tf.matmul(tf.reshape(tf.reduce_sum(tf.cast(tf.matmul(D,tf.transpose(D))>0,tf.float32),1),(-1,1)),tf.reshape(tf.ones([1,num_labls]),(1,-1))),-1,0)))
	#***********************************************************************
	# Cost function with new label matrix
	iter = 2000;
	# Initialize to unambiguous labels
	G2 = np.copy(Gnp)
	G2[rowsums>1,:] = 0;
	YI = np.matmul(Y,G2);
	#print(YI)
	YIrs = np.sum(YI,axis=1);
	print(YIrs.shape)
	G2 = tf.cast(G2, tf.float32);
	yi = tf.matmul(ys,G2);
	trainI = train2[np.in1d(train2,np.where(YIrs==1))];
	testI = test[np.in1d(test,np.where(YIrs==1))];
	lossI_1 = tf.reduce_mean(tf.reduce_sum(tf.square(yi-predict_1),reduction_indices=[1]))
	lossI_2 = tf.reduce_mean(tf.reduce_sum(tf.square(yi-predict_2),reduction_indices=[1]))
	lossI_3 = tf.reduce_mean(tf.reduce_sum(tf.square(yi-predict_3),reduction_indices=[1]))
	lossI_4 = tf.reduce_mean(tf.reduce_sum(tf.square(yi-predict_4),reduction_indices=[1]))
	lossI_5 = tf.reduce_mean(tf.reduce_sum(tf.square(yi-predict_5),reduction_indices=[1]))
	lossI_All = tf.reduce_mean(tf.reduce_sum(tf.square(yi-((predict_1+predict_2+predict_3+predict_4+predict_5)/5)),reduction_indices=[1]))
	train_stepI_1 = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(lossI_1)
	train_stepI_2 = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(lossI_2)
	train_stepI_3 = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(lossI_3)
	train_stepI_4 = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(lossI_4)
	train_stepI_5 = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(lossI_5)
	# LAmbDA optimization
	loss1_1 = tf.reduce_mean(tf.reduce_sum(tf.square(yn_1-predict_1),reduction_indices=[1]))
	loss1_2 = tf.reduce_mean(tf.reduce_sum(tf.square(yn_2-predict_2),reduction_indices=[1]))
	loss1_3 = tf.reduce_mean(tf.reduce_sum(tf.square(yn_3-predict_3),reduction_indices=[1]))
	loss1_4 = tf.reduce_mean(tf.reduce_sum(tf.square(yn_4-predict_4),reduction_indices=[1]))
	loss1_5 = tf.reduce_mean(tf.reduce_sum(tf.square(yn_5-predict_5),reduction_indices=[1]))
	loss1_All = tf.reduce_mean(tf.reduce_sum(tf.square(((yn_1+yn_2+yn_3+yn_4+yn_5)/5)-((predict_1+predict_2+predict_3+predict_4+predict_5)/5)),reduction_indices=[1]))
	train_step_1 = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(loss1_1+lambda2*tf.reduce_mean(E_1*M1)-lambda3*tf.reduce_mean(E_1*M2))
	train_step_2 = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(loss1_2+lambda2*tf.reduce_mean(E_2*M1)-lambda3*tf.reduce_mean(E_2*M2))
	train_step_3 = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(loss1_3+lambda2*tf.reduce_mean(E_3*M1)-lambda3*tf.reduce_mean(E_3*M2))
	train_step_4 = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(loss1_4+lambda2*tf.reduce_mean(E_4*M1)-lambda3*tf.reduce_mean(E_4*M2))
	train_step_5 = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(loss1_5+lambda2*tf.reduce_mean(E_5*M1)-lambda3*tf.reduce_mean(E_5*M2))
	train_step_All = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(loss1_All+lambda2*tf.reduce_mean(((E_1+E_2+E_3+E_4+E_5)/5)*M1)-lambda3*tf.reduce_mean(((E_1+E_2+E_3+E_4+E_5)/5)*M2))		#******Testing********
	#train_step = tf.train.AdamOptimizer(learning_rate=0.01,epsilon=1e-8).minimize(loss1+(lambda2*(tf.reduce_sum(E*M1)/tf.cast(tf.count_nonzero(M1>0),tf.float32)))-(lambda3*(tf.reduce_sum(E*M2)/tf.cast(tf.count_nonzero(M2>0),tf.float32))))
	init=tf.global_variables_initializer()
	trainI.astype(int)
	tensor_trainI = {xs: X[trainI, :], ys: Y[trainI, :]}
	tensor_testI = {xs: X[testI, :], ys: Y[testI, :]}
	tensor_train = {xs: X[train2[0:bs], :], ys: Y[train2[0:bs], :]}
	tensor_test = {xs: X[test, :], ys: Y[test, :]}
	# run training process
	sess=tf.Session()
	sess.run(init)
	for i in range(iter + 1):
		if i <= 1000:					# changed
			sess.run(train_stepI_1, feed_dict=tensor_trainI)
			sess.run(train_stepI_2, feed_dict=tensor_trainI)
			sess.run(train_stepI_3, feed_dict=tensor_trainI)
			sess.run(train_stepI_4, feed_dict=tensor_trainI)
			sess.run(train_stepI_5, feed_dict=tensor_trainI)
			if i % 10 == 0:
				print(str(sess.run(lossI_All, feed_dict=tensor_trainI)) + ' ' + str(sess.run(lossI_All, feed_dict=tensor_testI)));
		else:
			sess.run(train_step_1, feed_dict=tensor_train)
			sess.run(train_step_2, feed_dict=tensor_train)
			sess.run(train_step_3, feed_dict=tensor_train)
			sess.run(train_step_4, feed_dict=tensor_train)
			sess.run(train_step_5, feed_dict=tensor_train)
			#sess.run(train_step_All, feed_dict=tensor_train)		#************Testing********************
			if i % 10 == 0:
				print(str(sess.run(loss1_All, feed_dict=tensor_train)) + ' ' + str(sess.run(loss1_All, feed_dict=tensor_test)) + ' ' + str(lambda2*sess.run(tf.reduce_mean(((E_1+E_2+E_3+E_4+E_5)/5) * M1), feed_dict=tensor_train)) + ' ' + str(lambda3*sess.run(tf.reduce_mean(((E_1+E_2+E_3+E_4+E_5)/5) * M2), feed_dict=tensor_train)));
			elif i % 50 == 0:
				np.random_shuffle(train2);
				tensor_train = {xs: X[train2[0:bs], :], ys: Y[train2[0:bs], :]}
	if prt:
		blah = sess.run((predict_1+predict_2+predict_3+predict_4+predict_5)/5, feed_dict=tensor_test);
		blah2_1 = sess.run(layer1_1, feed_dict=tensor_test);
		blah2_2 = sess.run(layer1_1, feed_dict=tensor_test);
		blah2_3 = sess.run(layer1_1, feed_dict=tensor_test);
		blah2_4 = sess.run(layer1_1, feed_dict=tensor_test);
		blah2_5 = sess.run(layer1_1, feed_dict=tensor_test);
		sio.savemat('preds_cv' + str(cv) + '.mat', {'preds': blah});
		sio.savemat('truth_cv' + str(cv) + '.mat', {'labels': Y[test, :]});
		sio.savemat('hidden_cv' + str(cv) + '.mat', {'hidden1': blah2_1, 'hidden2': blah2_2, 'hidden3': blah2_3, 'hidden4': blah2_4, 'hidden5': blah2_5});
	print("loss1_All=%.4f, gamma=%.4f, delta=%.4f, tau=%.4f, prc_cut=%.4f, bs_prc=%.4f, do_prc=%.4f, hidden_feats=%.4f, lambda1= %.4f, lambda2= %.4f, lambda3= %.4f" % ( sess.run(loss1_All, feed_dict=tensor_test), gamma, delta, tau, prc_cut, bs_prc, do_prc, hidden_feats, lambda1, lambda2, lambda3))
	acc = sess.run(loss1_All, feed_dict=tensor_test) 
	tf.reset_default_graph();
	return(acc)



# Running CV
for i in range(1,11):
	global X, Y, Gnp, Dnp, train, test, prt, cv
	cv = i;
	print('Cross validation step: '+str(cv))
	
	#Simulated hold 1 splat2
	X = sio.loadmat('LAmbDA_CV_revision/splat2_hold1_expr.mat');
	X = np.array(X['X']);
	Y = sio.loadmat('LAmbDA_CV_revision/splat2_hold1_labs.mat');
	Y = np.array(Y['Y']);
	G = sio.loadmat('LAmbDA_CV_revision/splat2_hold1_mask.mat');
	Gnp = np.array(G['G']);
	D = sio.loadmat('LAmbDA_CV_revision/splat2_hold1_dat.mat');
	Dnp = np.array(D['D']);
	
	'''
	#Simulated hold 2
	X = sio.loadmat('LAmbDA_CV_revision/splat_hold2_expr.mat');
	X = np.array(X['X']);
	Y = sio.loadmat('LAmbDA_CV_revision/splat_hold2_labs.mat');
	Y = np.array(Y['Y']);
	G = sio.loadmat('LAmbDA_CV_revision/splat_hold2_mask.mat');
	Gnp = np.array(G['G']);
	D = sio.loadmat('LAmbDA_CV_revision/splat_hold2_dat.mat');
	Dnp = np.array(D['D']);
	'''
	'''
	# brain
	X = sio.loadmat('LAmbDA/LAmbDA_data/ZeiselLakeDarm_cpmtpm2.mat');
	X = np.array(X['X']);
	Y = sio.loadmat('LAmbDA/LAmbDA_data/ZeiselLakeDarm_labels.mat');
	Y = np.array(Y['Y']);
	G = sio.loadmat('LAmbDA/LAmbDA_data/label_mask6.mat');
	Gnp = np.array(G['G']);
	D = sio.loadmat('LAmbDA/LAmbDA_data/ZeiselLakeDarm_dset.mat');
	Dnp = np.array(D['D']);
	'''
	'''
	# pancreas
	X = sio.loadmat('/home/johnstrs/Desktop/pancreas/pancreasXexpr.mat');
	X = np.array(X['X']);
	Y = sio.loadmat('/home/johnstrs/Desktop/pancreas/pancreasYlabels.mat');
	Y = np.array(Y['Y']);
	G = sio.loadmat('/home/johnstrs/Desktop/pancreas/pancreasGmask2.mat');
	Gnp = np.array(G['G']);
	D = sio.loadmat('/home/johnstrs/Desktop/pancreas/pancreasDdset.mat');
	Dnp = np.array(D['D']);
	'''
	'''
	# glioma
	X = np.loadtxt('Glioma_LAmbDA/data/gliomaXexpr2.csv',delimiter=',')
	#X = sio.loadmat('Glioma_LAmbDA/data/gliomaXexpr.mat');
	#X = np.array(X['X']);
	Y = sio.loadmat('Glioma_LAmbDA/data/gliomaYlabels.mat');
	Y = np.array(Y['Y']);
	G = sio.loadmat('Glioma_LAmbDA/data/gliomaGmask.mat');
	Gnp = np.array(G['G']);
	D = sio.loadmat('Glioma_LAmbDA/data/gliomaDdset.mat');
	Dnp = np.array(D['D']);
	'''
	#X = np.log2(X/10+1);
	X = np.log2(np.transpose(select_feats(np.transpose(X),0.5,80))/10+1);
	train_samp = int(np.floor(0.6*X.shape[0]))
	test_samp = int(np.floor(0.2*X.shape[0]))
	val_samp = int(X.shape[0] - (train_samp+test_samp))
	perm = np.random.permutation(X.shape[0]);
	train = perm[0:train_samp+1];
	test = perm[train_samp+1:train_samp+test_samp+1];
	val = perm[train_samp+test_samp+1:train_samp+test_samp+val_samp+1];
	while(np.sum(np.sum(Y[train,:],0)<5)>0):		# less than 5 for brain
		perm = np.random.permutation(X.shape[0]);
		train = perm[0:train_samp+1];
		test = perm[train_samp+1:train_samp+test_samp+1];
		val = perm[train_samp+test_samp+1:train_samp+test_samp+val_samp+1];
	prt = False
	opt_params = None
	opt_params, _, _ = opt.minimize(run_LAmbDA,solver_name='sobol', gamma=[0.8,1.2], delta=[0.05,0.9], tau=[1.0,2.0], prc_cut=[20,50], bs_prc=[0.2,0.6], do_prc=[0.5,1], hidden_feats=[50,100], lambda1=[0,5], lambda2=[3,5], lambda3=[3,5], num_evals=50)
	prt = True
	train = perm[0:train_samp+test_samp+1]
	test = val
	err = run_LAmbDA(opt_params['gamma'], opt_params['delta'], opt_params['tau'], opt_params['prc_cut'], opt_params['bs_prc'], opt_params['do_prc'], opt_params['hidden_feats'], opt_params['lambda1'], opt_params['lambda2'], opt_params['lambda3'])
	tf.reset_default_graph();
	print('Cross validation step: '+str(cv)+' Error: '+str(err))
	print(opt_params)


