import tensorflow as tf
import numpy as np
import math
import scipy.io as sio
import optunity as opt
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import os

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

def select_feats(X,num_zero_prc_cut,var_prc_cut):
	#*********************************************************************
	# remove features with many zeros
	num_feat_zeros = np.sum(X==0,axis=1);
	X = X[num_feat_zeros<num_zero_prc_cut*X.shape[1],:]
	#*********************************************************************
	# remove features with low variance
	feat_vars = np.var(X,axis=1)
	X = X[feat_vars>np.percentile(feat_vars,var_prc_cut),:]
	return(X)

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

def get_yi(rowsums,G2,ys):
	G2 = tf.cast(G2, tf.float32);
	ys = tf.cast(ys, tf.float32);
	yi = tf.cast(tf.matmul(ys,G2), dtype=tf.float32);
	return(yi)

def run_LAmbDA(gamma, delta, tau, prc_cut, bs_prc, num_trees, max_nodes):
	global X, Y, Gnp, Dnp, train, test, prt, cv
	D = tf.cast(Dnp, tf.float32);
	G = tf.cast(Gnp, tf.float32);
	#optunity_it = optunity_it+1;
	num_trees = int(num_trees);
	max_nodes = int(max_nodes);
	prc_cut = int(np.ceil(prc_cut));
	print("gamma=%.4f, delta=%.4f, tau=%.4f, prc_cut=%i, bs_prc=%.4f, num_trees=%i, max_nodes=%i" % (gamma, delta, tau, prc_cut, bs_prc, num_trees, max_nodes))
	input_feats = X.shape[1];
	num_labls = G.shape.as_list();
	output_feats = num_labls[1];
	#print(output_feats)
	num_labls = num_labls[0];
	rowsums = np.sum(Gnp,axis=1);
	train2 = resample(prc_cut, Y, Gnp, train, gamma);				# Bug??
	bs = int(np.ceil(bs_prc*train2.size))
	xs = tf.placeholder(tf.float32, [None,input_feats])
	#ys = tf.placeholder(tf.float32, [None,num_labls])
	yin = tf.placeholder(tf.int32, [None])
	print("Vars loaded xs and ys created")
	hparams = tensor_forest.ForestHParams(num_classes=output_feats,
									num_features=input_feats,
									num_trees=num_trees,
									max_nodes=max_nodes).fill()
	print("Tensor forest hparams created")								
	forest_graph = tensor_forest.RandomForestGraphs(hparams)
	print("Tensor forest graph created")
	train_op = forest_graph.training_graph(xs, yin)
	loss_op = forest_graph.training_loss(xs, yin)
	print("Loss and train ops created")
	predict = forest_graph.inference_graph(xs)
	print("Tensor forest variables created through predict")
	accuracy_op = tf.reduce_mean(tf.reduce_sum(tf.square(tf.one_hot(yin,output_feats)-predict),reduction_indices=[1]))
	print(tf.reduce_sum(tf.square(tf.one_hot(yin,output_feats)-predict),reduction_indices=[1]))
	#predict = tf.one_hot(pred);
	print("Lambda specific variables created")
	# Creating training and testing steps
	G2 = np.copy(Gnp);
	G2[rowsums>1,:] = 0;
	YI = np.matmul(Y,G2);
	YIrs = np.sum(YI,axis=1);
	trainI = train2[np.in1d(train2,np.where(YIrs==1))];
	testI = test[np.in1d(test,np.where(YIrs==1))];
	print("trainI testI created")
	init_vars=tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init_vars)
	print("Session started")
	beep = sess.run(predict,feed_dict={xs:X[train2[0:bs],:]});
	tensor_trainI = {xs: X[trainI, :], yin: sess.run(tf.argmax(get_yi(rowsums,G2,Y[trainI, :]),axis=1))}
	print("tensor_trainI made")
	tensor_testI = {xs: X[testI, :], yin: sess.run(tf.argmax(get_yi(rowsums,G2,Y[testI, :]),axis=1))}
	print("tensor_testI made")
	tensor_train = {xs: X[train2[0:bs], :], yin: sess.run(tf.argmax(get_yn(sess.run(predict,feed_dict={xs:X[train2[0:bs],:]}),Y[train2[0:bs], :],delta,tau,output_feats),axis=1))}
	print("tensor_train made")
	tensor_test = {xs: X[test, :], yin: sess.run(tf.argmax(get_yn(sess.run(predict,feed_dict={xs:X[test,:]}),Y[test, :],delta,tau,output_feats),axis=1))}
	print("tensor_test made")
	#**********************************
	#print("Loss and training steps created with sample tensors")
	# Setting params and initializing
	print("Beginning iterations")
	# Starting training iterations
	print(X.shape)
	for i in range(1,101):
		if i < 50:
			sess.run(train_op, feed_dict=tensor_trainI)
			#print("ran train op")
			if i % 10 == 0:
				print(str(sess.run(accuracy_op, feed_dict=tensor_trainI)) + ' ' + str(sess.run(accuracy_op, feed_dict=tensor_testI)))
		else:
			sess.run(train_op, feed_dict=tensor_train)
			if i % 10 == 0:
				print(str(sess.run(accuracy_op, feed_dict=tensor_train)) + ' ' + str(sess.run(accuracy_op, feed_dict=tensor_test)))
			elif i % 10 == 0:
				np.random_shuffle(train2);
				tensor_train = {xs: X[train2[0:bs], :], yin: sess.run(get_yn(sess.run(predict,feed_dict={xs:X[train2[0:bs],:]}),Y[train2[0:bs], :],delta,tau,output_feats))}
	if prt:
		blah = sess.run(predict, feed_dict=tensor_test);
		sio.savemat('preds118_cv' + str(cv) + '.mat', {'preds': blah});
		sio.savemat('truth118_cv' + str(cv) + '.mat', {'labels': Y[test, :]});
	acc = sess.run(accuracy_op, feed_dict=tensor_test) 
	print("loss1=%.4f, gamma=%.4f, delta=%.4f, tau=%.4f, prc_cut=%i, bs_prc=%.4f, num_trees=%i, max_nodes=%i" % (acc, gamma, delta, tau, prc_cut, bs_prc, num_trees, max_nodes))
	tf.reset_default_graph();
	return(acc)

# Running CV
for i in range(1,11):
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	global X, Y, Gnp, Dnp, train, test, prt, cv
	cv = i;
	print('Cross validation step: '+str(cv))
	'''
	X = sio.loadmat('LAmbDA/LAmbDA_data/ZeiselLakeDarm_cpmtpm.mat');
	X = np.array(X['X']);
	X = X.astype(np.float32)
	X = np.log2(np.transpose(select_feats(np.transpose(X),0.5,80))/10+1);
	Y = sio.loadmat('LAmbDA/LAmbDA_data/ZeiselLakeDarm_labels.mat');
	Y = np.array(Y['Y']);
	Y = Y.astype(np.float32)
	G = sio.loadmat('LAmbDA/LAmbDA_data/label_mask6.mat');
	Gnp = np.array(G['G']);
	D = sio.loadmat('LAmbDA/LAmbDA_data/ZeiselLakeDarm_dset.mat');
	Dnp = np.array(D['D']);
	'''
	X = sio.loadmat('pancreas/pancreasXexpr.mat');
	X = np.array(X['X']);
	Y = sio.loadmat('pancreas/pancreasYlabels.mat');
	Y = np.array(Y['Y']);
	G = sio.loadmat('pancreas/pancreasGmask2.mat');
	Gnp = np.array(G['G']);
	D = sio.loadmat('pancreas/pancreasDdset.mat');
	Dnp = np.array(D['D']);
	train_samp = int(np.floor(0.6*X.shape[0]))
	test_samp = int(np.floor(0.2*X.shape[0]))
	val_samp = int(X.shape[0] - (train_samp+test_samp))
	perm = np.random.permutation(X.shape[0]);
	train = perm[0:train_samp+1];
	test = perm[train_samp+1:train_samp+test_samp+1];
	val = perm[train_samp+test_samp+1:train_samp+test_samp+val_samp+1];
	while(np.sum(np.sum(Y[train,:],0)<1)>0):
		perm = np.random.permutation(X.shape[0]);
		train = perm[0:train_samp+1];
		test = perm[train_samp+1:train_samp+test_samp+1];
		val = perm[train_samp+test_samp+1:train_samp+test_samp+val_samp+1];
	optunity_it = 0;
	prt = False
	opt_params = None
	opt_params, _, _ = opt.minimize(run_LAmbDA,solver_name='sobol', gamma=[0.8,1.2], delta=[0.05,0.95], tau=[1.0,2.0], prc_cut=[20,50], bs_prc=[0.2,0.6], num_trees=[10,200], max_nodes=[100,1000], num_evals=50)
	print(opt_params)
	prt = True
	train = perm[0:train_samp+test_samp+1]
	test = val
	err = run_LAmbDA(opt_params['gamma'], opt_params['delta'], opt_params['tau'], opt_params['prc_cut'], opt_params['bs_prc'], opt_params['num_trees'], opt_params['max_nodes'])
	tf.reset_default_graph();
	print('Cross validation step: '+str(cv))
	print(opt_params)
	print(err)

	
#*****************************************************
# For testing
'''
os.environ["CUDA_VISIBLE_DEVICES"] = ""
global X, Y, G, D, train, test, prt, cv
cv = 1;
X = sio.loadmat('pancreas/pancreasXexpr.mat');
X = np.array(X['X']);
X = np.log2(np.transpose(select_feats(np.transpose(X),0.3,50))/10+1);
X = X.astype(np.float32)
Y = sio.loadmat('pancreas/pancreasYlabels.mat');
Y = np.array(Y['Y']);
Y = Y.astype(np.float32)
G = sio.loadmat('pancreas/pancreasGmask.mat');
G = np.array(G['G']);
G = tf.cast(G, tf.float32);
D = sio.loadmat('pancreas/pancreasDdset.mat');
D = np.array(D['D']);
D = tf.cast(D, tf.float32);
prt = True;
train_samp = 8140;
test_samp = 2268;
val_samp = 2267;
perm = np.random.permutation(12675);
train = perm[0:train_samp+1];
test = perm[train_samp+1:train_samp+test_samp+1];
val = perm[train_samp+test_samp+1:train_samp+test_samp+val_samp+1]
run_LAmbDA(0.8,0.8,1.0,60,0.6,100,1000)
'''								

