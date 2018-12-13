# LAmbDA
Label Ambiguous Domain Adaption  
Travis S Johnson MS, Zhi Huang MS, Christina Yu BS, Tongxin Wang BS, Yi Wu PhD, Yatong Han BS, Kun Huang PhD, Jie Zhang PhD

Dependencies: Numpy, Math, Scipy, TensorFlow, Optunity

LAmbDA is an algorithm that assigns unknown labels to datasets and trains
on those labels across multiple datasets. This process requires inputting
a matrix X which is the concatenation of all datasets using the intersecting
features (rows samples and columns features). Y which is the matrix of labels such that each cell type in
each starting dataset has it's own label in onehot format(rows samples and columns labels). G which is
the relationships of these labels to the subset of labels that are being
trained (rows input labels and columns output labels). D is the dataset of each label.

Other specifications include:

All model types (LR, FF1, FF3, RNN1, RF)
gamma: the weighting related to the ambiguity of each label.  
delta: constant for the weighted average of a single cell to input label
(i.e. how much the individual cell is weighted by the average of its input cell type)  
tau: the dispersion constant that promotes more diverse output labels
prc_cut: the over/under-sampling cutoff
bs_prc: batch size (percentage of total samples)

Regression based model types (LR, FF1, FF3, RNN1)
do_prc: the percent of network nodes retained for drop-out
lambda1: the regularization term to be used in the neural network model

Neural network model types (FF1, FF3, RNN1)
lambda2: the attraction weight between similar subtypes at the hidden layer
lambda3: the dispersion weight between dissimilar subtypes at the hidden layer
n_hidden: number of nodes in each hidden layer

Decision tree model types (RF)
n_trees: number of trees in random forest
n_max: max nodes in each tree

All files needed to run LAmbDA are included except for the data matrix which is too large
to be hosted on github
