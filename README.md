# LAmbDA
Label Ambiguous Domain Adaption  
Travis S Johnson MS, Yi Wu PhD, Yatong Han BS, Kun Huang PhD

Dependencies: Numpy, Math, Scipy, TensorFlow

LAmbDA is an algorithm that assigns unknown labels to datasets and trains
on those labels across multiple datasets. This process requires inputting
a matrix X which is the concatenation of all datasets using the intersecting
features (rows samples and columns features). Y which is the matrix of labels such that each cell type in
each starting dataset has it's own label in onehot format(rows samples and columns labels). G which is
the relationships of these labels to the subset of labels that are being
trained (rows input labels and columns output labels).

Other specifications include  
gamma: the weighting related to the ambiguity of each label.  
delta: constant for the weighted average of a single cell to input label
(i.e. how much the individual cell is weighted by the average of its input cell type)  
tau: the dispersion constant that promotes more diverse output labels  
lambda1: the regularization term to be used in the neural network model  
lambda2: the attraction weight between similar subtypes at the hidden layer
lambda3: the dispersion weight between dissimilar subtypes at the hidden layer

All files needed to run LAmbDA are included except for the data matrix which is too large
to be hosted on github
