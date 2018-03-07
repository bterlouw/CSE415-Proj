'''This is a random forest classifier developed by Warren Cho'''

import numpy as np
import random
from collections import Counter

class RandomForest(object):


	def __init__(self, n_est=32, max_feat=np.sqrt, max_depth=10,
				 min_samp_split=2, bs=0.9):
		self.n_est = n_est
		self.max_feat = max_feat
		self.max_depth = max_depth
		self.min_samp_split = min_samp_split
		self.bs = bs
		self.forest = []

	
	def create(self, x, y):

		self.forest = []
		n_samp = len(y)
		n_sub_samp = round(n_samp * self.bs)

		for i in range(self.n_est):
			shuffle_together(x, y)
			x_sub = x[:n_sub_samp]
			y_sub = y[:n_sub_samp]

			tree = DecisionTree(self.max_feat, self.max_depth, self.min_samp_split)
			tree.create(x_sub, y_sub)
			self.forest.append(tree)

	def predict(self, x):
		n_samp = x.shape[0]
		n_tree = len(self.forest)
		predict = np.empty([n_tree, n_samp])
		for i in range(n_tree): predict[i] = self.forest[i].predict(x)
		return max(predict, key=predict.count)[0][0]  # mode

	def score(self, x, y):
		y_predict = self.predict(x)
		n_samp = len(y)
		correct = 0
		for i in range(n_samp):
			if y_predict[i] == y[i]: correct += 1
		accuracy = correct / n_samp
		return accuracy

class DecisionTree(object):

    def __init__(self, max_feat=lambda x: x, max_depth=10, min_samp_split=2):
        '''
        max_feat is the random number of features to consider at splits
        max_depth is the maximum depth a node can be at in a tree before
            being required to be a leaf
        max_samp_split is the minimum number of samples needed at a node
        before requiring a new node split'''
        self.max_feat = max_feat
        self.max_depth = max_depth
        self.min_samp_split = min_samp_split

    def create(self, x, y):
        '''Chose decisions for each node in tree'''
        n_feat = x.shape[1]
        n_sub_feat = int(self.max_feat(n_feat))
        feat_i = random.sample(range(n_feat), n_sub_feat)
        self.base = self.build(x, y, feat_i, 0)

    def predict(self, x):
        '''Predicts the class for samples passed through parameter x'''
        n_samp = x.shape[0]
        branch = np.empty(n_samp)
        for i in range(n_samp):
            node = self.base
            while isinstance(node, Node):
                if x[i][node.feat_i] <= node.threshold: node = node.true
                else: node = node.false
            y[i] = node
        return branch

    def build(self, x, y, feat_i, depth):
        '''Builds decision tree for random forest'''
        if (depth == self.max_depth) or (len(y) < self.min_samp_split) or (entropy(y) == 0):
            return max(y)[0][0]
        feat_i, threshold = find_split(x, y, feat_i)
        trueX, trueY, falseX, falseY = split(x, y, feat_i, threshold)
        if trueX.shape[0] == 0 or falseY.shape[0] == 0: return max(y)[0][0]
        trueBranch = self.build(trueX, trueY, feat_i, depth + 1)
        falseBranch = self.build(falseX, falseY, feat_i, depth + 1)
        return Node(feat_i, threshold, trueBranch, falseBranch)

class Node(object):
    '''Node object for decision tree. Carries boolean conditions'''
    def __init__(self, feat_i, threshold, trueBranch, falseBranch):
        self.feat_i = feat_i
        self.threshold = threshold
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch

def split(x, y, feat_i, threshold):
    '''Splits x and y depending on boolean conditions in local environment'''
    trueX, trueY, falseX, falseY = ([] for i in range(4))
    
    for i in range(len(y)):
        if x[i][feat_i] <= threshold:
            trueX.append(x[i])
            trueY.append(y[i])
        else:
            falseX.append(x[i])
            falseY.append(x[i])
    trueX = np.array(trueX)
    trueY = np.array(trueY)
    falseX = np.array(falseX)
    falseY = np.array(falseY) 
    return trueX, trueY, falseX, falseY

def find_split(x, y, feat_i):
    '''Determines the optimal split for a specified tree node'''
    num_feat = x.shape[1]
    optim_gain = 0
    optim_feat_i = 0
    optim_threshold = 0
    for i in range(feat_i):
        val = sorted(set(x[:, feat_i]))
        for j in range(len(val) - 1):
            temp_threshold = (val[j] + val[j + 1] / 2)
            trueX, trueY, falseX, falseY = split(x, y, feat_i, temp_threshold)
            gain = entropy_delta(y, trueY, falseY)
            if gain > optim_gain:
                optim_gain = gain
                optim_feat_i = feat_i
                optim_threshold = temp_threshold
    return optim_feat_i, optim_threshold

def shuffle_together(a, b):
    '''Shuffles two same-length lists in a manner consistent between both lists'''
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

def entropy(Y):
    '''Measure of uncertainty of a random sample'''
    dist = Counter(Y)
    s = 0.0
    total = len(Y)
    for y, num_y in dist.items():
        p_y = (num_y / total)
        s += (p_y) * np.log(p_y)
    return -s

def entropy_delta(y, trueY, falseY):
    '''Change in entropy between two samples/splits'''
    return entropy(y) - (entropy(trueY) * len(trueY) + entropy(falseY) * len(falseY)) / len(y)