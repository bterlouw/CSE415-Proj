'''This is a random forest classifier developed by Warren Cho'''

import numpy as np
import random
from collections import Counter
from scipy.stats import mode

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
		#print('x -> ' + str(x))
		n_samp = x.shape[0]		#should be 4
		n_tree = len(self.forest)
		predict = np.zeros([n_tree, n_samp])
		#for i in range(n_tree):
		#	print("predict(x) for tree in forest: " + str(self.forest[i].predict(x)))
		#	print('predict[i]: ' + str(predict[i]))
		#	predict[i] = self.forest[i].predict(x)
		#print('mode output: '+ str(mode(predict)))
		#return mode(predict)[0][0]
		#print('predict: ' + str(predict))
		return predict[0][0]

	def score(self, x, y):
		y_predict = self.predict(x)
		n_samp = len(y)
		correct = 0
		for i in range(n_samp):
			if y_predict == y[i]: correct += 1
		accuracy = correct / n_samp
		return accuracy

class DecisionTree(object):

	def __init__(self, max_feat=np.sqrt, max_depth=10, min_samp_split=2):
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
		'''Choose decisions for each node in tree'''
		n_feat = x.shape[1]
		n_sub_feat = int(self.max_feat(n_feat))
		print('Number of features in subset: ' + str(n_sub_feat))                   ### testing print
		feat_i = random.sample(range(n_feat), n_sub_feat)
		print('Indecies of features in subset: '+  str(feat_i))                     ### testing print
		self.base = self.build(x, y, feat_i[0], 0)

	def predict(self, x):
		'''Predicts the class for samples passed through parameter x'''
		n_samp = x.shape[0]
		y = np.empty(n_samp)

		node = self.base
		if isinstance(node, Node):
			if x[0][node.feat_i] <= node.threshold: node = node.trueBranch
			else: node = node.falseBranch
		y = node
		#print(y)

		#for i in range(n_samp):
		#	node = self.base
		#	while isinstance(node, Node):
		#		print(node)
		#		if x[i][node.feat_i] <= node.threshold: node = node.trueBranch
		#		else: node = node.falseBranch
		#	print(y[i])
		#	y[i] = node
		return y

	def build(self, x, y, feat_i, depth):
		'''Builds decision tree for random forest'''
		if (depth == self.max_depth) or (len(y) < self.min_samp_split) or (entropy(y) == 0):
			return mode(y)[0][0]
		feat_index, threshold = find_split(x, y, feat_i)
		trueX, trueY, falseX, falseY = split(x, y, feat_index, threshold)
		if trueX.shape[0] == 0 or falseY.shape[0] == 0: return mode(y)[0][0]
		trueBranch = self.build(trueX, trueY, feat_i, depth + 1)
		falseBranch = self.build(falseX, falseY, feat_i, depth + 1)
		return Node(feat_index, threshold, trueBranch, falseBranch)

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
	#assert len(a) == len(b)
	#p = np.random.permutation(len(a))
	#return a[p], b[p]
	state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(state)
	np.random.shuffle(b)

def entropy(y):
	'''Measure of uncertainty of a random sample'''
	dist = Counter(y.tostring())
	s = 0.0
	total = len(y)
	for y2, num_y in dist.items():
		p_y = (num_y / total)
		s += (p_y) * np.log(p_y)
	return -s

def entropy_delta(y, trueY, falseY):
	'''Change in entropy between two samples/splits'''
	return entropy(y) - (entropy(trueY) * len(trueY) + entropy(falseY) * len(falseY)) / len(y)