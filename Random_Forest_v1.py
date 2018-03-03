'''This is a random forest classifier developed by Warren Cho'''

import numpy as np
import random

class RandomForest(object):


	def __init__(self, n_est=32, max_feat=np.sqrt, max_depth=10,
				 min_samp_split=2, bs=0.9):
		self.n_est = n_est
		self.max_feat = max_feat
		self.max_depth = max_depth
		self.min_samp_split = min_samp_split
		self.bs = bs
		self.forest = []

	
	def fit(self, x, y):

		self.forest = []
		n_samp = len(y)
		n_sub_samp = round(n_samp * self.bs)

		for i in range(self.n_est):
			shuffle_together(x, y)
			x_sub = x[:n_sub_samp]
			y_sub = y[:n_sub_samp]

			tree = DecisionTreeClassifier(self.max_feat, self.max_depth, self.min_samp_split)
			tree.fit(x_sub, y_sub)
			self.forest.append(tree)

	def predict(self, x):
		n_samp = x.shape[0]
		n_tree = len(self.forest)
		predict = np.empty([n_tree, n_samp])
		for i in range(n_tree): predict[i] = self.forest[i].predict(x)
		return mode(predict)[0][0]

	def score(self, x, y):
		y_predict = self.predict(x)
		n_samp = len(y)
		correct = 0
		for i in range(n_samp):
			if y_predict[i] == y[i]: correct += 1
		accuracy = correct / n_samp
		return accuracy


def shuffle_together(a, b):
	assert len(a) == len(b)
	p = numpy.random.permutation(len(a))
	return a[p], b[p]

def entropy(Y):
	dist = Counter(Y)
	s = 0.0
	total = len(Y)
	for y, num_y in dist.items():
		p_y = (num_y / total)
		s += (p_y) * np.log(p_y)
	return -s

def entropy_delta(y, y_true, y_false):
	return entropy(y) - (entropy(y_true) * len(y_true) + entropy(y_false) * len(y_false)) / len(y)