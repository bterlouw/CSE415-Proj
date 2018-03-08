import numpy as np
import csv
import random
from Random_Forest_v1 import RandomForest
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from _sqlite3 import Row

def run():
	data = []
	with open('IrisData.csv', newline = '') as f:
		reader = csv.reader(f)
		next(reader)
		for row in reader:
			curr = row.pop(4)
			temp = []
			if curr == 'setosa':
				temp.append(0)
				temp.extend([float(i) for i in row])
				data.append(temp)
			elif curr == 'versicolor':
				temp.append(1)
				temp.extend([float(i) for i in row])
				data.append(temp)
	random.shuffle(data)
	finalTest = data.pop()

	###

	x = np.array(data)[:, 1:]
	y = np.array(data)[:, 0]

	xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(x, y)

	forest = RandomForest()
	forest.create(xTrain, yTrain)
	#print(xTest)
	#print(yTest)

	accuracy = forest.score(xTest, yTest)
	print('Acurracy: ' + str(100*accuracy) + '% on test data')

	classification = forest.predict(xTest)
	guess = 'Virginica'
	if classification == 0.0: 
		guess = 'Setosa'
	elif classification == 1.0: 
		guess = 'Versicolor'

	print('Sample at index 0 of test data was classified as: ' + guess)

###

#parameter_gridsearch = {
#                 'max_depth' : [3, 4],  #depth of each decision tree
#                 'n_estimators': [50, 20],  #count of decision tree
#                 'max_features': ['sqrt', 'auto', 'log2'],      
#                 'min_samples_split': [2],      
#                 'min_samples_leaf': [1, 3, 4],
#                 'bootstrap': [True, False],
#                 }

#train = data
#test = data

#randomforest = RandomForest()
#crossvalidation = StratifiedKFold(np.array(train)[0::,0], n_folds=5)

#gridsearch = GridSearchCV(randomforest, scoring = 'accuracy',
#                          param_grid = parameter_gridsearch, cv = crossvalidation)

#gridsearch.fit(np.array(train)[0::, 1::], np.array(train)[0::, 0])
#model = gridsearch
#param = gridsearch.best_params_

#print('Best score: {}'.format(gridsearch.best_score_))