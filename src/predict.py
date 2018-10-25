# copyright: yueshi@usc.edu
import pandas as pd 
import hashlib
import os 
from utils import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


from sklearn.feature_selection import SelectFromModel
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC 
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from sklearn.decomposition import PCA # Principal Component Analysis module
from sklearn.manifold import TSNE # TSNE module

from utils import logger

#def lassoSelection(X,y,)

def lassoSelection(X_train, y_train, n):
	'''
	Lasso feature selection.  Select n features. 
	'''
	#lasso feature selection
	#print (X_train)
	clf = LassoCV()
	sfm = SelectFromModel(clf, threshold=0)
	sfm.fit(X_train, y_train)
	X_transform = sfm.transform(X_train)
	n_features = X_transform.shape[1]
	
	#print(n_features)
	while n_features > n:
		sfm.threshold += 0.01
		X_transform = sfm.transform(X_train)
		n_features = X_transform.shape[1]
	features = [index for index,value in enumerate(sfm.get_support()) if value == True  ]
	logger.info("selected features are {}".format(features))
	return features


def specificity_score(y_true, y_predict):
	'''
	true_negative rate
	'''
	true_negative = len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[0]==pair[1] and pair[0]==0 ])
	real_negative = len(y_true) - sum(y_true)
	return true_negative / real_negative 

def model_fit_predict(X_train,X_test,y_train,y_test):

	np.random.seed(2018)
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.ensemble import ExtraTreesClassifier
	from sklearn.svm import SVC
	from sklearn.metrics import precision_score
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import f1_score
	from sklearn.metrics import recall_score
	models = {
		'LogisticRegression': LogisticRegression(),
		'ExtraTreesClassifier': ExtraTreesClassifier(),
		'RandomForestClassifier': RandomForestClassifier(),
    	'AdaBoostClassifier': AdaBoostClassifier(),
    	'GradientBoostingClassifier': GradientBoostingClassifier(),
    	'SVC': SVC()
	}
	tuned_parameters = {
		'LogisticRegression':{'C': [1, 10]},
		'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
		'RandomForestClassifier': { 'n_estimators': [16, 32] },
    	'AdaBoostClassifier': { 'n_estimators': [16, 32] },
    	'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    	'SVC': {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
	}
	scores= {}
	for key in models:
		clf = GridSearchCV(models[key], tuned_parameters[key], scoring=None,  refit=True, cv=10)
		clf.fit(X_train,y_train)
		y_test_predict = clf.predict(X_test)
		#precision = precision_score(y_test, y_test_predict)
		#accuracy = accuracy_score(y_test, y_test_predict)
		#f1 = f1_score(y_test, y_test_predict)
		#recall = recall_score(y_test, y_test_predict)
		#specificity = specificity_score(y_test, y_test_predict)
		#scores[key] = [precision,accuracy,f1,recall,specificity]
		#print(scores)
		#return scores

def graph(numlabels, features):

	# Invoke the PCA method. Since this is a binary classification problem
	# let's call n_components = 2
	pca = PCA(n_components=2)
	pca_2d = pca.fit_transform(features)

	# Invoke the TSNE method
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
	tsne_results = tsne.fit_transform(features)

	# Plot the TSNE and PCA visuals side-by-side
	plt.figure(figsize = (16,11))
	plt.subplot(121)
	plt.scatter(pca_2d[:,0],pca_2d[:,1], c = numlabels, 
		    cmap = "coolwarm", edgecolor = "None", alpha=0.35)
	plt.colorbar()
	plt.title('PCA Scatter Plot')
	plt.subplot(122)
	plt.scatter(tsne_results[:,0],tsne_results[:,1],  c = numlabels, 
		    cmap = "coolwarm", edgecolor = "None", alpha=0.35)
	plt.colorbar()
	plt.title('TSNE Scatter Plot')
	plt.show()

	
def draw(scores):
	'''
	draw scores.
	'''
	import matplotlib.pyplot as plt
	logger.info("scores are {}".format(scores))
	ax = plt.subplot(111)
	precisions = []
	accuracies =[]
	f1_scores = []
	recalls = []
	categories = []
	specificities = []
	N = len(scores)
	ind = np.arange(N)  # set the x locations for the groups
	width = 0.1        # the width of the bars
	for key in scores:
		categories.append(key)
		precisions.append(scores[key][0])
		accuracies.append(scores[key][1])
		f1_scores.append(scores[key][2])
		recalls.append(scores[key][3])
		specificities.append(scores[key][4])

	precision_bar = ax.bar(ind, precisions,width=0.1,color='b',align='center')
	accuracy_bar = ax.bar(ind+1*width, accuracies,width=0.1,color='g',align='center')
	f1_bar = ax.bar(ind+2*width, f1_scores,width=0.1,color='r',align='center')
	recall_bar = ax.bar(ind+3*width, recalls,width=0.1,color='y',align='center')
	specificity_bar = ax.bar(ind+4*width,specificities,width=0.1,color='purple',align='center')

	print(categories)
	ax.set_xticks(np.arange(N))
	ax.set_xticklabels(categories)
	ax.legend((precision_bar[0], accuracy_bar[0],f1_bar[0],recall_bar[0],specificity_bar[0]), ('precision', 'accuracy','f1','sensitivity','specificity'))
	ax.grid()
	plt.show()

if __name__ == '__main__':


	data_dir ="/Users/RyanLiu/Projects/lab10/EE542Lab10/"

	data_file = data_dir + "miRNA_matrix.csv"


	#read file and remove file_id field
	df = pd.read_csv(data_file)
	df.pop('file_id')

	labels = df['label']
	y_data = labels

	#convert labels into numbers and remove label field
	df['label'] = df['label'].factorize()[0]
	numlabels = df['label']
	df.pop('label')

	print ("label converted into numbers in the order of following columns from 0 to X:")
	#print (numlabels)

	# Turn dataframe into arrays (X value)
	features = df.values 


	#check the features and labels
	#print(features)
	#print()
	#print(labels)


	# split the data to train and test set
	X_train, X_test, y_train, y_test = train_test_split(features, numlabels, test_size=0.3, random_state=0)
	
	#standardize the data.
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	#check the training,test set shape
	print("X_train shape: {}".format(X_train.shape))
	print("X_test shape: {}".format(X_test.shape))
	print("y_train shape: {}".format(y_train.shape))
	print("y_test shape: {}".format(y_test.shape))
	

	#standardize the data.
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

#   	#Training SVM model
#	svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
#   	#Use model to predict
#	svm_predictions = svm_model_linear.predict(X_test) 
 
#	precision, recall, fscore, support = score(y_test, svm_predictions)
#	print('precision: {}'.format(precision))
#	print('recall: {}'.format(recall))
#	print('fscore: {}'.format(fscore))
#	print('support: {}'.format(support))

	
	n = 55
	feaures_columns = lassoSelection(X_train, y_train, n)

	print("these are feature headers:")
	print(np.unique(labels))
	n_classes = len(np.unique(labels)) - 1

	#visualization
	#graph(numlabels, features[:,feaures_columns])

	classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
	y_score = classifier.fit(X_train[:,feaures_columns], y_train)

	print(classification_report(y_train, classifier.predict(X_train[:,feaures_columns])))

	# Compute ROC curve and ROC area for each class






	#scores = model_fit_predict(X_train[:,feaures_columns],X_test[:,feaures_columns],y_train,y_test)
	#draw(scores)

	#lasso cross validation
	# lassoreg = Lasso(random_state=0)
	# alphas = np.logspace(-4, -0.5, 30)
	# tuned_parameters = [{'alpha': alphas}]
	# n_fold = 10
	# clf = GridSearchCV(lassoreg,tuned_parameters,cv=10, refit = False)
	# clf.fit(X_train,y_train)

