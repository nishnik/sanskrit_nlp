# Applies 10 fold cross validation
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import csv
#from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
#from sklearn.externals.six import StringIO
import random
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle
# #rom sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import pylab as pl
import json
with open('feature_train.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)


your_list = your_list[1:] # skips header

feature = []
label = []
first_word = []
second_word = []
for data in your_list[0:]:
	feature.append(data[0:11270])
	label.append(data[11270])
	first_word.append(data[11271])
	second_word.append(data[11272])

feature_f = []
for row in feature:
	row = [float(i) for i in row]
	feature_f.append(row)

X = np.array(feature_f)
y = np.array(label)
first_word = np.array(first_word)
second_word = np.array(second_word)

n_samples, n_features = X.shape 
p = list(range(n_samples)) # an index array, 0:n_samples
random.seed(random.random())
random.shuffle(p) # the index array is now shuffled

X, y = X[p], y[p] # both the arrays are now shuffled
first_word, second_word = first_word[p], second_word[p]
y = y.reshape(len(y), 1) # this is important to vstack y_train


num_folds = 10
subset_size = len(X)/num_folds
dict_ = {}

for i in range(num_folds):
    X_test = X[i*subset_size:(i+1)*subset_size]
    y_test = y[i*subset_size:(i+1)*subset_size]
    X_train = np.vstack([X[:i*subset_size], X[(i+1)*subset_size:]])
    y_train = np.vstack([y[:i*subset_size], y[(i+1)*subset_size:]]).reshape(len(y_train),) # reshape is necessary to train
    first_word_test = first_word[i*subset_size:(i+1)*subset_size]
    second_word_test = second_word[i*subset_size:(i+1)*subset_size]
    clf = RandomForestClassifier(max_features = 11270 )
    clf = clf.fit(X_train, y_train)
    predicted_label = clf.predict(X_test)
    print (classification_report(y_test, predicted_label))
    print (i + 1, 'Accuracy:',accuracy_score(y_test, predicted_label))
    for j in range(len(X_test)):
        hash_ = first_word_test[j] +","+ second_word_test[j]
        dict_[hash_] = {}
        dict_[hash_]["actual"] = y_test[j][0]
        dict_[hash_]["predicted"] = predicted_label[j]

with open('test.txt', 'w') as outfile:
    json.dump(dict_, outfile, indent = 4, ensure_ascii=False)
