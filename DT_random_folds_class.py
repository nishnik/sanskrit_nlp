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
with open('feature_without_zero_coarse.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)


your_list = your_list[1:]

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
kfold = 10 # no. of folds (better to have this at the start of the code)

skf = StratifiedKFold(y,kfold)

# Stratified KFold: This first divides the data into k folds. Then it also makes sure that the distribution of the data in each fold follows the original input distribution 
# Note: in future versions of scikit.learn, this module will be fused with kfold

skfind = [None]*len(skf) # indices
cnt=0
for train_index in skf:
  skfind[cnt] = train_index
  cnt = cnt + 1


for i in range(kfold):
 train_indices = skfind[i][0]
 test_indices = skfind[i][1]


X_train = X[train_indices]
y_train = y[train_indices]
first_word_train, second_word_train = first_word[train_indices], second_word[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]
first_word_test, second_word_test = first_word[test_indices], second_word[test_indices]


#X_train, X_test, y_train, y_test = train_test_split(feature_f, label, test_size=0.3)
clf = RandomForestClassifier(max_features = 11270 )
#cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.3)
clf = clf.fit(X_train, y_train)

predicted_label = clf.predict(X_test)

print (classification_report(y_test, predicted_label))
print ('Accuracy:',accuracy_score(y_test, predicted_label))

dict_ = {}

for i in range(len(X_test)):
	hash_ = first_word_test[i] +","+ second_word_test[i]
	dict_[hash_] = {}
	dict_[hash_]["actual"] = y_test[i]
	dict_[hash_]["predicted"] = predicted_label[i]


with open('test.txt', 'w') as outfile:
     json.dump(dict_, outfile, indent = 4, ensure_ascii=False)

predicted_label = clf.predict(X_train)

dict_ = {}

for i in range(len(X_train)):
	hash_ = first_word_train[i] +","+ second_word_train[i]
	dict_[hash_] = {}
	dict_[hash_]["actual"] = y_train[i]
	dict_[hash_]["predicted"] = predicted_label[i]


with open('train.txt', 'w') as outfile:
     json.dump(dict_, outfile, indent = 4, ensure_ascii=False)


with open('test.txt') as json_data:
    d = json.load(json_data)
    print(d)

# with open("coarse.dot", 'w') as f:
# 	f = tree.export_graphviz(clf, out_file=f)


# conf_labels = ['A','B','D','T']
# cm = confusion_matrix(y_test, predicted_label,conf_labels)
# print ('confusion matrix:')
# print (cm)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(cm)
# pl.title('Confusion matrix of the classifier')
# fig.colorbar(cax)
# ax.set_xticklabels([''] + conf_labels)
# ax.set_yticklabels([''] + conf_labels)
# pl.xlabel('Predicted')
# pl.ylabel('True')
# pl.show()