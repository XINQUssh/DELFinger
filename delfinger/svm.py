### SVM Classifier    
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import time

normal_feature_1 = np.loadtxt('normal_feature_1.txt')
print(normal_feature_1)

normal_feature_test_1 = np.loadtxt('normal_feature_test_1.txt')

train_x = normal_feature_1[:,:5]
train_y = normal_feature_1[:,-1]

test_x = normal_feature_test_1[:,:5]
test_y = normal_feature_test_1[:,-1]

print(train_x)
print(train_y)

clf = SVC(kernel='rbf', probability=True)
clf.fit(train_x, train_y)

# print(clf.best_params_)
pre_y_train = clf.predict(train_x)
pre_y_test = clf.predict(test_x)


cnt_0,cnt_1 = 0,0
error_0,error_1 = 0,0
for i in range(len(test_y)):
    if test_y[i] == 0:
        cnt_0 += 1
        if pre_y_test[i] == 1:
            error_0 += 1
    elif test_y[i] == 1:
        cnt_1 += 1
        if pre_y_test[i] == 0:
            error_1 += 1
            
print(cnt_0, cnt_1)
print(error_0, error_1)
print(len(test_y))


print("SVM precision_score : {0}".format(metrics.precision_score(test_y, pre_y_test)))
print("SVM recall_score : {0}".format(metrics.recall_score(test_y, pre_y_test)))
print("SVM Metrics : {0}".format(metrics.precision_recall_fscore_support(test_y, pre_y_test)))