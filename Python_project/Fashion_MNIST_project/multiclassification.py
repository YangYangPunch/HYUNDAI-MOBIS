import numpy as np
import pandas as pd
import time

''' check the computing time '''
start = time.time()

''' load a testset, 10000 '''
pd_fashion_test = pd.read_csv("fashion-mnist_test.csv")
np_fashion_test = np.array(pd_fashion_test)

fashion_test_label = np.array(np_fashion_test[:, 0])
fashion_test_set = np.array(np_fashion_test[:, 1:785])


''' load a trainset, 60000 '''
pd_fashion_train = pd.read_csv("fashion-mnist_train.csv")
np_fashion_train = np.array(pd_fashion_train)

fashion_train_label = np.array(np_fashion_train[:, 0])
fashion_train_set = np.array(np_fashion_train[:, 1:785])


''' z_normalization function '''
def z_normalization(train_set):
    train_set.astype(np.float64)
    train_set = train_set / 255
    
    std_train = np.zeros_like(train_set)
    
    for row in range(len(train_set[:, 0])):
        std_train[row] = (train_set[row] - np.mean(train_set[row])) / np.std(train_set[row]+0.01)
    return std_train


'''  stardardiation '''
fashion_train_set = z_normalization(fashion_train_set)
fashion_test_set = z_normalization(fashion_test_set)


from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


''' Support Vector Classifier 89%'''
svm_model = SVC(C=1, kernel='rbf')
svm_model.fit(fashion_train_set, fashion_train_label)

y_pred = svm_model.predict(fashion_test_set)

precision = precision_score(fashion_test_label, y_pred, average="macro")
recall = recall_score(fashion_test_label, y_pred, average="macro")
report = metrics.classification_report(fashion_test_label, y_pred)
print("precision: ", precision)
print("recall: ", recall)
print("Recall + Precision: ", recall + precision)
print("\nclassification_report\n", report)


''' KNeighbors Classifier 88% accuracy'''
# knn_clf = KNeighborsClassifier()
# knn_clf.fit(fashion_train_set, fashion_train_label)

# y_pred = knn_clf.predict(fashion_test_set)


'''LearRegression classifier 82% accuracy '''
# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(fashion_train_set, fashion_train_label)

# y_pred = sgd_clf.predict(fashion_test_set)

# precision = precision_score(fashion_test_label, y_pred, average="weighted")
# recall = recall_score(fashion_test_label, y_pred, average="weighted")
# report = metrics.classification_report(fashion_test_label, y_pred)
# print("precision: ", precision)
# print("recall: ", recall)
# print("Recall + Precision: ", recall + precision)
# print("\nclassification_report\n", report)


''' RandomForest Classifier 88% accuracy'''
# forest_clf = RandomForestClassifier(n_estimators=500, random_state=42)
# forest_clf.fit(fashion_train_set, fashion_train_label)

# y_pred = forest_clf.predict(fashion_test_set)

# precision = precision_score(fashion_test_label, y_pred, average="weighted")
# recall = recall_score(fashion_test_label, y_pred, average="weighted")
# report = metrics.classification_report(fashion_test_label, y_pred)
# print("precision: ", precision)
# print("recall: ", recall)
# print("Recall + Precision: ", recall + precision)
# print("\nclassification_report\n", report)



# precision = precision_score(fashion_test_label, y_pred, average="weighted")
# recall = recall_score(fashion_test_label, y_pred, average="weighted")
# report = metrics.classification_report(fashion_test_label, y_pred)
# print("precision: ", precision)
# print("recall: ", recall)
# print("Recall + Precision: ", recall + precision)
# print("\nclassification_report\n", report)


end = time.time()

total_time = int(end - start)

hours = total_time // 3600
minutes = (total_time - hours*60) // 60
second = total_time % 60
print("hours: ", hours)
print("minutes: ", minutes)
print("second: ", second)