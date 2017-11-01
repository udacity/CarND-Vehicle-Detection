import pickle
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV


with open("X_train_small.p", mode='rb') as f:
        X_train = pickle.load(f)

with open("y_train_small.p", mode='rb') as f:
        y_train = pickle.load(f)

parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 0.5, 1, 2, 4, 8]}
svc = svm.SVC()

clf = GridSearchCV(svc, parameters, verbose=4)
clf.fit(X_train, y_train)
# Use a linear SVC (support vector classifier)
print(clf.cv_results_)
print(clf.best_params_)
print(clf.best_score_)

