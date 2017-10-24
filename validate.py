import pickle
from sklearn import svm


with open("X_test.p", mode='rb') as f:
        X_test = pickle.load(f)

with open("y_test.p", mode='rb') as f:
        y_test = pickle.load(f)

with open("LinearSVC_trained.p", mode='rb') as f:
        clf = pickle.load(f)

print(clf.score(X_test, y_test))
