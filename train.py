import pickle
from sklearn import svm


with open("X_train.p", mode='rb') as f:
        X_train = pickle.load(f)

with open("y_train.p", mode='rb') as f:
        y_train = pickle.load(f)

# parameters were determined by tune_train.py
#clf = svm.SVC(C=2.0, kernel='rbf', verbose=4)


clf = svm.LinearSVC(verbose=4)
clf.fit(X_train, y_train)
# Use a linear SVC (support vector classifier)

with open("LinearSVC_trained.p", mode='wb') as f:
    pickle.dump(clf, f)

