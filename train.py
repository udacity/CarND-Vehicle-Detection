import pickle
from sklearn import svm


X_train_file = './data/raw/train/X_train.p'
y_train_file = './data/raw/train/y_train.p'
cls_file = './model/raw/LinearSVC.p'

with open(X_train_file, mode='rb') as f:
        X_train = pickle.load(f)

with open(y_train_file, mode='rb') as f:
        y_train = pickle.load(f)

# parameters were determined by tune_train.py
#clf = svm.SVC(C=2.0, kernel='rbf', verbose=4)


clf = svm.LinearSVC(verbose=True, max_iter=1000000)
clf.fit(X_train, y_train)
# Use a linear SVC (support vector classifier)

with open(cls_file, mode='wb') as f:
    pickle.dump(clf, f)

