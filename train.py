import pickle
from sklearn import svm

# input
X_train_file = './data/augmented/train/X_train.p'
y_train_file = './data/augmented/train/y_train.p'

# output
cls_file = './model/augmented/LinearSVC.p'

with open(X_train_file, mode='rb') as f:
        X_train = pickle.load(f)

with open(y_train_file, mode='rb') as f:
        y_train = pickle.load(f)

print("X_train length: ", len(X_train))
print("y_train length: ", len(y_train))

# parameters were determined by tune_train.py
#clf = svm.SVC(C=2.0, kernel='rbf', verbose=4)


clf = svm.LinearSVC(penalty='l2', dual=False, C=0.1, verbose=2, max_iter=1000000)
clf.fit(X_train, y_train)
# Use a linear SVC (support vector classifier)

with open(cls_file, mode='wb') as f:
    pickle.dump(clf, f)

