import pickle

X_test_file = './data/raw/test/X_test.p'
y_test_file = './data/raw/test/y_test.p'
model_file = './model/raw/LinearSVC.p'

with open(X_test_file, mode='rb') as f:
        X_test = pickle.load(f)

with open(y_test_file, mode='rb') as f:
        y_test = pickle.load(f)

with open(model_file, mode='rb') as f:
        clf = pickle.load(f)

print(clf.score(X_test, y_test))
