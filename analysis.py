import sys

import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from numpy import genfromtxt

def run_analysis(fpath):
    my_data = pd.read_csv(fpath, sep=',', header=None).values
    X, y = my_data[:,:-1], my_data[:, -1]

# Build a classification task using 3 informative features

# Create the RFE object and compute a cross-validated score.
    svc = LogisticRegression()
# The "accuracy" scoring is proportional to the number of correct
# classifications
    rfecv = RFECV(estimator=svc, step=1, cv=KFold(len(X)),
                  scoring='accuracy')
    rfecv.fit(X, y)
    print(rfecv.ranking_)

if __name__ == "__main__":
    run_analysis(sys.argv[1])
