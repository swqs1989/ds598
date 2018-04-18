import importlib
import func
importlib.reload(func)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold
pd.options.display.max_rows = 100
importlib.reload(func)
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import svm
from sklearn.svm import NuSVC
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

df_101_tier = pd.read_excel("data/df_tierc700.xlsx")
print(df_101_tier.shape)

X = df_101_tier.drop("Tier1D", axis=1)
y = df_101_tier["Tier1D"]

print(y.groupby(y).count())

rfmodel = RandomForestClassifier(n_estimators=500, class_weight='balanced', n_jobs=-1)
rfmodel.fit(X, y)
feat_import = pd.DataFrame({"Feature": X.columns, "Importance": rfmodel.feature_importances_})\
        .sort_values("Importance", ascending=False)
feat = feat_import.iloc[:20, 0].tolist()

# X_train, X_test, y_train, y_test = train_test_split(X.loc[:, feat20],
#                                                     y,
#                                                     test_size=0.25,
#                                                     stratify=y,
#                                                     random_state=2018)
labels = ["Persist-Tier1D", "Non-Tier1D"]

# X_train, y_train = SMOTE().fit_sample(X_train, y_train)
# print(X_train.shape)
# print(Counter(y_train).items())
# print(X_train.shape)
# print("Training size: %r" %X_train.shape[0])
# print("Test size: %r" %X_test.shape[0])
if True:
    file = "feat"
    for n in [5, 7, 10, 15, 20]:
        clfs = func.Clf(X.loc[:, feat[:n]], y, labels)
        clfs.runKfold(RandomForestClassifier, {"n_estimators": [500],
                                              'max_features': ['sqrt', 'log2', 0.2],
                                              'max_depth': [None, 3, 4, 5]}, True)
        clfs.runKfold(LinearSVC, {"dual": [True], "penalty": ['l2'], 'C': [0.01, 0.1, 1, 10, 100]}, True)
        clfs.runKfold(LinearSVC, {"dual": [False], "penalty": ['l2'], 'C': [0.01, 0.1, 1, 10, 100]}, True)
        clfs.runKfold(LinearSVC, {"dual": [False], "penalty": ['l1'], 'C': [0.01, 0.1, 1, 10, 100]}, True)
        clfs.runKfold(LogisticRegression, {"penalty": ['l1', 'l2'], 'dual': [False], "C": [0.01, 0.1, 1, 10, 100]}, True)
        clfs.runKfold(LogisticRegression, {"penalty": ['l2'], 'dual': [True], "C": [0.01, 0.1, 1, 10, 100]}, True)

        clfs.runKfold(KNeighborsClassifier, {"n_neighbors": [5, 7, 10], 'n_jobs': [-1]}, True)
        clfs.runKfold(AdaBoostClassifier, {"n_estimators": [500], 'learning_rate': [0.01, 0.1, 1]}, True)
        clfs.runKfold(RidgeClassifier, {"alpha": [1, 10, 100]}, True)

        clfs.runKfold(GaussianNB, {}, True)
        clfs.runKfold(GaussianNB, {}, False)

        clfs.runKfold(BernoulliNB, {}, True)
        clfs.runKfold(BernoulliNB, {}, False)

        # print(clfs.clf_eval)
        clfs.clf_eval.to_excel("report/feat_%r_eval700.xlsx" %(n))

else:
    file = "pca"
    clfs = func.Clf(X, y, labels)
    clfs.runKfold(RandomForestClassifier, {"n_estimators": [500],
                                          'max_features': ['sqrt', 'log2', 0.2],
                                          'max_depth': [None, 3, 4, 5]}, True, dr=True)
    clfs.runKfold(LinearSVC, {"dual": [True], "penalty": ['l2'], 'C': [0.01, 0.1, 1, 10, 100]}, True, dr=True)
    clfs.runKfold(LinearSVC, {"dual": [False], "penalty": ['l2'], 'C': [0.01, 0.1, 1, 10, 100]}, True, dr=True)
    clfs.runKfold(LinearSVC, {"dual": [False], "penalty": ['l1'], 'C': [0.01, 0.1, 1, 10, 100]}, True, dr=True)
    clfs.runKfold(LogisticRegression, {"penalty": ['l1', 'l2'], 'dual': [False], "C": [0.01, 0.1, 1, 10, 100]}, True, dr=True)
    clfs.runKfold(LogisticRegression, {"penalty": ['l2'], 'dual': [True], "C": [0.01, 0.1, 1, 10, 100]}, True, dr=True)

    clfs.runKfold(KNeighborsClassifier, {"n_neighbors": [5, 7, 10], 'n_jobs': [-1]}, True, dr=True)
    clfs.runKfold(AdaBoostClassifier, {"n_estimators": [500], 'learning_rate': [0.01, 0.1, 1]}, True, dr=True)
    clfs.runKfold(RidgeClassifier, {"alpha": [1, 10, 100]}, True, dr=True)

    clfs.runKfold(GaussianNB, {}, True, dr=True)
    clfs.runKfold(GaussianNB, {}, False, dr=True)

    clfs.runKfold(BernoulliNB, {}, True, dr=True)
    clfs.runKfold(BernoulliNB, {}, False, dr=True)

    # print(clfs.clf_eval)
    clfs.clf_eval.to_excel("report/%r_eval.xlsx" %(file))
