import pandas as pd
import numpy as np
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

from collections import Counter
from sklearn import svm, datasets
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from torch.autograd import Variable
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn import svm
from sklearn.svm import NuSVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
import time
import copy
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, DBSCAN
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, scale
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.cm as cm
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
import seaborn as sns

def completeRTN(row):
    tr = str(int(row["TRACKING_REGION_NUMBER"]))
    tn = str(int(row["TRACKING_NUMBER"]))
    if len(tn) < 7:
        tn = ("0" * (7 - len(tn))) + tn
    return tr + "-" + tn

def preprocess(df, coldef, type):
    columndef = pd.read_excel(coldef)
    cols = df.columns
    if type == '101191':
        # process extraction
        df["A1"] = (df["A1"] == "AM").astype(int)
        df["B6"] = (df["B6"] == "PRP").astype(int)

        df["D11"] = 0

        df["D4N"] = (df["D4M"] == "Y").astype(int)
        df["D4M"] = (df["D4L"] == "Y").astype(int)
        df["D4L"] = (df["D4K"] == "Y").astype(int)
        df["D4K"] = (df["D4J"] == "Y").astype(int)
        df["D4J"] = (df["D4I"] == "Y").astype(int)
        df["D4I"] = (df["D4H"] == "Y").astype(int)
        df["D4H"] = (df["D4G"] == "Y").astype(int)
        df["D4G"] = (df["D4F"] == "Y").astype(int)
        df["D4F"] = (df["D4E"] == "Y").astype(int)
        df["D4E"] = (df["D4D"] == "Y").astype(int)
        df['D4D'] = 0

        df["D9M"] = (df["D9L"] == "Y").astype(int)
        df["D9L"] = (df["D9J"] == "Y").astype(int)
        df["D9J"] = (df["D9K"] == "Y").astype(int)
        df["D9K"] = 0

        df["D10U"] = (df["D10R"] == "Y").astype(int)
        df["D10T"] = (df["D10Q"] == "Y").astype(int)
        df["D10R"] = (df["D10P"] == "Y").astype(int)
        df["D10P"] = 0
        df["D10Q"] = 0
        df["D10S"] = 0

        df["F12"] = (df["F12"] == "PRP").astype(int)

        df["G"] = df["G1"]
        df["G1"] = (df["G"] == "PRENOT").astype(int)
        df["G2"] = 0
        df["G3"] = (df["G"] == "ASSESS").astype(int)
        df["G4"] = (df["G"] == "APORAL").astype(int)
        df["G5"] = (df["G"] == "APORMD").astype(int)
        df["G6"] = (df["G"] == "REQPLN").astype(int)
        df["G7"] = (df["G"] == "INTENT").astype(int)
        df["G8"] = 0
        df["G9"] = 0
        df.drop("G", axis=1, inplace=True)

        df["G11A"] = (df["G8A"] == 'Y').astype(int)
        df["G11B"] = (df["G8B"] == "Y").astype(int)
        df.drop("G8A", axis=1, inplace=True)
        df.drop("G8B", axis=1, inplace=True)

        df["LUST_ELIGIBLE_NO"] = (df["LUST_ELIGIBLE"] == "N").astype(int)
        df["LUST_ELIGIBLE_UNKNOWN"] = (df["LUST_ELIGIBLE"] == "U").astype(int)
        df["LUST_ELIGIBLE_YES"] = (df["LUST_ELIGIBLE"] == "Y").astype(int)
        df.drop("LUST_ELIGIBLE", axis=1, inplace=True)
    elif type == "101592":
        # process extraction
        df["A1"] = (df["A1"] == "AM").astype(int)

        df["B6"] = (df["B6"] == "PRP").astype(int)

        df["D10U"] = (df["D10R"] == "Y").astype(int)
        df["D10T"] = (df["D10Q"] == "Y").astype(int)
        df["D10R"] = (df["D10P"] == "Y").astype(int)
        df["D10P"] = 0
        df["D10Q"] = 0
        df["D10S"] = 0

        df["F12"] = (df["F12"] == "PRP").astype(int)

        df["G"] = df["G1"]
        df["G1"] = (df["G"] == "IRA-PRENOT").astype(int)
        df["G2"] = (df["G"] == "IRA-NOAPP").astype(int)
        df["G3"] = (df["G"] == "IRA-ASSESS").astype(int)
        df["G4"] = (df["G"] == "IRA-APORAL").astype(int)
        df["G5"] = (df["G"] == "IRA-APORMD").astype(int)
        df["G6"] = (df["G"] == "IRA-REQPLN").astype(int)
        df["G7"] = (df["G"] == "URAM-INTENT").astype(int)
        df["G8"] = (df["G"] == "IRA-D-APORAL").astype(int)
        df["G9"] = (df["G"] == "IRA-D-WORKST").astype(int)
        df.drop("G", axis=1, inplace=True)

        df["LUST_ELIGIBLE_NO"] = (df["LUST_ELIGIBLE"] == "N").astype(int)
        df["LUST_ELIGIBLE_UNKNOWN"] = (df["LUST_ELIGIBLE"] == "U").astype(int)
        df["LUST_ELIGIBLE_YES"] = (df["LUST_ELIGIBLE"] == "Y").astype(int)
        df.drop("LUST_ELIGIBLE", axis=1, inplace=True)
    elif type == "101607":
        df["A1"] = (df["A1AM"] == "Y").astype(int)
        df["B6"] = (df["B6OTHER"] == "Y").astype(int)
        df["F12"] = (df["F12OTHER"] == "Y").astype(int)

        df["LUST_ELIGIBLE_NO"] = (df["LUST_ELIGIBLE"] == "N").astype(int)
        df["LUST_ELIGIBLE_UNKNOWN"] = (df["LUST_ELIGIBLE"] == "U").astype(int)
        df["LUST_ELIGIBLE_YES"] = (df["LUST_ELIGIBLE"] == "Y").astype(int)
        df.drop("LUST_ELIGIBLE", axis=1, inplace=True)

    # standard processing
    for col in cols:
        if col == "RTN":
            continue
        # print(col)
        proc = columndef[columndef["feature"] == col]["proc"].values[0]

        # deal with Y/N
        if proc == "transyo10":
            df[col].replace(to_replace={"Y": 1, "Off": 0, "off": 0}, inplace=True)
            df[col] = df[col].astype(int)
        elif proc == "translate10":
            df[col].replace(to_replace={"Y": 1, "N": 0}, inplace=True)
            df[col] = df[col].astype(int)
        # drop column
        elif proc == "drop":
            df.drop(col, axis=1, inplace=True)
        # to be discussed
        elif proc == "?":
            df.drop(col, axis=1, inplace=True)
        # mostly float, and some str
        elif proc == "floatandstr":
            # df.drop(col, axis=1, inplace=True)
            df[col] = df[col].apply(extractlargenumber)
        # change the type to float
        elif proc == "float":
            df[col] = df[col].astype(float)
        else:
            pass

def prepmissing(df):
    attributes = df.columns
    nominalvalues = {}

    # df = df.replace('N/A', np.NaN)
    # df = df.replace('?', np.NaN)
    for col in df.columns:
        # deal with missing values
        if sum(pd.isnull(df[col])) != 0 or sum(df[col].isin(["?"])) > 0:
            print("%r column (type: %r): %r null" %(col, df[col].dtype, sum(pd.isnull(df[col]))))
#             if df[col].dtype == "object":
#                 md = df[df[col] != np.NaN][col].mode()[0]
#                 df[col] = df[col].replace(np.NaN, md)
#             else:
#                 mn = df[col].astype(float).mean()
#                 df[col] = df[col].replace(np.NaN, mn)

def processChemicals(df):
    return None

def extractlargenumber(cell):
    cell = str(cell)
    if cell == "":
        return 0

    strs = cell.split(" ")
    number = 0.
    for s in strs:
        try:
            num = float(s.replace(",", ""))
            if num > number:
                number = num
        except ValueError:
            pass
    return number

def processtiers(df):
    dftier = pd.DataFrame(columns=["RTN", "Tier"])
    for rtn in df["RTN"].unique().tolist():
        if len(df[df['RTN'] == rtn]) > 1:
            flag = df[df['RTN'] == rtn]['newtc'].sum() + df[df['RTN'] == rtn]['revisedtc'].sum()
            if flag > 0:
                dftier = dftier.append({"RTN": rtn, "Tier": 1}, ignore_index=True)
            else:
                dftier = dftier.append({"RTN": rtn, "Tier": 2}, ignore_index=True)
        else:
            flag = (df[df['RTN'] == rtn]['newtc'].values[0]) or (df[df['RTN'] == rtn]['revisedtc'].values[0])
            if flag:
                dftier = dftier.append({"RTN": rtn, "Tier": 1}, ignore_index=True)
            else:
                dftier = dftier.append({"RTN": rtn, "Tier": 2}, ignore_index=True)
    return dftier

# def isTier1D(row):
#     t1 = row["Notification"]
#     t2 = row["Phase1Cs"]
#     t3 = row["End Date"]
#     t4 = row["Regulatory End Date"]
#     status = row["Status"]
#     if not pd.isnull(t2): # those with phase 1 date
#         T = t2 - t1 # the length of phase 1
#     elif not pd.isnull(t3):
#         T = t3 - t1 # the length of the whole project
#     else: # those without phase 1 date
#         return True
#     if T.days <= 365: # 365 + 7 = 372
#         return False # not Tier 1D
#     else:
#         return True # Tier 1D
def isTier1D(row):
    t = row["length"]
    if t < 400:
        return "Non-Tier1D"
    else:
        return "Tier1D"

def isPTier1D(row):
    t = row["length"]
    if t < 700:
        return "Non-Tier1D"
    else:
        return "Persist-Tier1D"

def daylength(row):
    t1 = row["Notification"]
    t2 = row["Phase1Cs"]
    # t3 = row["End Date"]
    t4 = row["Regulatory End Date"]
    raonr = row["RaoNr"]
    status = row["Status"]
    if status == "TIER1D":
        return 999
    else:
        if status in ["RAO", "PSC", "PSNC", "SPECPR", "TMPS", "RAONR"]:
            T = t4 - t1
        elif status in ["REMOPS", "ROSTRM", "TCLASS", "TIERI", "TIERII"]:
            if not pd.isnull(t2):
                T = t2 - t1
            else:
                t5 = raonr.split(" ")[1]

                m, d, y = t5.split("/")
                t5 = datetime(int(y), int(m), int(d))

                T = t5 - t1
    return T.days

def runclassifier(clf, X_train, y_train, X_test, y_test, labels):
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    plot_confusion_matrix(confusion_matrix(y_test, y_predict, labels=labels), classes=labels)
    print(classification_report(y_test, y_predict, labels=labels, target_names=labels))

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 5) # original image size: 143
        self.pool1 = nn.AvgPool1d(3, stride=2) # window size: 2, stride: 2
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.pool2 = nn.AvgPool1d(2, stride=2) # window size: 2, stride: 2
        self.fc1 = nn.Linear(32 * 32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

class ANet(nn.Module):
    # def __init__(self, input_dim, hidden_layers_dim, output_dim, activations, criterion="CrossEntropy", optimizer="sgd", lr=0.01):
    #     super(ANet, self).__init__()
    #     self.linear_layers = nn.ModuleList()
    #     self.activation_layers = nn.ModuleList()
    #
    #     for i, hidden_dim in enumerate(hidden_layers_dim):
    #         if i == 0:
    #             self.activation_layers.append(input_dim, hidden_dim)
    #         else:
    #             self.linear_layers.append(nn.Linear(self.linear_layers[-1].out_features, hidden_dim))
    #         self.activation_layers.append(get_)
    #
    # def forward(self, x):
    #     x = x.view(-1, 143)
    #     x = F.sigmoid(self.fc1(x))
    #     x = F.sigmoid(self.fc2(x))
    #     x = F.sigmoid(self.fc3(x))
    #     x = F.sigmoid(self.fc4(x))
    #     x = F.sigmoid(self.fc5(x))
    #     x = F.sigmoid(self.fc6(x))
    #     x = F.sigmoid(self.fc7(x))
    #     x = self.fc8(x)
    #
    #     return x
    def __init__(self):
        super(convNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

class Clf():
    def __init__(self, X, y, labels):
        self.X = X
        self.y = y
        self.labels = labels
        self.clf_eval = pd.DataFrame(data=None, columns=['clf', 'params', 'smote', 'Trecall', 'Tprecision', 'Tf1', 'Frecall', 'Fprecision', 'Ff1'])
        # self.clf_eval = {}
        # self.clf_eval['clf'] = str(self.clf).split('(')[0]
        # self.clf_eval['smote'] = self.smote

    def runKfold(self, classifier, param, smote=False, dr=False, drp=10, k=4):
        skf = StratifiedKFold(n_splits=k, random_state=122, shuffle=True)
        param_grid = ParameterGrid(param)
        for params in param_grid:
            Trecalls = []
            Tprecisions = []
            Frecalls = []
            Fprecisions = []
            Tf1s = []
            Ff1s = []
            for train_index, test_index in skf.split(self.X, self.y):

                X_train, X_test = self.X.iloc[train_index, :], self.X.iloc[test_index, :]
                y_train, y_test = self.y[train_index], self.y[test_index]

                if dr:
                    pca = PCA(n_components=drp)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)

                if smote:
                    X_train, y_train = SMOTE().fit_sample(X_train, y_train)
                # train
                clf = classifier(**params)
                clf.fit(X_train, y_train)

                y_predict = clf.predict(X_test)

                metrics = precision_recall_fscore_support(y_test, y_predict, labels=self.labels)

                Tprecision, Fprecision = metrics[0][0], metrics[0][1]
                Trecall, Frecall = metrics[1][0], metrics[1][1]
                Tf1, Ff1 = metrics[2][0], metrics[2][1]

                Trecalls.append(Trecall)
                Tprecisions.append(Tprecision)
                Frecalls.append(Frecall)
                Fprecisions.append(Fprecision)
                Tf1s.append(Tf1)
                Ff1s.append(Ff1)
                # print(classification_report(y_test, y_predict, labels=[1, 0], target_names=['Tier 1D','other']))
                # report = classification_report(y_test, y_predict, labels=[True, False], target_names=['Tier 1D','other'])

            self.clf_eval = self.clf_eval.append({'clf': str(clf).split('(')[0],
                                  'params': params,
                                  'smote': smote,
                                  'Trecall': np.mean(Trecalls),
                                  'Tprecision': np.mean(Tprecisions),
                                  'Tf1': np.mean(Tf1s),
                                  'Frecall': np.mean(Frecalls),
                                  'Fprecision': np.mean(Fprecisions),
                                  'Ff1': np.mean(Ff1s)}, ignore_index=True)
            print("%r %r" %(str(clf).split('(')[0], params))
            print("Done")
            # self.clf_eval['params'] = params
            # self.clf_eval['recall'] = np.mean(recalls)

def distance(inst1, inst2):
    try:
        inst1 = np.array(inst1)
        inst2 = np.array(inst2)

        if len(inst1) != len(inst2):
            raise NameError("Not the same dimension.")

        total = 0
        for i in range(len(inst1)):
            total += (inst1[i] - inst2[i])**2
        return total
    except ValueError:
        return 100

class Clustering():
    def __init__(self, df, ndf, labelstr, labelindex, display):
        """
        :param k: int, number of cluster
        :param df: dataframe, raw dataframe
        :param ndf: dataframe, the dataframe used in model
        :param labelstr: str, label in the dataframe
        :param labelindex: int, index of label in the dataframe
        :param display: int, 0-display all label, 1-display count label
        """
        self.df = df
        self.ndf = ndf
        self.labelindex = labelindex

        self.display = display

        self.label = copy.deepcopy(df[labelstr])
        if display == 1:
            print("Labels distribution: ")
            print(self.label.groupby(self.label).count())

            #labels = self.label.unique().tolist()

            #for i in range(len(self.label)):
            #    self.label[i] = labels.index(self.label[i])


    def KMeans_fit_predict(self, k, preproc=0, seed=32):
        """
        :param k: number of clusters
        :param preproc: 0-no preprocess, 1-normalize, 2-standardiza
        :return:
        """
        if preproc == 1:
            ndf = pd.DataFrame(normalize(self.ndf, axis=0, copy=True))
        elif preproc == 2:
            ndf = pd.DataFrame(scale(self.ndf, axis=0, copy=True))
        else:
            ndf = self.ndf

        summary = "Summary: \n"
        summary += "KMeans, k=%r \n" % k

        kmeans = KMeans(n_clusters=k, random_state=seed)
        y_pred = kmeans.fit_predict(ndf)
        sse = kmeans.inertia_

        summary += "SSE: %r \n" %sse

        clusters = dict.fromkeys([i for i in range(k)], None)
        for i in range(len(clusters)):
            clusters[i] = []
        for label in range(len(y_pred)):
            clusters[y_pred[label]].append(label)

        categories = {}
        for i in range(k):
            categories[i] = [self.df.iloc[m, self.labelindex] for m in clusters.get(i)]
        for d in range(len(categories)):
            if self.display == 0:
                summary += "Cluster %r: %r\n" %(d, ", ".join(categories[d]))
            else:
                summary +=  "Cluster %r: %r\n" % (d, Counter(categories[d]))

        cluster_center_ = {}
        for i in range(len(kmeans.cluster_centers_)):
            cluster_center_[i] = kmeans.cluster_centers_[i]

        return categories, y_pred, cluster_center_, sse, summary

    def AgglomerativeClustering_fit_predict(self, k, affinity="euclidean", linkage="complete", preproc=0, seed=32):
        if preproc == 1:
            ndf = pd.DataFrame(normalize(self.ndf, axis=0, copy=True))
        elif preproc == 2:
            ndf = pd.DataFrame(scale(self.ndf, axis=0, copy=True))
        else:
            ndf = self.ndf

        summary = "Summary: \n"
        summary += "AgglomerativeClustering, k=%r \n" % k

        ac = AgglomerativeClustering(n_clusters=k, affinity=affinity, linkage=linkage)
        y_pred = ac.fit_predict(ndf)

        clusters = dict.fromkeys([i for i in range(k)], None)
        for i in range(len(clusters)):
            clusters[i] = []
        for label in range(len(y_pred)):
            clusters[y_pred[label]].append(label)
        # calculate the center
        cluster_center_ = {}
        for i in range(k):
            cluster_center_[i] = ndf.iloc[clusters[i], :].mean().tolist()
        sse = 0
        for i in range(k):
            for iid in clusters[i]:
                sse += distance(ndf.iloc[iid, :], cluster_center_[i])

        summary += "SSE: %r \n" % sse

        categories = {}
        for i in range(k):
            categories[i] = [self.df.iloc[m, self.labelindex] for m in clusters.get(i)]
        for d in range(len(categories)):
            if self.display == 0:
                summary += "Cluster %r: %r\n" %(d, ", ".join(categories[d]))
            else:
                summary +=  "Cluster %r: %r\n" % (d, Counter(categories[d]))

        return categories, y_pred, cluster_center_, sse, summary

    def DBSCAN_fit_predict(self, eps, min_s=1, preproc=0, seed=32):
        if preproc == 1:
            ndf = pd.DataFrame(normalize(self.ndf, axis=0, copy=True))
        elif preproc == 2:
            ndf = pd.DataFrame(scale(self.ndf, axis=0, copy=True))
        else:
            ndf = self.ndf

        summary = "Summary: \n"
        summary += "DBSCAN \n"

        dbs = DBSCAN(eps=eps, min_samples=min_s, n_jobs=-1)
        y_pred = dbs.fit_predict(ndf)

        clusters = dict.fromkeys([i for i in set(y_pred)], None)
        for i in set(y_pred):
            clusters[i] = []
        for label in range(len(y_pred)):
            clusters[y_pred[label]].append(label)

        cluster_center_ = {}
        for i in set(y_pred):
            cluster_center_[i] = ndf.iloc[clusters[i], :].mean().tolist()
        sse = 0
        for i in set(y_pred):
            for iid in clusters[i]:
                sse += distance(ndf.iloc[iid, :], cluster_center_[i])

        summary += "SSE: %r \n" %sse

        # summary += "Std: %r \n" %

        categories = {}
        for i in set(y_pred):
            categories[i] = [self.df.iloc[m, self.labelindex] for m in clusters.get(i)]

        if -1 not in cluster_center_.keys():
            for d in range(len(categories)):
                if self.display == 0:
                    summary += "Cluster %r: %r\n" %(d, ", ".join(categories[d]))
                else:
                    summary +=  "Cluster %r: %r\n" % (d, Counter(categories[d]))
        else:
            for d in range(-1, len(categories) - 1):
                if self.display == 0:
                    summary += "Cluster %r: %r\n" %(d, ", ".join(categories[d]))
                else:
                    summary +=  "Cluster %r: %r\n" % (d, Counter(categories[d]))

        return categories, y_pred, cluster_center_, sse, summary

    def externalEval(self, y_pred, true_label):
        true_label = np.array(true_label)
        n_cluster = len(set(true_label))
        y_pred_modi = y_pred.copy()
        result = [[] for i in range(len(set(y_pred)))]
        for i in range(len(y_pred)):
            result[y_pred[i]].append(i)
        dict1 = dict.fromkeys([i for i in range(n_cluster)], None)
        for i in list(dict1.keys()):
            dict1[i] = []
        nummostnum = 0
        for i in range(len(result)):
            if len(true_label[result[i]]) > 0:
                mostnum = Counter(true_label[result[i]]).most_common(1)[0][0]
                nummostnum += Counter(true_label[result[i]]).most_common(1)[0][1]
                dict1[mostnum] += (result[i])
        for r in list(dict1.keys()):
            for i in dict1[r]:
                y_pred_modi[i] = r
        nmi = normalized_mutual_info_score(true_label, y_pred)
        purity = nummostnum / len(y_pred_modi)
        fowlkes_mallows = fowlkes_mallows_score(true_label, y_pred_modi)
        return nmi, purity, fowlkes_mallows

    def silhouette(self, range_n_clusters, cluster_labelss):
        X = self.ndf
        for n_cluster in range_n_clusters:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(12, 6)

            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(X) + (n_cluster + 1) * 10])

            cluster_labels = cluster_labelss[n_cluster-2]

            # categories, cluster_labels, cluster_centers_, summary = self.kmeans_fit_predict(n_cluster, preproc)

            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_cluster,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_cluster):
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.spectral(float(i) / n_cluster)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # mds
            # mds
            similarities = euclidean_distances(X)
            mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=random_state,
                               dissimilarity="precomputed", n_jobs=1)
            pos = mds.fit(similarities).embedding_
            df_pos = pd.DataFrame(pos, columns=["comp1", "comp2"])
            df_pos["pred"] = cluster_labels

            for i in range(n_cluster):
                color = cm.spectral(float(i) / n_cluster)
                ax2.scatter(df_pos[df_pos["pred"] == i].iloc[:, 0], df_pos[df_pos["pred"] == i].iloc[:, 1], c=color)

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st MDS feature")
            ax2.set_ylabel("Feature space for the 2nd MDS feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_cluster),
                         fontsize=14, fontweight='bold')
            # end mds
            plt.show()

def report(nX, y):
    nmetrics = pd.DataFrame(data=None, columns=['clf', 'params', 'smote', 'recall', 'precision'])
    nclf1 = clf(RandomForestClassifier, nX, y, {"n_estimators": [200], 'max_features': ['sqrt', 'log2', 0.2]}, True)
    nclf1.runKfold()
    nmetrics = nmetrics.append(nclf1.clf_eval, ignore_index=True)
    nclf2 = clf(RandomForestClassifier, nX, y, {"n_estimators": [200], 'max_features': ['sqrt', 'log2', 0.2]}, False)
    nclf2.runKfold()
    nmetrics = nmetrics.append(nclf2.clf_eval, ignore_index=True)
    nclf7 = clf(LinearSVC, nX, y, {"dual": [False], "penalty": ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}, True)
    nclf7.runKfold()
    nmetrics = nmetrics.append(nclf7.clf_eval, ignore_index=True)
    nclf7 = clf(LinearSVC, nX, y, {"dual": [False], "penalty": ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}, False)
    nclf7.runKfold()
    nmetrics = nmetrics.append(nclf7.clf_eval, ignore_index=True)
    nclf9 = clf(LinearSVC, nX, y, {"dual": [True], "penalty": ['l2'], 'C': [0.01, 0.1, 1, 10, 100]}, True)
    nclf9.runKfold()
    nmetrics = nmetrics.append(nclf9.clf_eval, ignore_index=True)
    nclf9 = clf(LinearSVC, nX, y, {"dual": [True], "penalty": ['l2'], 'C': [0.01, 0.1, 1, 10, 100]}, False)
    nclf9.runKfold()
    nmetrics = nmetrics.append(nclf9.clf_eval, ignore_index=True)
    nclf10 = clf(LogisticRegression, nX, y, {"penalty": ['l1', 'l2'], 'dual': [False], "C": [0.01, 0.1, 1, 10, 100]}, True)
    nclf10.runKfold()
    nmetrics = nmetrics.append(nclf10.clf_eval, ignore_index=True)
    nclf10 = clf(LogisticRegression, nX, y, {"penalty": ['l1', 'l2'], 'dual': [False], "C": [0.01, 0.1, 1, 10, 100]}, False)
    nclf10.runKfold()
    nmetrics = nmetrics.append(nclf10.clf_eval, ignore_index=True)
    nclf11 = clf(LogisticRegression, nX, y, {"dual": [True], "penalty": ['l2'], 'C': [0.01, 0.1, 1, 10, 100]}, True)
    nclf11.runKfold()
    nmetrics = nmetrics.append(nclf11.clf_eval, ignore_index=True)
    nclf11 = clf(LogisticRegression, nX, y, {"dual": [True], "penalty": ['l2'], 'C': [0.01, 0.1, 1, 10, 100]}, False)
    nclf11.runKfold()
    nmetrics = nmetrics.append(nclf11.clf_eval, ignore_index=True)
    return nmetrics
