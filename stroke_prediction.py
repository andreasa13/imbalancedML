#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, plot_roc_curve
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import EasyEnsembleClassifier

from sklearn import svm, datasets
from sklearn.cluster import KMeans

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.under_sampling import CondensedNearestNeighbour

from sklearn.metrics import roc_curve
from imblearn.over_sampling import BorderlineSMOTE

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import ADASYN

from imblearn.under_sampling import TomekLinks
from sklearn.metrics import plot_confusion_matrix

import warnings
from sklearn.exceptions import DataConversionWarning

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve

from sklearn.linear_model import LogisticRegression

# import matplotlib.pyplot as plt
# from sklearn.pipeline import Pipeline

from imblearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import average_precision_score


def get_categorical_features(df):
    types = df.columns.to_series().groupby(df.dtypes).groups
    dict_of_types = {k.name: v for k, v in types.items()}
    objects_list = dict_of_types.get('object').tolist()
    return objects_list


def one_hot_encoding(df, col_name):
    one_hot = pd.get_dummies(df[col_name])
    df = df.drop(col_name, axis=1)
    df = df.join(one_hot)
    return df


def model_evaluation(X_train, X_test, y_train, y_test):

    clf = RandomForestClassifier()
    # clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report_imbalanced(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print('Accuracy: %.2f' % accuracy)
    print('Balanced accuracy: %.2f' % balanced_acc)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    print("fpr: ", fpr)
    print("tpr: ", tpr)
    print(thresholds)

    plot_precision_recall_curve(clf, X_test, y_test, pos_label=1)
    plot_roc_curve(clf, X_test, y_test)
    plt.show()



    #     print('Recall: %.2f' % recall_score(y_test, y_pred))
    #     print('Precision: %.2f' % precision_score(y_test, y_pred))
    #     print('F1: %.2f' % f1_score(y_test, y_pred))


def scatter2D(x, y):
    x['stroke'] = y
    df = x
    fig, ax3 = plt.subplots(figsize=(15, 8))

    colors = {0: 'blue', 1: 'red'}
    sizevalues = {0: 5, 1: 20}
    alphavalues = {0: 0.4, 1: 0.8}
    ax3.scatter(df['age'], df['avg_glucose_level'],
               c=df['stroke'].apply(lambda x: colors[x]),
               s=df['stroke'].apply(lambda x: sizevalues[x]),
               alpha= .5)

    plt.show()

def scatter3D(x, y, cols):
    print(cols)
    if 'stroke' in cols:
        cols.remove('stroke')
    df2 = pd.DataFrame(x)
    df2.columns = cols
    df2['stroke'] = y

    fig3 = plt.figure(1)
    ax3 = Axes3D(fig3)

    # positive class
    positive = pd.DataFrame(df2[df2['stroke'] == 1], columns=df2.columns)
    print(positive.columns)
    positive.columns = df2.columns
    ppx = positive['age']
    ppy = positive['hypertension']
    ppz = positive['avg_glucose_level']

    # negative class
    negative = df2[df2['stroke'] == 0]
    negative.columns = df2.columns
    pnx = negative['age']
    pny = negative['hypertension']
    pnz = negative['avg_glucose_level']

    # plotting
    ax3.scatter(pnx, pny, pnz, label='Class 2', c='blue')
    ax3.scatter(ppx, ppy, ppz, label='Class 1', c='red')

    ax3.set_xlabel('age')
    ax3.set_ylabel('hypertension')
    ax3.set_zlabel('avg_glucose_level')

    ax3.set_title('SMOTE')
    plt.title('SMOTE')

    plt.show()


def scatter_plot(x, y):

    # normalization
    scaler = StandardScaler()
    x = scaler.fit_transform(x)


    # 2-dimensional PCA
    pca = PCA(n_components=3)
    pca.fit(x)
    x = pd.DataFrame(pca.transform(x))
    print(pca.explained_variance_ratio_)

    # φτιαξε αυτο για να μη χαλαει το x_train πριν την εκπαιδευση του RF
    x['stroke'] = y
    pcaDF = x
    pcaDF.columns = ['pc1', 'pc2', 'pc3', 'stroke']

    # 3D plot
    fig3 = plt.figure(1)
    ax3 = Axes3D(fig3)

    # positive class
    positive = pcaDF[pcaDF['stroke'] == 1]
    ppx = positive['pc1']
    ppy = positive['pc2']
    ppz = positive['pc3']

    # negative class
    negative = pcaDF[pcaDF['stroke'] == 0]
    pnx = negative['pc1']
    pny = negative['pc2']
    pnz = negative['pc3']

    # ax3.set_xlim3d(-4, 4)
    # ax3.set_ylim3d(-4, 4)
    # ax3.set_zlim3d(-4, 4)

    # plotting
    ax3.scatter(pnx, pny, pnz, label='Class 2', c='blue')
    ax3.scatter(ppx, ppy, ppz, label='Class 1', c='red')
    
    plt.show()


def kNNUndersampling(X_train, X_test, y_train, y_test):
    # define the undersampling method
    print('UNDERSAMPLING: ')
    undersample = CondensedNearestNeighbour(n_neighbors=1)
    X_train, y_train = undersample.fit_resample(X_train, y_train)
    scatter_plot(X_train, y_train)
    model_evaluation(X_train, X_test, y_train, y_test)


def bSMOTE(X_train, y_train, i):
    # borderline SMOTE
    from imblearn.over_sampling import SMOTE, KMeansSMOTE
    sm = BorderlineSMOTE()
    # sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)
    # print(y_train['stroke'].value_counts())
    if i == 0:
        # scatter_plot(X_train, y_train)
        scatter3D(X_train, y_train, cols)

    # remove TomekLinks
    tl = TomekLinks(sampling_strategy='auto')
    X_train, y_train = tl.fit_resample(X_train, y_train)
    # print(y_train['stroke'].value_counts())
    if i == 0:
        # scatter_plot(X_train, y_train)
        scatter3D(X_train, y_train, cols)
    return X_train, y_train


def adaptiveSynthetic(X_train, X_test, y_train, y_test):
    ada = ADASYN()
    X_train, y_train = ada.fit_resample(X_train, y_train)
    scatter_plot(X_train, y_train)
    model_evaluation(X_train, X_test, y_train, y_test)


def kMeansRos(X_train):
    from sklearn.cluster import SpectralClustering

    kmeans = KMeans(n_clusters=2).fit(X_train)
    # kmeans = SpectralClustering(n_clusters=2).fit(X_train)
    y_labels = kmeans.labels_

    countPos = np.count_nonzero(y_labels == 0)
    print(countPos)
    countNeg = np.count_nonzero(y_labels == 1)
    print(countNeg)

    if countPos < countNeg:
        indices_one = y_labels == 1
        indices_zero = y_labels == 0
        y_labels[indices_one] = 0  # replacing 1s with 0s
        y_labels[indices_zero] = 1  # replacing 0s with 1s

    # scatter_plot(X_train, y_labels)

    ros = RandomOverSampler()
    X_train, y_train = ros.fit_resample(X_train, y_labels)

    return X_train, y_train


def plotlyBarChart(df):

    import plotly.graph_objects as go


    x = df.index.values.tolist()
    y = df.tolist()

    fig = go.Figure(data=[go.Bar(
        x=x, y=y,
        text=y,
        textposition='auto',
    )])

    fig.show()


if __name__ == '__main__':

    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    # load dataset
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    print(df.describe())

    print("Total number of entries/cases: ", len(df))

    # positive/negative
    pos_df = df[df["stroke"] == 1]
    neg_df = df[df["stroke"] == 0]
    print('positive cases: ', len(pos_df))
    print('positive cases: ', len(neg_df))

    # Drop id column & Drop Missing values
    df = df.drop(['id'], axis=1)
    print(df.isnull().sum())
    df = df.dropna()
    print("After removal of missing values: ", len(df))

    # get categorical features
    categorical_features = get_categorical_features(df)
    print('categorical features: ', categorical_features)

    # one hot encoding for categorical features
    for feature in categorical_features:
        df = one_hot_encoding(df, feature)

    print(df.columns)
    cols = df.columns.tolist()

    count_cases = df['stroke'].value_counts()
    print(type(count_cases))
    print(count_cases)
    ax = count_cases.plot.bar(rot=0, title='Number of cases per class')
    ax.set_xlabel('stroke prediction')
    ax.set_ylabel('# of cases')
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.show()

    corrDF = df.drop(["stroke"], axis=1).apply(lambda x: x.corr(df.stroke.astype('category').cat.codes))
    print(type(corrDF))
    print(corrDF.abs().sort_values(ascending=False))

    scoring = ['accuracy', 'balanced_accuracy']

    X = df.drop(['stroke'], axis=1)
    y = df[['stroke']]

    folds = StratifiedKFold(n_splits=10, shuffle=True)

    tprs = []
    aucs = []
    acc = []
    bacc = []

    aucs2 = []
    precs = []
    recs = []
    ap = []

    mean_fpr = np.linspace(0, 1, 100)
    # mean_pr = np.linspace(0, 1, 100)
    mean_rec = np.linspace(0, 1, 100)

    figROC = plt.figure(2)
    axROC = figROC.add_subplot(111)

    figPR = plt.figure(3)
    axPR = figPR.add_subplot(111)

    for i, (train, test) in enumerate(folds.split(X, y)):

        X_train, y_train = X.iloc[train], y.iloc[train]
        X_test, y_test = X.iloc[test], y.iloc[test]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # scaler = MinMaxScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        if i == 0:
            # scatter_plot(X_train, y_train)
            scatter3D(X_train, y_train, cols)

        # sm = BorderlineSMOTE(sampling_strategy='all')
        # sm = SMOTE()
        # X_train, y_train = sm.fit_resample(X_train, y_train)
        X_train, y_train = kMeansRos(X_train)
        # from imblearn.over_sampling import KMeansSMOTE
        # sm = KMeansSMOTE(k_neighbors=2)
        # X_train, y_train = bSMOTE(X_train, y_train, i)

        # from imblearn.under_sampling import NearMiss, RandomUnderSampler
        # rus = RandomUnderSampler()
        # X_train, y_train = rus.fit_resample(X_train, y_train)

        if i == 0:
            # scatter_plot(X_train, y_train)
            scatter3D(X_train, y_train, cols)

        # clf = SVC()
        # clf = LogisticRegression()
        clf = RandomForestClassifier(max_depth=3)

        clf.fit(X_train, y_train)
        # scatter_plot(X.iloc[train], y.iloc[train])
        viz = plot_roc_curve(clf, X_test, y_test,
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=axROC)

        prc = plot_precision_recall_curve(clf, X_test, y_test,
                                           name='PR fold {}'.format(i),
                                           alpha=0.3, lw=1, ax=axPR, pos_label=1)


        # precision interpolation
        precInterp = np.interp(mean_rec, prc.recall[::-1], prc.precision[::-1])
        precInterp[0] = 1.0
        precs.append(precInterp)
        ap.append(prc.average_precision)

        # tpr interpolation
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        # Accuracy & Balanced Accuracy
        y_pred = clf.predict(X_test)
        acc.append(accuracy_score(y_test, y_pred))
        bacc.append(balanced_accuracy_score(y_test, y_pred))


    axROC.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    mean_prec = np.mean(precs, axis=0)
    mean_prec[-1] = 0.0
    mean_ap = np.mean(ap)
    std_ap = np.std(ap)

    axROC.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    axPR.plot(mean_rec, mean_prec, color='b',
               label = r'Mean PR (AP = %0.2f $\pm$ %0.2f)' % (mean_ap, std_ap),
               lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    std_prec = np.std(precs, axis=0)
    prec_upper = np.minimum(mean_prec + std_prec, 1)
    prec_lower = np.maximum(mean_prec - std_prec, 0)

    axROC.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    axPR.fill_between(mean_rec, prec_lower, prec_upper, color='grey', alpha=.2,
                       label=r'$\pm$ 1 std. dev.')

    axROC.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="ROC curve & AUC")

    axPR.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Precision-Recall curves & Average Precision")

    axROC.legend(loc="lower right")
    figROC.savefig('ROC_CV.png')
    figROC.show()

    axPR.legend(loc="upper right")
    figPR.savefig('PR_CV.png')
    figPR.show()

    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(acc), np.std(acc)))
    print('Mean Balanced Accuracy: %.3f (%.3f)' % (np.mean(bacc), np.std(bacc)))
    print('AUC %.3f (%.3f)' % (np.mean(aucs), np.std(aucs)))


    # print('NO PREPROCESSING: ')
    # scatter_plot(X_train, y_train)
    # # scatter2D(X, y)
    # print(y_train['stroke'].value_counts())
    # model_evaluation(X_train, X_test, y_train, y_test)

    # ros = RandomOverSampler()
    # X_ros, y_ros = ros.fit_resample(X_train, y_train)
    # scatter_plot(X_ros, y_ros)
    # print(y_ros['stroke'].value_counts())
    # model_evaluation(X_ros, X_test, y_ros, y_test)
    #
    # rus = RandomUnderSampler()
    # X_rus, y_rus = rus.fit_resample(X_train, y_train)
    # scatter_plot(X_rus, y_rus)
    # print(y_ros['stroke'].value_counts())
    # model_evaluation(X_rus, X_test, y_rus, y_test)

    # print('Borderline SMOTE & Tomek Links')
    # brSMOTE(X_train, X_test, y_train, y_test)

    # print('KMeans & Random Oversampling')
    # kMeansRos(X_train, X_test, y_train, y_test)

    # plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
    # plt.show()



