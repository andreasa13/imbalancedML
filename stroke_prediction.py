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


def scatter_plot(x, y):

    # normalization
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

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

    # plotting
    ax3.scatter(pnx, pny, pnz, label='Class 2', c='blue')
    ax3.scatter(ppx, ppy, ppz, label='Class 1', c='red')
    
    plt.show()


def compute_accuracy(TP, FP, TN, FN):
    print()


# aggregate predictions of each fold
def aggregate_labels(labels_list):
    y = []
    for l in labels_list:
        y.extend(l)
    return y
    
    
def perf_measure(y_actual, y_pred):
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i] == y_pred[i] == 0:
            TP += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == 1:
            TN += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FN += 1

    return TP, FP, TN, FN


def evaluate_model(X, y, oversampling):
    skf = StratifiedKFold(10, shuffle=True, random_state=42)

#     predicted_targets = np.array([])
#     actual_targets = np.array([])

#     tp_list = []
#     fp_list = []
#     tn_list = []
#     fn_list = []

    y_test_agg_list = []
    y_pred_agg_list = []
    for train_index, test_index in skf.split(X, y):
#             print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
            pos_df = y_test[y_test["stroke"] == 1]
            neg_df = y_test[y_test["stroke"] == 0]

            print('positive cases: ', len(pos_df))
            print('negative cases: ', len(neg_df))
        
        
            if oversampling:            
                kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
                ros = RandomOverSampler()
                X_train, y_train = ros.fit_resample(X_train, kmeans.labels_)


            # Fit the classifier
            classifier = svm.SVC().fit(X_train, y_train)
#             classifier = RandomForestClassifier().fit(X_train, y_train)

            # Predict the labels of the test set samples
            y_pred = classifier.predict(X_test)
            print(y_pred)
            
            print(type(y_test))
            print(type(y_pred))
            
            for yt in y_test.values.ravel():
                y_test_agg_list.append(yt)
            
            for yp in y_pred.tolist():
                y_pred_agg_list.append(yp)
            
            
#             y_test_agg_list.append(y_test.values.ravel())
#             y_pred_agg_list.append(y_pred.tolist())
            
#             tp, fp, tn, fn = perf_measure(y_test.values.ravel(), y_pred.tolist())
            
#             tp_list.append(tp)
#             fp_list.append(fp)
#             tn_list.append(tn)
#             fn_list.append(fn)

#             predicted_targets = np.append(predicted_targets, predicted_labels)
#             actual_targets = np.append(actual_targets, y_test)
    
#     tp_avg = sum(tp_list) / len(tp_list)
#     fp_avg = sum(fp_list) / len(fp_list)
#     tn_avg = sum(tn_list) / len(tn_list)
#     fn_avg = sum(fn_list) / len(fn_list)
    
#     y_test = aggregate_labels(y_test_agg_list)
#     print(y_test)
#     y_pred = aggregate_labels(y_pred_agg_list)
#     print(y_pred)
    
    return y_pred_agg_list, y_test_agg_list
    
#     return tp_avg, fp_avg, tn_avg, fn_avg


def kNNUndersampling(X_train, X_test, y_train, y_test):
    # define the undersampling method
    print('UNDERSAMPLING: ')
    undersample = CondensedNearestNeighbour(n_neighbors=1)
    X_train, y_train = undersample.fit_resample(X_train, y_train)
    scatter_plot(X_train, y_train)
    model_evaluation(X_train, X_test, y_train, y_test)


def brSMOTE(X_train, X_test, y_train, y_test):
    # borderline SMOTE
    from imblearn.over_sampling import SMOTE, KMeansSMOTE
    sm = BorderlineSMOTE()
    # sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(y_train['stroke'].value_counts())
    scatter_plot(X_train, y_train)
    model_evaluation(X_train, X_test, y_train, y_test)

    # remove TomekLinks
    tl = TomekLinks(sampling_strategy='auto')
    X_train, y_train = tl.fit_resample(X_train, y_train)
    print(y_train['stroke'].value_counts())
    scatter_plot(X_train, y_train)
    model_evaluation(X_train, X_test, y_train, y_test)


def adaptiveSynthetic(X_train, X_test, y_train, y_test):
    ada = ADASYN()
    X_train, y_train = ada.fit_resample(X_train, y_train)
    scatter_plot(X_train, y_train)
    model_evaluation(X_train, X_test, y_train, y_test)


def kMeansRos(X_train, X_test, y_train, y_test):
    kmeans = KMeans(n_clusters=2).fit(X_train)
    y_train = kmeans.labels_

    print(kmeans.cluster_centers_)

    countPos = np.count_nonzero(y_train == 0)
    print(countPos)
    countNeg = np.count_nonzero(y_train == 1)
    print(countNeg)


    if countPos < countNeg:
        indices_one = y_train == 1
        indices_zero = y_train == 0
        y_train[indices_one] = 0  # replacing 1s with 0s
        y_train[indices_zero] = 1  # replacing 0s with 1s

    scatter_plot(X_train, y_train)


    ros = RandomOverSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    model_evaluation(X_train, X_test, y_train, y_test)


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

    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    print(df.describe())

    print("Total number of entries: ", len(df))

    # positive/negative
    pos_df = df[df["stroke"] == 1]
    neg_df = df[df["stroke"] == 0]

    # print length
    print('positive cases: ', len(pos_df))
    print('positive cases: ', len(neg_df))

    # Drop id column & drop Missing values
    df = df.drop(['id'], axis=1)
    print(df.isnull().sum())
    df = df.dropna()
    print("After removal of missing values: ", len(df))

    # Separate features and ground truth label
    # X = df.drop(['stroke'], axis=1)
    # y = df[['stroke']]

    categorical_features = get_categorical_features(df)
    print('categorical features: ', categorical_features)

    # one hot encoding for categorical features
    for feature in categorical_features:
        df = one_hot_encoding(df, feature)


    print(df.columns)


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

    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)

    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)


    # pipeline = Pipeline(steps=[('scaler', scaler), ('imb', smote), ('clf', clf)])
    folds = StratifiedKFold(n_splits=10, shuffle=True)
    # n_scores = cross_validate(pipeline, X, y, scoring=['accuracy', 'balanced_accuracy', 'roc_auc'], cv=folds, n_jobs=-1)
    # print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores['test_accuracy']), np.std(n_scores['test_accuracy'])))
    # print('Balanced: %.3f (%.3f)' % (np.mean(n_scores['test_balanced_accuracy']), np.std(n_scores['test_balanced_accuracy'])))
    # print('AUC: %.3f (%.3f)' % (np.mean(n_scores['test_roc_auc']), np.std(n_scores['test_roc_auc'])))

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

    from sklearn.metrics import auc
    from sklearn.metrics import plot_roc_curve

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
            scatter_plot(X_train, y_train)

        sm = BorderlineSMOTE()
        X_train, y_train = sm.fit_resample(X_train, y_train)

        if i == 0:
            scatter_plot(X_train, y_train)

        # clf = SVC()
        clf = RandomForestClassifier(max_depth=3)

        clf.fit(X_train, y_train)
        # scatter_plot(X.iloc[train], y.iloc[train])
        viz = plot_roc_curve(clf, X_test, y_test,
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=axROC)

        prc = plot_precision_recall_curve(clf, X_test, y_test,
                                           name='PR fold {}'.format(i),
                                           alpha=0.3, lw=1, ax=axPR, pos_label=1)

        # print(prc.recall)
        # aucs2.append(auc(pr, rec))

        precInterp = np.interp(mean_rec, prc.recall, prc.precision)
        precInterp[0] = 1.0
        # precs.append(prc.precision)
        precs.append(precInterp)
        ap.append(prc.average_precision)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        # aucs.append(auc(viz.fpr, viz.tpr))

        y_pred = clf.predict(X_test)
        acc.append(accuracy_score(y_test, y_pred))
        bacc.append(balanced_accuracy_score(y_test, y_pred))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    axROC.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    mean_prec = np.mean(precs, axis=0)
    print(mean_prec)
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
           title="Receiver operating characteristic example")

    axPR.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic example")

    axROC.legend(loc="lower right")
    figROC.savefig('ROC_CV.png')
    figROC.show()

    axPR.legend(loc="upper right")
    figPR.savefig('PR_CV.png')
    figPR.show()

    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(acc), np.std(acc)))
    print('Mean Balanced Accuracy: %.3f (%.3f)' % (np.mean(bacc), np.std(bacc)))
    print('AUC %.3f (%.3f)' % (np.mean(aucs), np.std(aucs)))

    # print(y_train['stroke'].value_counts())
    # print(y_test['stroke'].value_counts())

    # print('NO PREPROCESSING: ')
    # scatter_plot(X_train, y_train)
    # # scatter2D(X, y)
    # print(y_train['stroke'].value_counts())
    # model_evaluation(X_train, X_test, y_train, y_test)



    # k-fold cross-validation
    # y_pred, y_test = evaluate_model(X, y, True)

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



